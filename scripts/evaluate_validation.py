from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medcheck.data import load_dataset
from medcheck.evaluation import (
    extract_option_number,
    rouge_l_f1,
    token_f1,
)
from medcheck.index import MedicalHybridIndex
from medcheck.llm import LocalTransformersClient


SYSTEM_PROMPT = """
당신은 한국어 전문 의학지식 평가 문항에 답하는 모델이다.

규칙:
- 제공된 검색 근거를 참고하여 문제에 답한다.
- 객관식 문제라면 선택지 번호와 선택지 내용을 함께 작성한다.
- 근거가 부족하면 임의로 확신하지 않는다.
- 불필요한 설명을 길게 작성하지 않는다.
- 반드시 JSON 객체 하나만 출력한다.
"""


class EvaluationResponse(BaseModel):
    answer: str = ""
    abstain: bool = False
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


def build_prompt(
    question: str,
    evidence_text: str,
) -> str:
    return f"""
[문제]
{question}

[검색 근거]
{evidence_text}

[출력 규칙]
- answer에는 최종 답변을 작성한다.
- 객관식이면 반드시 선택지 번호와 선택지 내용을 함께 작성한다.
- 검색 근거만으로 판단하기 어렵다면 abstain=true로 설정한다.
- 사용한 검색 근거의 ID를 evidence_ids에 작성한다.
- JSON 이외의 텍스트를 출력하지 않는다.

[출력 예시]
{{
  "answer": "3) 급성 심근경색",
  "abstain": false,
  "reason": "흉통, 방사통, ST 분절 상승이 제시되었다.",
  "evidence_ids": ["E1"]
}}
"""


def normalize_response(
    raw: dict[str, Any],
) -> dict[str, Any]:
    data = dict(raw)

    data.setdefault(
        "answer",
        data.get("prediction", data.get("response", "")),
    )
    data.setdefault("abstain", False)
    data.setdefault("reason", "")
    data.setdefault("evidence_ids", [])

    if not isinstance(data["evidence_ids"], list):
        data["evidence_ids"] = [data["evidence_ids"]]

    data["answer"] = str(data["answer"]).strip()
    data["reason"] = str(data["reason"]).strip()
    data["evidence_ids"] = [
        str(item).strip()
        for item in data["evidence_ids"]
        if str(item).strip()
    ]

    return data


def select_validation_records(
    records: list[Any],
    *,
    limit: int,
    seed: int,
    shuffle: bool,
) -> list[Any]:
    selected = list(records)

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)

    if limit > 0:
        selected = selected[:limit]

    return selected


def retrieve_without_leakage(
    index: MedicalHybridIndex,
    *,
    question: str,
    validation_qa_id: str,
    top_k: int,
) -> list[Any]:
    # 여유 있게 검색한 뒤 동일 qa_id를 제거합니다.
    candidates = index.search(
        question,
        top_k=max(top_k * 5, top_k),
    )

    filtered = [
        item
        for item in candidates
        if str(item.qa_id) != str(validation_qa_id)
    ]

    return filtered[:top_k]


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None

    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index-dir",
        required=True,
    )
    parser.add_argument(
        "--validation-root",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="0이면 전체 Validation 데이터를 평가합니다.",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Validation 데이터를 섞지 않고 앞에서부터 평가합니다.",
    )

    args = parser.parse_args()

    validation, load_errors = load_dataset(
        args.validation_root
    )

    validation = select_validation_records(
        validation,
        limit=args.limit,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    index = MedicalHybridIndex.load(
        args.index_dir
    )
    llm = LocalTransformersClient()

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    mcq_total = 0
    mcq_correct = 0

    subjective_total = 0
    subjective_f1_scores: list[float] = []
    subjective_rouge_scores: list[float] = []

    all_f1_scores: list[float] = []
    all_rouge_scores: list[float] = []

    abstentions = 0
    failures = 0
    success_count = 0

    success_latencies: list[float] = []
    failure_latencies: list[float] = []

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for index_number, record in enumerate(
            validation,
            start=1,
        ):
            started = time.perf_counter()

            try:
                evidence = retrieve_without_leakage(
                    index,
                    question=record.question,
                    validation_qa_id=record.qa_id,
                    top_k=args.top_k,
                )

                if not evidence:
                    raise RuntimeError(
                        "동일 qa_id를 제외한 검색 근거가 없습니다."
                    )

                evidence_text = "\n\n".join(
                    (
                        f"[{item.evidence_id}]\n"
                        f"질문: {item.question}\n"
                        f"답변: {item.answer}"
                    )
                    for item in evidence
                )

                raw_response = llm.chat_json(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=build_prompt(
                        record.question,
                        evidence_text,
                    ),
                    temperature=0.0,
                    json_schema=(
                        EvaluationResponse.model_json_schema()
                    ),
                )

                normalized_response = normalize_response(
                    raw_response
                )

                try:
                    response = (
                        EvaluationResponse.model_validate(
                            normalized_response
                        )
                    )
                except ValidationError as exc:
                    raise RuntimeError(
                        "EvaluationResponse 검증 실패: "
                        f"{exc}\n"
                        f"보정 출력: {normalized_response}"
                    ) from exc

                valid_evidence_ids = {
                    item.evidence_id
                    for item in evidence
                }

                response.evidence_ids = [
                    evidence_id
                    for evidence_id
                    in response.evidence_ids
                    if evidence_id in valid_evidence_ids
                ]

                prediction = response.answer
                abstain = response.abstain

                abstentions += int(abstain)
                success_count += 1

                gold_option = extract_option_number(
                    record.answer
                )
                pred_option = extract_option_number(
                    prediction
                )

                mcq_match: bool | None = None

                if record.q_type_code == "1":
                    mcq_total += 1

                    mcq_match = (
                        gold_option is not None
                        and pred_option is not None
                        and gold_option == pred_option
                    )

                    mcq_correct += int(mcq_match)

                score_f1 = token_f1(
                    prediction,
                    record.answer,
                )
                score_rouge = rouge_l_f1(
                    prediction,
                    record.answer,
                )

                all_f1_scores.append(score_f1)
                all_rouge_scores.append(score_rouge)

                if record.q_type_code != "1":
                    subjective_total += 1
                    subjective_f1_scores.append(
                        score_f1
                    )
                    subjective_rouge_scores.append(
                        score_rouge
                    )

                latency = (
                    time.perf_counter()
                    - started
                )
                success_latencies.append(latency)

                row = {
                    "qa_id": record.qa_id,
                    "domain": record.domain_name,
                    "q_type": record.q_type_name,
                    "q_type_code": record.q_type_code,
                    "question": record.question,
                    "gold": record.answer,
                    "prediction": prediction,
                    "reason": response.reason,
                    "abstain": abstain,
                    "mcq_gold_option": gold_option,
                    "mcq_pred_option": pred_option,
                    "mcq_match": mcq_match,
                    "token_f1": score_f1,
                    "rouge_l_f1": score_rouge,
                    "retrieved_evidence": [
                        {
                            "evidence_id": item.evidence_id,
                            "qa_id": item.qa_id,
                            "domain": item.domain,
                            "score": item.score,
                        }
                        for item in evidence
                    ],
                    "cited_evidence_ids": (
                        response.evidence_ids
                    ),
                    "latency_sec": latency,
                    "error": None,
                }

            except Exception as exc:
                failures += 1

                latency = (
                    time.perf_counter()
                    - started
                )
                failure_latencies.append(latency)

                row = {
                    "qa_id": record.qa_id,
                    "domain": getattr(
                        record,
                        "domain_name",
                        None,
                    ),
                    "q_type": getattr(
                        record,
                        "q_type_name",
                        None,
                    ),
                    "question": record.question,
                    "gold": record.answer,
                    "prediction": None,
                    "abstain": None,
                    "mcq_match": None,
                    "token_f1": None,
                    "rouge_l_f1": None,
                    "retrieved_evidence": [],
                    "cited_evidence_ids": [],
                    "latency_sec": latency,
                    "error": str(exc),
                }

            rows.append(row)

            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()

            status = (
                "FAIL"
                if row["error"]
                else "OK"
            )

            print(
                f"[{index_number}/{len(validation)}] "
                f"qa_id={record.qa_id} "
                f"status={status} "
                f"latency={row['latency_sec']:.2f}s"
            )

    summary = {
        "num_validation_records": len(validation),
        "num_successful_inferences": success_count,
        "load_errors": len(load_errors),
        "inference_failures": failures,
        "failure_rate": (
            failures / len(validation)
            if validation
            else None
        ),
        "mcq_accuracy": (
            mcq_correct / mcq_total
            if mcq_total
            else None
        ),
        "mcq_correct": mcq_correct,
        "mcq_total": mcq_total,
        "subjective_total": subjective_total,
        "subjective_mean_token_f1": safe_mean(
            subjective_f1_scores
        ),
        "subjective_mean_rouge_l_f1": safe_mean(
            subjective_rouge_scores
        ),
        "all_mean_token_f1": safe_mean(
            all_f1_scores
        ),
        "all_mean_rouge_l_f1": safe_mean(
            all_rouge_scores
        ),
        "abstention_rate_over_success": (
            abstentions / success_count
            if success_count
            else None
        ),
        "mean_success_latency_sec": safe_mean(
            success_latencies
        ),
        "mean_failure_latency_sec": safe_mean(
            failure_latencies
        ),
        "top_k": args.top_k,
        "seed": args.seed,
        "shuffled": not args.no_shuffle,
        "prediction_file": str(output_path),
    }

    summary_path = output_path.with_suffix(
        ".summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
