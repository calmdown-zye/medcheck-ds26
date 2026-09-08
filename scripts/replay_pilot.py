from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CASE_TEXT_KEYS = (
    "patient_case",
    "case",
    "case_text",
    "input",
    "question",
    "text",
)
CASE_ID_KEYS = ("case_id", "id", "qa_id", "patient_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="저장된 파일럿 사례를 MedCheck 파이프라인으로 다시 실행합니다."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="사례가 저장된 JSON 또는 JSONL 파일",
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot_replay.jsonl"),
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=Path("results/pilot_replay_failures.jsonl"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="출력 파일에 이미 기록된 case_id는 건너뜁니다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="처리할 최대 사례 수. 시험 실행에 사용합니다.",
    )
    return parser.parse_args()


def _as_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("JSON 최상위 값은 객체 또는 배열이어야 합니다.")

    for key in ("cases", "records", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return [payload]


def load_records(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    if path.suffix.lower() == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"JSONL {line_number}번째 줄을 읽을 수 없습니다: {exc}"
                    ) from exc
        return records

    with path.open("r", encoding="utf-8") as file:
        return _as_records(json.load(file))


def extract_case_text(record: Any) -> str:
    if isinstance(record, str):
        return record.strip()
    if not isinstance(record, dict):
        return ""

    for key in CASE_TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_case_id(record: Any, position: int) -> str:
    if isinstance(record, dict):
        for key in CASE_ID_KEYS:
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return f"case-{position:06d}"


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = payload.get("case_id")
            if case_id is not None:
                completed.add(str(case_id))
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()


def summarize_result(result: Any) -> dict[str, Any]:
    decision = result.decision
    valid_ids = {item.evidence_id for item in result.evidence}
    cited_ids = {
        evidence_id
        for diagnosis in decision.differential_diagnoses
        for evidence_id in diagnosis.evidence_ids
    }
    invalid_ids = sorted(cited_ids - valid_ids)
    incomplete = [
        diagnosis.name
        for diagnosis in decision.differential_diagnoses
        if (
            not diagnosis.supporting_evidence
            or not diagnosis.contradicting_or_missing_evidence
            or not diagnosis.recommended_checks
            or not diagnosis.evidence_ids
        )
    ]

    return {
        "diagnosis_count": len(decision.differential_diagnoses),
        "diagnosis_names": [
            diagnosis.name for diagnosis in decision.differential_diagnoses
        ],
        "evidence_count": len(result.evidence),
        "invalid_evidence_ids": invalid_ids,
        "incomplete_diagnoses": incomplete,
        "abstain": decision.abstain,
        "abstain_reason": decision.abstain_reason,
        "forced_safety_abstention": result.forced_safety_abstention,
    }


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k는 1 이상이어야 합니다.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit은 1 이상이어야 합니다.")

    records = load_records(args.input)
    completed_ids = load_completed_ids(args.output) if args.resume else set()

    from medcheck.index import MedicalHybridIndex
    from medcheck.llm import LocalTransformersClient
    from medcheck.models import PipelineResult
    from medcheck.pipeline import MedCheckPipeline

    index = MedicalHybridIndex.load(args.index_dir)
    llm = LocalTransformersClient()
    pipeline = MedCheckPipeline(index=index, llm=llm, top_k=args.top_k)

    processed = 0
    succeeded = 0
    failed = 0
    skipped = 0

    for position, record in enumerate(records, start=1):
        if args.limit is not None and processed >= args.limit:
            break

        case_id = extract_case_id(record, position)
        if case_id in completed_ids:
            skipped += 1
            continue

        patient_case = extract_case_text(record)
        processed += 1
        print("\n" + "=" * 80)
        print(f"[{position}/{len(records)}] case_id={case_id}")

        if not patient_case:
            failed += 1
            failure = {
                "case_id": case_id,
                "position": position,
                "error_type": "InvalidInput",
                "error": f"환자 사례 텍스트가 없습니다. 지원 키: {CASE_TEXT_KEYS}",
                "source_record": record,
            }
            append_jsonl(args.failures, failure)
            print("FAIL:", failure["error"])
            continue

        try:
            result = pipeline.run(
                patient_case,
                domain=args.domain,
                skip_clarification=True,
                verify=args.verify,
            )
            if not isinstance(result, PipelineResult):
                raise TypeError(
                    f"예상하지 않은 결과 타입: {type(result).__name__}"
                )
        except Exception as exc:
            failed += 1
            failure = {
                "case_id": case_id,
                "position": position,
                "patient_case": patient_case,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "source_record": record,
            }
            append_jsonl(args.failures, failure)
            print(f"FAIL: {type(exc).__name__}: {exc}")
            continue

        summary = summarize_result(result)
        output = {
            "case_id": case_id,
            "position": position,
            "patient_case": patient_case,
            "source_record": record,
            "settings": {
                "top_k": args.top_k,
                "domain": args.domain,
                "verify": args.verify,
            },
            "summary": summary,
            "result": result.model_dump(mode="json"),
        }
        append_jsonl(args.output, output)
        succeeded += 1

        print("진단:", summary["diagnosis_names"])
        print("abstain:", summary["abstain"])
        print("불완전 후보:", summary["incomplete_diagnoses"])
        print("잘못된 근거 ID:", summary["invalid_evidence_ids"])
        print("저장 완료")

    print("\n" + "=" * 80)
    print("전체 입력:", len(records))
    print("이번 실행 처리:", processed)
    print("성공:", succeeded)
    print("실패:", failed)
    print("건너뜀:", skipped)
    print("결과 파일:", args.output)
    print("실패 파일:", args.failures)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
