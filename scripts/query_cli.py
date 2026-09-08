from __future__ import annotations

import sys
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from medcheck.models import (
    ClarificationAssessment,
    PipelineResult,
    TextDemoResult,
)

from medcheck.demo_modes import (
    run_base_llm,
    run_rag_llm,
)
from medcheck.models import ClarificationAssessment
from medcheck.index import MedicalHybridIndex
from medcheck.llm import LocalTransformersClient
from medcheck.pipeline import MedCheckPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--mode",
        choices=[
            "base",
            "rag",
            "structured",
            "medcheck",
        ],
        default="medcheck",
        help=(
            "base: 검색 없는 기본 LLM, "
            "rag: 자유형 RAG, "
            "structured: 구조화 RAG, "
            "medcheck: 구조화 RAG와 검증기"
        ),
    )
    args = parser.parse_args()

    index = MedicalHybridIndex.load(args.index_dir)

    if args.retrieval_only:
        evidence = index.search(
            args.case,
            domain=args.domain,
            top_k=args.top_k,
        )
        print(
            json.dumps(
                [item.model_dump() for item in evidence],
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    llm = LocalTransformersClient()
    
 
    if args.mode == "base":
        result = run_base_llm(
            llm=llm,
            patient_case=args.case,
        )

    elif args.mode == "rag":
        result = run_rag_llm(
            index=index,
            llm=llm,
            patient_case=args.case,
            domain=args.domain,
            top_k=args.top_k,
        )

    elif args.mode == "structured":
        pipeline = MedCheckPipeline(
            index,
            llm,
            top_k=args.top_k,
        )

        result = pipeline.run(
            args.case,
            domain=args.domain,
            verify=False,
            # 4개 모드 비교에서는 Structured RAG가
            # 추가 질문 단계를 사용하지 않도록 설정
            skip_clarification=True,
        )

    elif args.mode == "medcheck":
        pipeline = MedCheckPipeline(
            index,
            llm,
            top_k=args.top_k,
        )

        result = pipeline.run(
            args.case,
            domain=args.domain,
            verify=True,
        )

        if isinstance(
            result,
            ClarificationAssessment,
        ):
            print(
                "\n추가 확인이 필요한 정보가 있습니다."
            )

            if result.urgent:
                print("\n[긴급 경고]")
                print(result.urgent_message)

            answers: dict[str, str] = {}

            for question in result.questions:
                print(
                    f"\n[{question.priority}] "
                    f"{question.question}"
                )
                print(
                    f"질문 이유: {question.why_needed}"
                )

                if question.answer_type == "yes_no":
                    answer = input(
                        "답변 [예/아니오/모름]: "
                    ).strip()

                elif question.answer_type == "choice":
                    if question.options:
                        print(
                            "선택지:",
                            ", ".join(
                                question.options
                            ),
                        )
                    answer = input("답변: ").strip()

                else:
                    answer = input("답변: ").strip()

                answers[
                    question.question_id
                ] = answer

            clarification = result

            result = pipeline.run(
                args.case,
                domain=args.domain,
                verify=True,
                clarification=clarification,
                clarification_answers=answers,
            )

    else:
        raise ValueError(
            f"지원하지 않는 모드입니다: {args.mode}"
        )

    if isinstance(result, TextDemoResult):
        print(result.answer)

    elif isinstance(result, PipelineResult):
        print(
            result.model_dump_json(
                ensure_ascii=False,
                indent=2,
            )
        )

    elif isinstance(
        result,
        ClarificationAssessment,
    ):
        print(
            result.model_dump_json(
                ensure_ascii=False,
                indent=2,
            )
        )

    else:
        raise TypeError(
            "지원하지 않는 결과 타입입니다: "
            f"{type(result).__name__}"
        )

if __name__ == "__main__":
    main()
