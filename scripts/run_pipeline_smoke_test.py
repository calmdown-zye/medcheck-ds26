from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from medcheck.models import PipelineResult


@dataclass(frozen=True)
class SmokeCase:
    name: str
    patient_case: str
    forbidden_patient_facts: tuple[str, ...] = ()


SMOKE_CASES = (
    SmokeCase(
        name="피부 발진",
        patient_case=(
            "35세 여성이 새로운 화장품을 사용한 뒤 얼굴과 목에 "
            "가려운 붉은 발진이 발생하였다. 발열과 호흡곤란은 없다."
        ),
        forbidden_patient_facts=(
            "물로 세안",
            "오일 클렌저",
            "식단에 변화는 없",
            "야외 활동",
        ),
    ),
    SmokeCase(
        name="발목 외상",
        patient_case=(
            "28세 남성이 계단에서 발목을 접질린 뒤 외측 발목의 "
            "통증과 부종이 발생하였다. 체중을 싣고 걷기 어렵다."
        ),
        forbidden_patient_facts=(
            "외측에서 심한 압통",
            "발열은 없",
            "톰슨 검사",
            "수동적 발바닥 굽힘",
        ),
    ),
    SmokeCase(
        name="요통",
        patient_case=(
            "60세 남성이 6개월 전부터 지속된 요통과 오른쪽 다리 "
            "뒤쪽의 저림으로 내원하였다. 보행할 때 악화되고 휴식하면 "
            "완화된다. 배뇨 장애와 근력 약화는 없으며 MRI는 시행하지 않았다."
        ),
        forbidden_patient_facts=(
            "당뇨병과 고혈압 병력",
            "당뇨병 병력",
            "고혈압 병력",
            "HbA1c",
            "당화혈색소",
            "발목 반사의 감소",
        ),
    ),
    SmokeCase(
        name="갑작스러운 두통",
        patient_case=(
            "42세 여성이 한 시간 전 갑자기 시작된 매우 심한 두통과 "
            "구토를 호소한다. 이전에 이와 같은 두통은 없었다."
        ),
    ),
    SmokeCase(
        name="우하복부 통증",
        patient_case=(
            "24세 남성이 하루 전부터 시작되어 점차 심해지는 우하복부 "
            "통증과 식욕 저하를 호소한다. 설사는 없고 체온은 측정하지 않았다."
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MedCheck 실제 파이프라인의 소규모 회귀 검사를 실행합니다."
    )
    parser.add_argument("--index-dir", required=True, help="저장된 검색 인덱스 경로")
    parser.add_argument("--top-k", type=int, default=3, help="검색 근거 개수")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="검증기까지 호출합니다. 지정하지 않으면 최초 생성만 검사합니다.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="경고가 하나라도 있으면 종료 코드 1을 반환합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="각 사례의 전체 PipelineResult를 JSON 파일로 저장합니다.",
    )
    return parser.parse_args()


def collect_result_issues(
    result: PipelineResult,
    smoke_case: SmokeCase,
) -> tuple[list[str], list[str]]:
    """실행을 실패시킬 오류와 사람이 검토할 경고를 구분합니다."""

    errors: list[str] = []
    warnings: list[str] = []
    decision = result.decision

    valid_ids = {item.evidence_id for item in result.evidence}
    cited_ids = {
        evidence_id
        for diagnosis in decision.differential_diagnoses
        for evidence_id in diagnosis.evidence_ids
    }
    cited_ids.update(
        evidence_id
        for step in decision.next_steps
        for evidence_id in step.evidence_ids
    )
    cited_ids.update(
        evidence_id
        for flag in decision.red_flags
        for evidence_id in flag.evidence_ids
    )

    invalid_ids = sorted(cited_ids - valid_ids)
    if invalid_ids:
        errors.append(f"검색 결과에 없는 evidence_ids: {invalid_ids}")

    if not result.evidence:
        errors.append("검색 근거가 0개입니다.")

    if not decision.differential_diagnoses:
        warnings.append("감별진단이 생성되지 않았습니다.")

    supporting_text = "\n".join(
        fact
        for diagnosis in decision.differential_diagnoses
        for fact in diagnosis.supporting_evidence
    ).casefold()
    leaked_facts = [
        fact
        for fact in smoke_case.forbidden_patient_facts
        if fact.casefold() in supporting_text
    ]
    if leaked_facts:
        warnings.append(f"알려진 환자정보 혼입 표현: {leaked_facts}")

    for diagnosis in decision.differential_diagnoses:
        empty_fields = [
            field_name
            for field_name, value in (
                ("supporting_evidence", diagnosis.supporting_evidence),
                (
                    "contradicting_or_missing_evidence",
                    diagnosis.contradicting_or_missing_evidence,
                ),
                ("recommended_checks", diagnosis.recommended_checks),
                ("evidence_ids", diagnosis.evidence_ids),
            )
            if not value
        ]
        if empty_fields:
            warnings.append(
                f"{diagnosis.name}: 빈 필드 {', '.join(empty_fields)}"
            )

    if decision.abstain:
        warnings.append(f"판단 보류: {decision.abstain_reason or '사유 없음'}")

    if result.forced_safety_abstention and decision.differential_diagnoses:
        errors.append("강제 안전 보류인데 감별진단이 남아 있습니다.")

    return errors, warnings


def print_case_result(
    smoke_case: SmokeCase,
    result: PipelineResult,
    errors: list[str],
    warnings: list[str],
) -> None:
    print("\n" + "=" * 80)
    print("사례:", smoke_case.name)
    print("진단:", [item.name for item in result.decision.differential_diagnoses])
    print("abstain:", result.decision.abstain)
    print("reason:", result.decision.abstain_reason)
    print("검색 근거 ID:", [item.evidence_id for item in result.evidence])

    for diagnosis in result.decision.differential_diagnoses:
        print("-" * 70)
        print("진단명:", diagnosis.name)
        print("supporting_evidence:", diagnosis.supporting_evidence)
        print(
            "contradicting_or_missing_evidence:",
            diagnosis.contradicting_or_missing_evidence,
        )
        print("recommended_checks:", diagnosis.recommended_checks)
        print("evidence_ids:", diagnosis.evidence_ids)

    for message in errors:
        print("ERROR:", message)
    for message in warnings:
        print("WARN:", message)

    if errors:
        print("판정: FAIL")
    elif warnings:
        print("판정: REVIEW")
    else:
        print("판정: PASS")


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k는 1 이상이어야 합니다.")

    from medcheck.index import MedicalHybridIndex
    from medcheck.llm import LocalTransformersClient
    from medcheck.models import PipelineResult
    from medcheck.pipeline import MedCheckPipeline

    index = MedicalHybridIndex.load(args.index_dir)
    llm = LocalTransformersClient()
    pipeline = MedCheckPipeline(index=index, llm=llm, top_k=args.top_k)

    total_errors = 0
    total_warnings = 0
    serialized_results: list[dict[str, object]] = []

    for smoke_case in SMOKE_CASES:
        try:
            result = pipeline.run(
                smoke_case.patient_case,
                skip_clarification=True,
                verify=args.verify,
            )
        except Exception as exc:  # 개별 사례 실패 후에도 나머지를 계속 검사합니다.
            total_errors += 1
            print("\n" + "=" * 80)
            print("사례:", smoke_case.name)
            print(f"ERROR: {type(exc).__name__}: {exc}")
            print("판정: FAIL")
            continue

        if not isinstance(result, PipelineResult):
            total_errors += 1
            print("\n" + "=" * 80)
            print("사례:", smoke_case.name)
            print(f"ERROR: 예상하지 않은 결과 타입: {type(result).__name__}")
            print("판정: FAIL")
            continue

        errors, warnings = collect_result_issues(result, smoke_case)
        total_errors += len(errors)
        total_warnings += len(warnings)
        print_case_result(smoke_case, result, errors, warnings)
        serialized_results.append(
            {
                "case_name": smoke_case.name,
                "patient_case": smoke_case.patient_case,
                "errors": errors,
                "warnings": warnings,
                "result": result.model_dump(mode="json"),
            }
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(serialized_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\n결과 저장:", args.output)

    print("\n" + "=" * 80)
    print("총 사례:", len(SMOKE_CASES))
    print("오류:", total_errors)
    print("검토 필요:", total_warnings)

    if total_errors or (args.strict and total_warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
