from __future__ import annotations

from dataclasses import dataclass
import unittest

from medcheck.models import (
    ClinicalDecision,
    DifferentialDiagnosis,
    EvidenceItem,
    PipelineResult,
)
from medcheck.pipeline import MedCheckPipeline


@dataclass(frozen=True)
class DamageCase:
    """검색 문서에서 현재 환자에게 잘못 옮겨지면 안 되는 사실."""

    name: str
    patient_case: str
    forbidden_patient_facts: tuple[str, ...]


DAMAGE_CASES = (
    DamageCase(
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
    DamageCase(
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
    DamageCase(
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
)


def _joined_supporting_evidence(result: PipelineResult) -> str:
    return "\n".join(
        evidence
        for diagnosis in result.decision.differential_diagnoses
        for evidence in diagnosis.supporting_evidence
    )


def find_patient_fact_leaks(
    result: PipelineResult,
    damage_case: DamageCase,
) -> list[str]:
    """이미 관찰된 검색문서→환자사실 혼입 표현을 찾습니다."""

    supporting_text = _joined_supporting_evidence(result).casefold()
    return [
        phrase
        for phrase in damage_case.forbidden_patient_facts
        if phrase.casefold() in supporting_text
    ]


def find_invalid_citations(result: PipelineResult) -> list[str]:
    """검색 결과에 실제로 존재하지 않는 인용 ID를 반환합니다."""

    valid_ids = {item.evidence_id for item in result.evidence}
    cited_ids = {
        evidence_id
        for diagnosis in result.decision.differential_diagnoses
        for evidence_id in diagnosis.evidence_ids
    }
    cited_ids.update(
        evidence_id
        for step in result.decision.next_steps
        for evidence_id in step.evidence_ids
    )
    cited_ids.update(
        evidence_id
        for flag in result.decision.red_flags
        for evidence_id in flag.evidence_ids
    )
    return sorted(cited_ids - valid_ids)


def _evidence(evidence_id: str = "E1") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        qa_id="fixture-1",
        domain="테스트",
        q_type="서술형",
        question="테스트 질문",
        answer="테스트 답변",
        source_file="fixture.json",
        score=1.0,
    )


def _diagnosis(
    *,
    supporting_evidence: list[str] | None = None,
    evidence_ids: list[str] | None = None,
) -> DifferentialDiagnosis:
    return DifferentialDiagnosis(
        rank=1,
        name="테스트 진단",
        supporting_evidence=supporting_evidence or ["환자 사례에 있는 증상"],
        contradicting_or_missing_evidence=["추가 확인이 필요한 정보"],
        recommended_checks=["추가 평가"],
        evidence_ids=evidence_ids or ["E1"],
    )


def _result(
    *,
    diagnosis: DifferentialDiagnosis | None = None,
    abstain: bool = False,
    forced_safety_abstention: bool = False,
) -> PipelineResult:
    diagnoses = [] if diagnosis is None else [diagnosis]
    return PipelineResult(
        decision=ClinicalDecision(
            case_summary="테스트 사례",
            differential_diagnoses=diagnoses,
            abstain=abstain,
            abstain_reason="테스트 보류" if abstain else "",
        ),
        evidence=[_evidence()],
        forced_safety_abstention=forced_safety_abstention,
    )


class _FakeIndex:
    def search(self, *_args: object, **_kwargs: object) -> list[EvidenceItem]:
        return [_evidence()]


class _FakeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def chat_json(self, **_kwargs: object) -> dict[str, object]:
        return self.payload


class DamageRegressionTests(unittest.TestCase):
    def test_known_search_document_facts_are_detected(self) -> None:
        for damage_case in DAMAGE_CASES:
            with self.subTest(case=damage_case.name):
                leaked_phrase = damage_case.forbidden_patient_facts[0]
                result = _result(
                    diagnosis=_diagnosis(supporting_evidence=[leaked_phrase]),
                )

                detected = find_patient_fact_leaks(result, damage_case)
                self.assertIn(leaked_phrase, detected)

    def test_clean_patient_evidence_is_not_flagged(self) -> None:
        for damage_case in DAMAGE_CASES:
            with self.subTest(case=damage_case.name):
                result = _result(
                    diagnosis=_diagnosis(
                        supporting_evidence=[damage_case.patient_case]
                    ),
                )

                self.assertEqual(
                    find_patient_fact_leaks(result, damage_case),
                    [],
                )

    def test_unknown_evidence_id_is_detected(self) -> None:
        result = _result(diagnosis=_diagnosis(evidence_ids=["E999"]))

        self.assertEqual(find_invalid_citations(result), ["E999"])

    def test_existing_evidence_id_is_accepted(self) -> None:
        result = _result(diagnosis=_diagnosis(evidence_ids=["E1"]))

        self.assertEqual(find_invalid_citations(result), [])

    def test_non_forced_abstention_keeps_diagnoses_for_audit(self) -> None:
        result = _result(diagnosis=_diagnosis(), abstain=True)

        self.assertTrue(result.decision.differential_diagnoses)
        self.assertFalse(result.forced_safety_abstention)

    def test_forced_safety_abstention_may_have_no_diagnoses(self) -> None:
        result = _result(
            diagnosis=None,
            abstain=True,
            forced_safety_abstention=True,
        )

        self.assertEqual(result.decision.differential_diagnoses, [])
        self.assertTrue(result.forced_safety_abstention)

    def test_pipeline_keeps_diagnosis_when_citation_guard_abstains(self) -> None:
        payload = {
            "case_summary": "요통 사례",
            "differential_diagnoses": [
                {
                    "rank": 1,
                    "name": "요추 질환",
                    "supporting_evidence": ["요통"],
                    "contradicting_or_missing_evidence": ["영상검사 미시행"],
                    "recommended_checks": ["신경학적 진찰"],
                    "evidence_ids": [],
                }
            ],
            "next_steps": [],
            "abstain": False,
        }
        pipeline = MedCheckPipeline(index=_FakeIndex(), llm=_FakeLLM(payload), top_k=1)

        result = pipeline.run(
            "60세 남성이 요통과 다리 저림을 호소한다.",
            skip_clarification=True,
            verify=False,
        )

        self.assertTrue(result.decision.abstain)
        self.assertEqual(len(result.decision.differential_diagnoses), 1)

    def test_pipeline_abstains_when_only_next_steps_exist(self) -> None:
        payload = {
            "case_summary": "두통 사례",
            "differential_diagnoses": [],
            "next_steps": [
                {
                    "action": "의료진 평가",
                    "rationale": "원인 확인",
                    "urgency": "일반",
                    "evidence_ids": ["E1"],
                }
            ],
            "abstain": False,
        }
        pipeline = MedCheckPipeline(index=_FakeIndex(), llm=_FakeLLM(payload), top_k=1)

        result = pipeline.run(
            "42세 여성이 갑작스러운 심한 두통을 호소한다.",
            skip_clarification=True,
            verify=False,
        )

        self.assertTrue(result.decision.abstain)
        self.assertIn("감별진단", result.decision.abstain_reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
