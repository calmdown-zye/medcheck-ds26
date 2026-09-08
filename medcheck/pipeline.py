from __future__ import annotations

import os
import re
from typing import Any

from pydantic import ValidationError

from .index import MedicalHybridIndex
from .llm import LocalTransformersClient
from .models import (
    ClarificationAssessment,
    ClinicalDecision,
    EvidenceItem,
    PipelineResult,
    VerificationResult,
)

from .prompts import (
    CLARIFICATION_SYSTEM_PROMPT,
    DECISION_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    build_clarification_prompt,
    build_decision_prompt,
    build_verification_prompt,
)

from .safety import filter_evidence_ids, requires_forced_abstention


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _as_list(value) if str(item).strip()]


def _normalize_decision_output(
    raw: dict[str, Any],
    *,
    patient_case: str,
) -> dict[str, Any]:
    """소형 로컬 LLM의 흔한 필드명/자료형 변형을 보정합니다."""
    data = dict(raw)
    data.setdefault("case_summary", data.get("summary", patient_case))
    if not isinstance(data.get("structured_case"), dict):
        data["structured_case"] = {}
    data.setdefault("red_flags", [])
    data.setdefault("missing_information", [])
    data.setdefault("abstain", False)
    data.setdefault("abstain_reason", "")
    data.setdefault(
        "safety_notice",
        "본 결과는 의료진의 판단을 보조하기 위한 참고 정보이며 "
        "확정 진단 또는 처방을 대체하지 않습니다.",
    )

    differentials: list[dict[str, Any]] = []
    raw_differentials = data.get("differential_diagnoses", [])
    for rank, item in enumerate(_as_list(raw_differentials), start=1):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item.setdefault("rank", rank)
        if not item.get("name"):
            item["name"] = (
                item.get("diagnosis")
                or item.get("title")
                or f"감별진단 {rank}"
            )
        item["supporting_evidence"] = _string_list(
            item.get("supporting_evidence", item.get("support", []))
        )
        
        
        item["contradicting_or_missing_evidence"] = _string_list(
            item.get(
                "contradicting_or_missing_evidence",
                item.get("missing_evidence", item.get("limitations", [])),
            )
        )
        
        
        item["recommended_checks"] = _string_list(
            item.get("recommended_checks", item.get("checks", []))
        )
        
        
        item["evidence_ids"] = _string_list(item.get("evidence_ids", []))
        differentials.append(item)
    data["differential_diagnoses"] = differentials

    next_steps: list[dict[str, Any]] = []
    for item in _as_list(data.get("next_steps", [])):
        if isinstance(item, str):
            next_steps.append(
                {
                    "action": item,
                    "rationale": "환자 상태와 위험도를 추가 평가하기 위한 조치입니다.",
                    "urgency": "일반",
                    "evidence_ids": [],
                }
            )
            continue
        if not isinstance(item, dict):
            continue
        item = dict(item)
        item.setdefault("action", item.get("step", item.get("recommendation", "")))
        item.setdefault("rationale", item.get("reason", "임상적 추가 평가를 위해 필요합니다."))
        if item.get("urgency") not in {"즉시", "긴급", "일반", "선택"}:
            item["urgency"] = "일반"
        item["evidence_ids"] = _string_list(item.get("evidence_ids", []))
        next_steps.append(item)
    data["next_steps"] = next_steps

    return data


class MedCheckPipeline:
    def __init__(
        self,
        index: MedicalHybridIndex,
        llm: LocalTransformersClient,
        *,
        top_k: int | None = None,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> None:
        self.index = index
        self.llm = llm
        self.top_k = top_k if top_k is not None else int(os.getenv("MEDCHECK_TOP_K", "8"))
        self.dense_weight = dense_weight if dense_weight is not None else float(
            os.getenv("MEDCHECK_DENSE_WEIGHT", "0.65")
        )
        self.bm25_weight = bm25_weight if bm25_weight is not None else float(
            os.getenv("MEDCHECK_BM25_WEIGHT", "0.35")
        )
        if self.top_k <= 0:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        if self.dense_weight < 0 or self.bm25_weight < 0:
            raise ValueError("검색 가중치는 0 이상이어야 합니다.")
        if self.dense_weight + self.bm25_weight <= 0:
            raise ValueError("검색 가중치 중 하나는 0보다 커야 합니다.")

    @staticmethod
    def _has_incomplete_differential_evidence(
        decision: ClinicalDecision,
    ) -> bool:
        """감별진단의 필수 근거 필드가 비어 있는지 검사합니다.

        현재는 평가와 회귀 테스트에만 사용하는 순수 검사 함수입니다.
        결과를 변경하거나 추가 LLM 호출을 실행하지 않습니다.
        """
        return any(
            not diagnosis.supporting_evidence
            or not diagnosis.contradicting_or_missing_evidence
            or not diagnosis.recommended_checks
            for diagnosis in decision.differential_diagnoses
        )

    @staticmethod
    def _sanitize_citations(
        decision: ClinicalDecision,
        valid_ids: set[str],
    ) -> ClinicalDecision:
        for red_flag in decision.red_flags:
            red_flag.evidence_ids = filter_evidence_ids(
                red_flag.evidence_ids, valid_ids
            )
        for diagnosis in decision.differential_diagnoses:
            diagnosis.evidence_ids = filter_evidence_ids(
                diagnosis.evidence_ids, valid_ids
            )
        for step in decision.next_steps:
            step.evidence_ids = filter_evidence_ids(
                step.evidence_ids, valid_ids
            )
        return decision
    
    

    def retrieve(
        self,
        patient_case: str,
        *,
        domain: str | None = None,
    ) -> list[EvidenceItem]:
        return self.index.search(
            patient_case,
            top_k=self.top_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            domain=domain,
        )
        
    @staticmethod
    def merge_clarification_answers(
        patient_case: str,
        assessment: ClarificationAssessment,
        answers: dict[str, str],
    ) -> str:
        """
        원래 환자 사례와 추가 질문에 대한 답변을 병합합니다.
        """
        answer_lines: list[str] = []

        for question in assessment.questions:
            answer = answers.get(
                question.question_id,
                "",
            ).strip()

            if not answer:
                continue

            answer_lines.append(
                f"- 질문: {question.question}\n"
                f"  답변: {answer}"
            )

        if not answer_lines:
            return patient_case.strip()

        return (
            "[원래 환자 사례]\n"
            f"{patient_case.strip()}\n\n"
            "[추가 확인 정보]\n"
            + "\n".join(answer_lines)
        )
        
    def assess_clarification(
        self,
        patient_case: str,
    ) -> ClarificationAssessment:
        """
        감별진단을 생성하기 전에 필수 임상정보가 부족한지 판단합니다.
        이 단계에서는 의료지식 검색과 감별진단 생성을 수행하지 않습니다.
        """
        if len(patient_case.strip()) < 10:
            raise ValueError(
                "환자 사례를 10자 이상 입력하세요."
            )

        raw_assessment = self.llm.chat_json(
            system_prompt=CLARIFICATION_SYSTEM_PROMPT,
            user_prompt=build_clarification_prompt(
                patient_case
            ),
            temperature=0.0,
            json_schema=(
                ClarificationAssessment.model_json_schema()
            ),
        )

        try:
            assessment = (
                ClarificationAssessment.model_validate(
                    raw_assessment
                )
            )
        except ValidationError as exc:
            raise RuntimeError(
                "추가 질문 결과가 ClarificationAssessment "
                f"스키마를 만족하지 않습니다: {exc}\n"
                f"원본 출력: {raw_assessment}"
            ) from exc

        # 빈 질문 제거
        assessment.questions = [
            question
            for question in assessment.questions
            if question.question.strip()
        ]

        # 중복 질문 제거
        unique_questions = []
        seen: set[str] = set()

        for question in assessment.questions:
            normalized = (
                question.question.strip()
                .replace(" ", "")
                .lower()
            )

            if normalized in seen:
                continue

            seen.add(normalized)
            unique_questions.append(question)

        # 최대 3개로 제한
        assessment.questions = unique_questions[:3]

        # 필수 질문이 있다면 코드에서도 질문 필요 상태로 강제
        has_required_question = any(
            question.priority == "필수"
            for question in assessment.questions
        )

        if has_required_question:
            assessment.requires_clarification = True

        # 질문이 없으면 clarification 상태를 해제
        if not assessment.questions:
            assessment.requires_clarification = False

        return assessment
    
    def _run_decision(
        self,
        patient_case: str,
        *,
        domain: str | None = None,
        verify: bool = True,
    ) -> PipelineResult:
        """
        추가 질문이 완료된 환자 사례에 대해
        RAG, 감별진단 생성 및 검증을 수행합니다.
        """
        evidence = self.retrieve(
            patient_case,
            domain=domain,
        )

        if not evidence:
            raise RuntimeError(
                "검색된 근거가 없습니다. "
                "진료과 필터를 확인하세요."
            )

        force_abstention = (
            requires_forced_abstention(
                patient_case
            )
        )

        raw_decision = self.llm.chat_json(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=build_decision_prompt(
                patient_case,
                evidence,
                force_abstention=force_abstention,
            ),
            temperature=0.0,
            json_schema=(
                ClinicalDecision.model_json_schema()
            ),
        )

        normalized_decision = (
            _normalize_decision_output(
                raw_decision,
                patient_case=patient_case,
            )
        )

        # RedFlag와 NextStep의 urgency 허용값이 다른데
        # 3B 모델이 이를 구분하지 못해 섞어 쓰는 사례가 있습니다.
        #   RedFlag.urgency  : 즉시 / 긴급 / 주의 / 낮음
        #   NextStep.urgency : 즉시 / 긴급 / 일반 / 선택
        flag_urgencies = {"즉시", "긴급", "주의", "낮음"}
        step_urgencies = {"즉시", "긴급", "일반", "선택"}
        flag_fixes = {
            "필수": "즉시",
            "중요": "긴급",
            "보조": "주의",
            "일반": "주의",
            "선택": "낮음",
        }
        step_fixes = {
            "필수": "즉시",
            "중요": "긴급",
            "보조": "일반",
            "주의": "긴급",
            "낮음": "선택",
        }

        for item in normalized_decision.get("red_flags", []) or []:
            value = item.get("urgency")
            if value not in flag_urgencies:
                item["urgency"] = flag_fixes.get(value, "주의")

        for item in normalized_decision.get("next_steps", []) or []:
            value = item.get("urgency")
            if value not in step_urgencies:
                item["urgency"] = step_fixes.get(value, "일반")

        # 모델이 서술 필드에 E1 같은 문서 ID를 넣고 evidence_ids를
        # 비우는 경우가 있어, 정확히 E숫자 형식인 항목만 이동합니다.
        evidence_id_pattern = re.compile(r"^E\d+$")

        for item in normalized_decision.get("differential_diagnoses", []) or []:
            if not isinstance(item, dict):
                continue

            moved_ids: list[str] = []

            for key in (
                "supporting_evidence",
                "contradicting_or_missing_evidence",
            ):
                kept_entries: list[str] = []

                for entry in item.get(key, []) or []:
                    text = str(entry).strip()

                    if evidence_id_pattern.fullmatch(text):
                        moved_ids.append(text)
                    else:
                        kept_entries.append(text)

                item[key] = kept_entries

            if moved_ids:
                existing_ids = [
                    str(value).strip()
                    for value in item.get("evidence_ids", []) or []
                    if str(value).strip()
                ]

                for evidence_id in moved_ids:
                    if evidence_id not in existing_ids:
                        existing_ids.append(evidence_id)

                item["evidence_ids"] = existing_ids

        try:
            decision = ClinicalDecision.model_validate(
                normalized_decision
            )
        except ValidationError as exc:
            raise RuntimeError(
                "LLM 출력을 보정한 뒤에도 "
                "ClinicalDecision 스키마를 "
                f"만족하지 않습니다: {exc}\n"
                f"보정된 출력: {normalized_decision}"
            ) from exc

        valid_ids = {item.evidence_id for item in evidence}
        decision = self._sanitize_citations(decision, valid_ids)

        unique_diagnoses = []
        seen_names: set[str] = set()
        for diagnosis in decision.differential_diagnoses:
            normalized_name = diagnosis.name.strip().casefold()
            if not normalized_name or normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            unique_diagnoses.append(diagnosis)
        decision.differential_diagnoses = unique_diagnoses[:3]

        # 다음 단계만 생성된 결과도 감별진단 결과로 간주하지 않습니다.
        if not decision.differential_diagnoses:
            decision.abstain = True

            if not decision.abstain_reason:
                decision.abstain_reason = (
                    "충분한 감별진단 또는 다음 단계를 "
                    "생성할 수 없습니다."
                )

        # 실제 문서 근거와 연결된 진단이 하나도 없는 경우
        grounded_diagnosis_exists = any(
            diagnosis.evidence_ids
            for diagnosis
            in decision.differential_diagnoses
        )

        if (
            decision.differential_diagnoses
            and not grounded_diagnosis_exists
        ):
            decision.abstain = True

            if not decision.abstain_reason:
                decision.abstain_reason = (
                    "검색된 의료지식과 직접 연결된 "
                    "감별진단이 없습니다."
                )

        if force_abstention:
            decision.abstain = True
            decision.differential_diagnoses = []
            decision.next_steps = []
            decision.abstain_reason = (
                "확정 진단 또는 구체적인 처방을 "
                "요구하는 고위험 요청이 감지되었습니다. "
                "의료진의 직접 평가가 필요합니다."
            )

        verification: VerificationResult | None = None

        if verify:
            raw_verification = self.llm.chat_json(
                system_prompt=VERIFIER_SYSTEM_PROMPT,
                user_prompt=build_verification_prompt(
                    patient_case,
                    decision,
                    evidence,
                ),
                temperature=0.0,
                json_schema=(
                    VerificationResult.model_json_schema()
                ),
            )

            try:
                verification = (
                    VerificationResult.model_validate(
                        raw_verification
                    )
                )
            except ValidationError as exc:
                raise RuntimeError(
                    "LLM 출력이 VerificationResult "
                    f"스키마를 만족하지 않습니다: {exc}"
                ) from exc

            for verdict in verification.verdicts:
                verdict.evidence_ids = (
                    filter_evidence_ids(
                        verdict.evidence_ids,
                        valid_ids,
                    )
                )

            if (
                verification.needs_abstention
                or not verification.overall_grounded
            ):
                decision.abstain = True

                if not decision.abstain_reason:
                    decision.abstain_reason = (
                        verification.reason
                    )

        return PipelineResult(
            decision=decision,
            evidence=evidence,
            verification=verification,
            forced_safety_abstention=(
                force_abstention
            ),
        )
        
    def run(
        self,
        patient_case: str,
        *,
        domain: str | None = None,
        verify: bool = True,
        clarification: (
            ClarificationAssessment | None
        ) = None,
        clarification_answers: (
            dict[str, str] | None
        ) = None,
        skip_clarification: bool = False,
    ) -> ClarificationAssessment | PipelineResult:
        """
        전체 MedCheck 흐름을 실행합니다.

        1. 추가 질문 필요 여부 판단
        2. 필요하면 ClarificationAssessment를 반환하고 종료
        3. 답변이 제공되면 환자 사례와 병합
        4. RAG 및 감별진단 생성
        5. 선택적으로 검증 수행
        """
        if len(patient_case.strip()) < 10:
            raise ValueError(
                "환자 사례를 10자 이상 입력하세요."
            )

        if (
            clarification_answers is not None
            and clarification is None
        ):
            raise ValueError(
                "clarification_answers를 사용하려면 "
                "이전에 생성된 clarification 결과도 "
                "함께 전달해야 합니다."
            )

        completed_case = patient_case.strip()

        # 이미 받은 추가 질문 결과와 사용자 답변이 있는 경우
        if (
            clarification is not None
            and clarification_answers is not None
        ):
            completed_case = (
                self.merge_clarification_answers(
                    patient_case,
                    clarification,
                    clarification_answers,
                )
            )

        # 아직 추가 질문 단계를 수행하지 않은 경우
        elif not skip_clarification:
            assessment = self.assess_clarification(
                patient_case
            )

            # 응급 여부와 관계없이 질문이 필요하면
            # 감별진단을 생성하지 않고 여기서 반환합니다.
            if assessment.requires_clarification:
                return assessment

        # 추가 질문이 필요 없거나 답변이 병합된 경우
        return self._run_decision(
            completed_case,
            domain=domain,
            verify=verify,
        )
