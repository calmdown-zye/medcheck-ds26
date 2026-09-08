from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class MedicalQARecord(BaseModel):
    qa_id: str
    domain_code: str = ""
    domain_name: str = "미분류"
    q_type_code: str = ""
    q_type_name: str = "미분류"
    question: str
    answer: str
    source_file: str
    split: str = "Unknown"

    @field_validator("qa_id", "question", "answer", mode="before")
    @classmethod
    def convert_to_str(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @property
    def retrieval_text(self) -> str:
        return (
            f"진료과: {self.domain_name}\n"
            f"질문 유형: {self.q_type_name}\n"
            f"질문: {self.question}\n"
            f"답변: {self.answer}"
        )


class EvidenceItem(BaseModel):
    evidence_id: str
    qa_id: str
    domain: str
    q_type: str
    question: str
    answer: str
    source_file: str
    score: float


class RedFlag(BaseModel):
    finding: str
    reason: str
    urgency: Literal["즉시", "긴급", "주의", "낮음"] = "주의"
    evidence_ids: list[str] = Field(default_factory=list)


class MissingInformation(BaseModel):
    question: str
    why_needed: str
    priority: Literal["필수", "중요", "보조"] = "중요"


class DifferentialDiagnosis(BaseModel):
    rank: int
    name: str
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_or_missing_evidence: list[str] = Field(default_factory=list)
    recommended_checks: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class NextStep(BaseModel):
    action: str
    rationale: str
    urgency: Literal["즉시", "긴급", "일반", "선택"] = "일반"
    evidence_ids: list[str] = Field(default_factory=list)


class ClinicalDecision(BaseModel):
    case_summary: str
    structured_case: dict[str, Any] = Field(default_factory=dict)
    red_flags: list[RedFlag] = Field(default_factory=list)
    missing_information: list[MissingInformation] = Field(default_factory=list)
    differential_diagnoses: list[DifferentialDiagnosis] = Field(default_factory=list)
    next_steps: list[NextStep] = Field(default_factory=list)
    abstain: bool = False
    abstain_reason: str = ""
    safety_notice: str = (
        "본 결과는 의료진의 판단을 보조하기 위한 참고 정보이며 "
        "확정 진단 또는 처방을 대체하지 않습니다."
    )


class ClaimVerdict(BaseModel):
    claim: str
    status: Literal["supported", "partially_supported", "unsupported"]
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class VerificationResult(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
    unsupported_claim_rate: float = 0.0
    unsafe_recommendations: list[str] = Field(default_factory=list)
    overall_grounded: bool = False
    needs_abstention: bool = False
    reason: str = ""


class PipelineResult(BaseModel):
    decision: ClinicalDecision
    evidence: list[EvidenceItem]
    verification: VerificationResult | None = None
    forced_safety_abstention: bool = False
    
    @property
    def answer(self) -> str:
        sections: list[str] = []

        sections.append(
            f"사례 요약\n{self.decision.case_summary}"
        )

        if self.decision.red_flags:
            red_flags = "\n".join(
                f"- {item.finding}: {item.reason}"
                for item in self.decision.red_flags
            )
            sections.append(
                f"주요 위험 신호\n{red_flags}"
            )

        if self.decision.differential_diagnoses:
            diagnoses = "\n".join(
                (
                    f"{item.rank}. {item.name}\n"
                    f"   지지 근거: "
                    f"{', '.join(item.supporting_evidence) or '없음'}\n"
                    f"   추가 확인: "
                    f"{', '.join(item.recommended_checks) or '없음'}"
                )
                for item in self.decision.differential_diagnoses
            )
            sections.append(
                f"감별진단\n{diagnoses}"
            )

        if self.decision.next_steps:
            next_steps = "\n".join(
                f"- [{item.urgency}] {item.action}: {item.rationale}"
                for item in self.decision.next_steps
            )
            sections.append(
                f"다음 단계\n{next_steps}"
            )

        if self.decision.abstain:
            sections.append(
                f"판단 보류\n{self.decision.abstain_reason}"
            )

        return "\n\n".join(sections)

class TextDemoResult(BaseModel):
    mode: Literal["base", "rag"]
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    
class ClarificationQuestion(BaseModel):
    question_id: str
    question: str
    why_needed: str
    priority: Literal["필수", "중요", "보조"]
    answer_type: Literal[
        "text",
        "yes_no",
        "number",
        "choice",
    ] = "text"
    options: list[str] = Field(default_factory=list)

class ClarificationAssessment(BaseModel):
    structured_case: dict[str, Any] = Field(default_factory=dict)

    requires_clarification: bool = False
    questions: list[ClarificationQuestion] = Field(
        default_factory=list
    )
    reason: str = ""

    # 응급 가능성이 있으면 질문 여부와 별도로 즉시 경고를 표시
    urgent: bool = False
    urgent_message: str = ""
    
class ClarificationResult(BaseModel):
    requires_clarification: bool
    questions: list[ClarificationQuestion] = Field(
        default_factory=list
    )
    reason: str = ""