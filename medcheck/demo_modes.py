from __future__ import annotations

from .index import MedicalHybridIndex
from .llm import LocalTransformersClient
from .models import TextDemoResult
from .prompts import format_evidence


BASE_SYSTEM_PROMPT = """
당신은 환자 사례를 검토하는 의료 의사결정 지원 AI다.

다음 원칙을 따른다.
- 확정 진단을 내리지 않는다.
- 가능한 질환과 그 이유를 설명한다.
- 추가로 확인해야 할 정보를 제시한다.
- 응급 가능성이 있으면 대면 의료 평가가 필요함을 안내한다.
- 구체적인 약물 용량이나 처방을 생성하지 않는다.
"""


RAG_SYSTEM_PROMPT = """
당신은 검색된 전문 의학지식을 참고하여 환자 사례를 분석하는
의료 의사결정 지원 AI다.

다음 원칙을 따른다.
- 제공된 검색 근거를 우선적으로 활용한다.
- 검색 근거에 없는 내용을 확정적으로 단정하지 않는다.
- 가능한 질환, 근거, 추가 확인사항, 다음 단계를 자연어로 설명한다.
- 사용한 근거는 E1, E2 형식으로 표시한다.
- 확정 진단이나 구체적인 처방을 생성하지 않는다.
"""


def run_base_llm(
    *,
    llm: LocalTransformersClient,
    patient_case: str,
) -> TextDemoResult:
    """
    검색 없이 환자 사례만 로컬 LLM에 전달한다.
    """

    answer = llm.chat_text(
        system_prompt=BASE_SYSTEM_PROMPT,
        user_prompt=f"""
[환자 사례]
{patient_case}

위 사례에서 고려할 수 있는 질환, 위험 신호,
추가로 필요한 정보와 다음 단계를 설명하라.
""",
        temperature=0.0,
    )

    return TextDemoResult(
        mode="base",
        answer=answer,
        evidence=[],
    )


def run_rag_llm(
    *,
    index: MedicalHybridIndex,
    llm: LocalTransformersClient,
    patient_case: str,
    domain: str | None = None,
    top_k: int = 3,
    dense_weight: float = 0.65,
    bm25_weight: float = 0.35,
) -> TextDemoResult:
    """
    의료지식 검색 결과를 제공하되 자유형 자연어 답변을 생성한다.
    """

    evidence = index.search(
        patient_case,
        top_k=top_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
        domain=domain,
    )

    if not evidence:
        raise RuntimeError(
            "관련 의료지식을 검색하지 못했습니다."
        )

    evidence_text = format_evidence(evidence)

    answer = llm.chat_text(
        system_prompt=RAG_SYSTEM_PROMPT,
        user_prompt=f"""
[환자 사례]
{patient_case}

[검색된 전문 의학지식]
{evidence_text}

[과제]
검색 근거를 바탕으로 다음을 자연어로 설명하라.

1. 현재 사례의 핵심 위험 신호
2. 고려할 수 있는 주요 질환
3. 추가로 확인해야 하는 정보
4. 권장되는 다음 평가 단계
5. 각 판단에 사용한 근거 ID

설명이나 결론을 검색 근거보다 강하게 단정하지 마라.
""",
        temperature=0.0,
    )

    return TextDemoResult(
        mode="rag",
        answer=answer,
        evidence=evidence,
    )
