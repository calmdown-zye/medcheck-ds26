from __future__ import annotations

import json

from .models import ClarificationAssessment, ClinicalDecision, EvidenceItem


DECISION_SYSTEM_PROMPT = """
당신은 의료진의 판단을 보조하는 근거 기반 임상 의사결정 지원 모델이다.

입력에는 서로 역할이 다른 두 자료가 있다.

- 환자 사례: 현재 환자에게 실제로 확인된 사실
- 검색 근거: 감별진단과 다음 단계의 의학적 타당성을 뒷받침하는 참고 문서

두 자료를 다음 원칙에 따라 사용한다.

1. 현재 환자의 증상, 과거력, 검사 결과는 [환자 사례]에 명시된 내용만 사용한다.
2. [검색 근거]에 등장하는 다른 환자의 증상, 과거력, 진찰 소견 또는
   검사 결과를 현재 환자의 사실로 옮기지 않는다.
3. supporting_evidence에는 [환자 사례]에서 직접 확인되는 사실만 작성한다.
   검색 문서의 환자 정보나 일반적인 의학 지식을 작성하지 않는다.
4. contradicting_or_missing_evidence에는 다음 중 하나 이상을 작성한다.
   - 후보와 맞지 않는 현재 환자의 사실
   - 후보를 확인하거나 배제하기 위해 현재 사례에서 부족한 핵심 정보
5. recommended_checks에는 후보를 확인하거나 배제하기 위한 추가 평가를 하나 이상 작성한다.
6. evidence_ids에는 후보의 의학적 타당성을 뒷받침하는 [검색 근거]의 E 형식 문서 ID를 하나 이상 작성한다.
7. 검색 근거의 환자와 현재 환자의 세부 조건이 완전히 같지 않더라도,
   문서가 해당 진단 후보를 의학적으로 뒷받침하면 사용할 수 있다.
   단, 검색 문서의 환자 정보를 현재 환자의 사실로 사용해서는 안 된다.
8. 다음 두 조건을 모두 만족하는 후보만 differential_diagnoses에 포함한다.
   - 현재 환자 사례에 후보를 고려하게 하는 직접적인 사실이 하나 이상 있다.
   - 후보를 의학적으로 뒷받침하는 검색 문서가 하나 이상 있다.
9. 후보 개수를 채우기 위해 관련성이 부족한 진단을 추가하거나
   환자에게 없는 사실을 만들어내지 않는다.
10. recommended_checks와 next_steps에는 현재 확인되지 않은 결과를 이미 확인된 사실처럼 작성하지 않는다.
11. 확정 진단, 구체적인 처방 또는 약물 용량을 제시하지 않는다.
12. 위 조건을 만족하는 후보를 안전하게 만들 수 없으면 abstain=true로 설정하고 구체적인 이유를 작성한다.
13. 출력은 제공된 JSON Schema를 따르는 JSON 객체 하나로 작성한다.
14. 모든 서술은 한국어로 작성한다.
JSON 객체 하나만 출력한다.
"""

VERIFIER_SYSTEM_PROMPT = """
당신은 의료 답변의 근거성을 엄격하게 심사하는 검증기다.

규칙:
1. 환자 입력과 검색 근거에 의해 직접 지지되는 주장만 supported로 판정한다.
2. 일부만 지지되면 partially_supported로 판정한다.
3. 근거가 없거나 검색 근거와 충돌하면 unsupported로 판정한다.
4. 확정 진단, 구체적 처방·용량, 위험한 지연 권고가 있으면 unsafe_recommendations에 기록한다.
5. 핵심 감별진단 또는 다음 단계가 unsupported이면 needs_abstention=true로 설정한다.
6. JSON 객체만 출력한다.
7. 서술은 한국어로 작성한다. 중국어, 일본어를 사용하지 않는다.
   의학 용어, 약물명, 검사명은 한국어를 우선하되 필요하면 괄호 안에 영어만 병기한다.
예 : 급성관상동맥증후군(ACS), 아스피린(aspirin), 심전도(electrocardiogram, ECG)
8. 환자 입력에 없는 검사 결과, 진단 결과, 증상 또는 과거력을
   현재 환자에게 확인된 사실처럼 사용한 주장은 unsupported로 판정한다.
9. 검색 문서 속 다른 환자의 MRI, CT 또는 검사 결과를
   현재 환자의 검사 결과처럼 작성한 경우 needs_abstention=true로 설정한다.
10. needs_abstention=true인 경우 reason에는 보류를 유발한
    구체적인 주장과 실패 이유를 함께 작성한다.
    어떤 환자 사실이 제공되지 않았으며, 생성 결과가 무엇을
    잘못 주장했는지 명시한다.
11. "추가 검사 결과에 따라 치료 계획이 달라질 수 있다"와 같은
    일반적인 설명만으로 reason을 작성하지 않는다.
"""


# ---------------------------------------------------------------------------
# 실험용 프롬프트 (현재 MedCheckPipeline 운영 경로에서는 사용하지 않음)
#
# 후보 필터는 동일 입력에서도 판정이 흔들렸고, 근거 보정은 관련성이 낮은
# 후보에 근거를 억지로 채우는 사례가 있었습니다. 향후 오프라인 평가와
# 프롬프트 실험을 위해 보존하되 pipeline.py에서는 import하거나 호출하지
# 않습니다.
# ---------------------------------------------------------------------------

EVIDENCE_REPAIR_SYSTEM_PROMPT = """
당신은 감별진단 후보와 현재 환자 사례의 관련성을 심사하고,
사용 가능한 후보의 근거 필드만 작성하는 의료 지원 모델이다.

각 후보에 대해 먼저 현재 환자 사례에 직접적인 지지 사실이 있는지 판단한다.

규칙:
1. 환자 사례에 직접적인 지지 사실이 있으면 usable=true로 설정한다.
2. 직접적인 지지 사실이 없으면 usable=false로 설정한다.
3. 나이만으로 특정 질환의 위험이 높다고 추정하지 않는다.
4. 환자에게 명시되지 않은 과거력, 검사 결과 또는 위험인자를 만들지 않는다.
5. 일반적인 질환 설명을 현재 환자의 supporting_evidence로 사용하지 않는다.
6. usable=false인 후보의 근거 필드를 억지로 채우지 않는다.
7. usable=true인 후보만 supporting_evidence,
   contradicting_or_missing_evidence, recommended_checks를 작성한다.
8. 진단명을 변경하거나 새로운 후보를 추가하지 않는다.
9. JSON 객체 하나만 출력한다.
10. 모든 서술은 한국어로 작성한다.
"""

CANDIDATE_FILTER_SYSTEM_PROMPT = """
당신은 확진 여부가 아니라, 현재 환자 사례에 감별진단 후보를
고려할 직접적인 지지 사실이 있는지만 판정한다.

환자 사례에 후보를 고려하게 하는 직접적인 사실이 하나 이상 있으면
patient_support_present=true로 판정한다.

직접적인 환자 사실이 없으면 patient_support_present=false로 판정한다.

검사 결과가 없다는 이유만으로 false로 판정하지 않는다.
환자에게 없는 과거력, 검사 결과 또는 위험인자를 만들지 않는다.
입력된 후보를 추가·누락하거나 진단명을 변경하지 않는다.
JSON 객체 하나만 출력한다.
"""

CLARIFICATION_SYSTEM_PROMPT = """
당신은 환자 사례에서 임상적으로 중요한 누락 정보를 찾는
의료 의사결정 지원 시스템이다.

이 단계에서는 감별진단, 확정 진단, 치료 또는 검사 권고를
생성하지 않는다.

반드시 다음 작업만 수행한다.

1. 환자 사례에 명시된 정보를 구조화한다.
2. 안전한 분석을 위해 반드시 필요한 정보가 빠져 있는지 판단한다.
3. 필요한 경우 가장 중요한 질문을 최대 3개 생성한다.
4. 환자 사례에 이미 존재하는 정보를 다시 질문하지 않는다.
5. 질문은 답변에 따라 이후 감별진단이 실제로 달라질 수 있는
   내용만 선택한다.
6. 응급 가능성이 있으면 urgent=true로 표시한다.
7. JSON 객체 하나만 출력한다.
8. 서술은 한국어로 작성한다. 중국어, 일본어를 사용하지 않는다.
   의학 용어, 약물명, 검사명은 한국어를 우선하되 필요하면 괄호 안에 영어만 병기한다.
예 : 급성관상동맥증후군(ACS), 아스피린(aspirin), 심전도(electrocardiogram, ECG)
"""

def format_evidence(evidence: list[EvidenceItem]) -> str:
    blocks: list[str] = []
    for item in evidence:
        blocks.append(
            f"[{item.evidence_id}]\n"
            f"qa_id: {item.qa_id}\n"
            f"진료과: {item.domain}\n"
            f"질문 유형: {item.q_type}\n"
            f"질문: {item.question}\n"
            f"답변: {item.answer}"
        )
    return "\n\n".join(blocks)


def build_decision_prompt(
    patient_case: str,
    evidence: list[EvidenceItem],
    *,
    force_abstention: bool = False,
) -> str:
    schema = ClinicalDecision.model_json_schema()
    return f"""
[환자 사례]
{patient_case}

[검색 근거]
{format_evidence(evidence)}

[강제 안전 조건]
{"확정 진단 또는 구체적 처방을 요구한 고위험 요청이므로 abstain=true로 설정한다." if force_abstention else "없음"}

[생성 절차]
다음 순서로 결과를 작성한다.

1. 환자 사례에 명시된 정보만 structured_case로 구조화한다.
2. 현재 환자에게 확인된 위험 신호를 red_flags에 작성한다.
3. 안전한 판단에 필요한 누락 정보를 missing_information에 작성한다.
4. 검색 근거와 환자 사실이 모두 있는 감별진단만 1~3개 작성한다.
   세 개를 채울 필요는 없다.
5. 각 감별진단에 다음 내용을 구분하여 작성한다.
   - supporting_evidence:
     현재 환자 사례에서 해당 후보를 지지하는 사실
   - contradicting_or_missing_evidence:
     해당 후보와 맞지 않는 사실 또는 현재 환자에게 부족한 정보
   - recommended_checks:
     해당 후보를 확인하거나 배제하기 위해 필요한 추가 평가
   - evidence_ids:
     해당 후보의 의학적 타당성을 뒷받침하는 검색 문서 ID
6. 필요한 다음 평가 단계를 next_steps에 작성한다.
7. 안전하게 판단할 수 없으면 abstain=true와 구체적인 사유를 작성한다.

[출력 불변조건]
- 검색 문서에 등장하는 다른 환자의 증상, 과거력, 진찰 소견 및 검사 결과를 현재 환자의 사실로 표현하지 않는다.
- supporting_evidence에는 [환자 사례]에 직접 적힌 사실만 작성한다. 일반적인 의학 지식이나 검색 문서 속 다른 환자의 정보는 작성하지 않는다.
- 각 감별진단의 supporting_evidence에는 하나 이상의 항목이 있어야 한다.
- 각 감별진단의 contradicting_or_missing_evidence에는 반대 소견 또는 현재 사례에서 확인되지 않은 핵심 정보를 하나 이상 작성한다.
- 각 감별진단의 recommended_checks에는 해당 후보를 확인하거나 배제하기 위한 추가 평가를 하나 이상 작성한다.
- 각 감별진단의 evidence_ids에는 해당 후보를 의학적으로 뒷받침하는 검색 문서 ID를 하나 이상 작성한다.
- evidence_ids에는 [검색 근거]에 제공된 E 형식 ID만 사용한다.
- 문서 ID는 supporting_evidence,contradicting_or_missing_evidence 또는 recommended_checks에 넣지 않고 evidence_ids에만 작성한다.
- 현재 환자 사례에 직접적인 지지 사실이 없는 후보는 생성하지 않는다.
- 후보를 뒷받침하는 검색 문서가 없는 후보는 생성하지 않는다.
- 위 네 근거 필드를 모두 근거 있게 작성할 수 없는 후보는 생성하지 않는다.
- 후보 개수를 채우기 위해 환자 사실을 만들거나 관련성이 낮은 후보를 추가하지 않는다.
- "기타 원인", "추가 검사 필요"처럼 평가 단계를 진단명으로 만들지 않는다.
- next_steps의 각 항목은 action, rationale, urgency, evidence_ids를 포함한다.
- red_flags의 urgency는 "즉시", "긴급", "주의", "낮음" 중 하나다.
- next_steps의 urgency는 "즉시", "긴급", "일반", "선택" 중 하나다.
- 같은 감별진단을 반복하지 않는다.
- JSON Schema에 정의되지 않은 필드를 추가하지 않는다.

[출력 JSON Schema]
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""

def build_evidence_repair_prompt(
    patient_case: str,
    diagnosis_names: list[str],
    ) -> str:
    """실험용 근거 보정 프롬프트를 만듭니다.

    현재 실시간 파이프라인에서는 사용하지 않습니다.
    """
    return f"""
[환자 사례]
{patient_case}

[감별진단 후보]
{json.dumps(diagnosis_names, ensure_ascii=False)}

[과제]
각 감별진단 후보에 대해 현재 환자 사례와의 관련성을 먼저 판정한다.

usable=true인 후보에 대해서만 다음 세 필드를 작성한다.

- supporting_evidence:
  현재 환자 사례에서 해당 후보를 직접 지지하는 사실
- contradicting_or_missing_evidence:
  해당 후보와 맞지 않는 환자 사실 또는 환자에게 부족한 정보
- recommended_checks:
  해당 후보를 확인하거나 배제하기 위한 추가 평가

[규칙]
- 입력으로 제공된 진단명만 사용한다.
- 후보를 추가하거나 진단명을 변경하지 않는다.
- 현재 환자 사례에 직접적인 지지 사실이 있으면 usable=true로 설정한다.
- 현재 환자 사례에 직접적인 지지 사실이 없으면 usable=false로 설정한다.
- 나이만으로 특정 질환의 위험이 높다고 추정하지 않는다.
- 환자에게 명시되지 않은 과거력, 검사 결과 또는 위험인자를 만들지 않는다.
- 일반적인 질환 설명을 현재 환자의 supporting_evidence로 사용하지 않는다.
- usable=true인 경우 세 근거 필드를 각각 최소 1개 작성한다.
- usable=false인 경우 reason에 제외 이유를 작성한다.
- usable=false인 경우 supporting_evidence,
  contradicting_or_missing_evidence,
  recommended_checks는 빈 배열로 둔다.
- 후보 개수를 유지하기 위해 근거를 만들어내지 않는다.
- JSON 객체 하나만 출력한다.

[출력 형식]
{{
  "differentials": [
    {{
      "name": "입력으로 제공된 진단명",
      "usable": true,
      "reason": "현재 환자 사례와의 관련성 판정 이유",
      "supporting_evidence": ["현재 환자 사례에서 직접 확인된 사실"],
      "contradicting_or_missing_evidence": ["반대 소견 또는 부족한 정보"],
      "recommended_checks": ["추가 평가"]
    }}
  ]
}}
"""


def build_verification_prompt(
    patient_case: str,
    decision: ClinicalDecision,
    evidence: list[EvidenceItem],
) -> str:
    decision_json = decision.model_dump_json(ensure_ascii=False, indent=2)
    return f"""
[환자 사례]
{patient_case}

[검색 근거]
{format_evidence(evidence)}

[검증 대상 답변]
{decision_json}

[출력 형식]
{{
  "verdicts": [
    {{
      "claim": "검증한 핵심 주장",
      "status": "supported | partially_supported | unsupported",
      "evidence_ids": ["E1"],
      "reason": "판정 이유"
    }}
  ],
  "unsupported_claim_rate": 0.0,
  "unsafe_recommendations": [],
  "overall_grounded": true,
  "needs_abstention": false,
  "reason": "전체 판정 이유"
}}

감별진단, 응급징후, 다음 단계 조치의 핵심 주장만 검증한다.
"""

def build_candidate_filter_prompt(
    patient_case: str,
    diagnosis_names: list[str],
) -> str:
    """실험용 후보 필터 프롬프트를 만듭니다.

    현재 실시간 파이프라인에서는 사용하지 않습니다.
    """
    return f"""
[환자 사례]
{patient_case}

[감별진단 후보]
{json.dumps(diagnosis_names, ensure_ascii=False)}

[과제]
각 후보에 대해 현재 환자 사례에 그 후보를 고려하게 하는
직접적인 환자 사실이 하나 이상 있는지 판정한다.

이 판정은 확진 가능 여부를 판단하는 것이 아니다.

[판정 기준]
- 직접적인 지지 사실이 있으면 patient_support_present=true
- 직접적인 지지 사실이 없으면 patient_support_present=false

[규칙]
- 입력된 모든 후보를 하나씩 판정한다.
- 후보를 추가·누락하거나 진단명을 변경하지 않는다.
- MRI 등 검사가 없다는 이유만으로 false로 판정하지 않는다.
- 나이만으로 특정 질환을 지지하지 않는다.
- 일반적인 의학 지식을 환자 사실로 사용하지 않는다.
- false인 경우 reason이 비어 있어도 된다.

[출력 형식]
{{
  "candidates": [
    {{
      "name": "입력된 진단명",
      "patient_support_present": true,
      "reason": "판정 이유"
    }}
  ]
}}
"""


def build_clarification_prompt(
    patient_case: str,
) -> str:
    schema = ClarificationAssessment.model_json_schema()

    return f"""
[환자 사례]
{patient_case}

[과제]
환자 사례에서 다음 정보를 구조화하라.

- age
- sex
- chief_complaint
- symptoms
- onset
- duration
- pain_character
- radiation
- vital_signs
- medical_history
- medications

그 후 안전한 감별진단을 수행하기 위해 반드시 필요한 정보가
빠져 있는지 판단하라.

[질문 생성 규칙]
- 나이, 성별, 주호소, 발생 시점, 주요 증상이 모두 확인되고
  감별진단을 좁힐 만한 정보가 이미 충분하면
  requires_clarification=false로 설정하고 questions를 빈 배열로 둔다.
- 억지로 질문을 만들지 않는다. 질문이 없어도 된다.
- 질문은 최대 3개만 생성한다.
- 이미 제공된 정보를 다시 묻지 않는다.
- 단순히 있으면 좋은 정보가 아니라 판단을 바꿀 수 있는 정보를 묻는다.
- 필수 질문이 하나 이상이면 requires_clarification=true로 설정한다.
- 이 단계에서는 질환명이나 감별진단을 생성하지 않는다.
- 응급 가능성이 있으면 urgent=true로 설정하고,
  urgent_message에 즉시 대면 평가가 필요할 수 있음을 작성한다.

[예시 1] 정보가 충분한 경우 — 질문 없음
사례: "32세 남자가 구강 점막의 궤양과 외음부의 궤양을 주소로 내원하였다.
구강 점막 병변의 경계는 명확했고, 최근 1년간 4번 재발하였다고 한다.
음낭에서도 유사한 궤양 병변이 발견되었다. 1개월 전 포도막염을 진단받은 과거력이 있다."
판단: 나이, 성별, 주호소, 병변 위치, 재발 양상, 과거력이 모두 확인되었다.
감별을 좁히기에 충분하므로 추가 질문이 필요하지 않다.
출력: {{"requires_clarification": false, "questions": [], "reason": "감별에 필요한 핵심 정보가 이미 제공되었다."}}

[예시 2] 정보가 부족한 경우 — 질문 필요
사례: "남성이 가슴이 아프다고 호소하며 내원하였다."
판단: 나이, 발생 시점, 통증 양상이 모두 없어 감별이 불가능하다.
출력: {{"requires_clarification": true, "questions": [3개 이내의 필수 질문]}}


[출력 JSON Schema]
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""
