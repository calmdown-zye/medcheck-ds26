"""
훼손(Damage) 모듈

AI-Hub 원본 question(검사와 진단이 이미 끝난 완성된 임상 시나리오)에서
정보 일부를 제거하여 "초진 시점의 불완전한 임상 기록"으로 바꾸고,
무엇을 지웠는지 함께 기록합니다.

문체는 원본의 임상 서술체를 그대로 유지합니다.
(사용자가 의료진이므로 구어체로 변환하지 않습니다.)

이 모듈의 산출물은 SFT 학습 데이터의 '입력' 쪽이 되며,
removed 필드는 이후 생성 결과를 채점하는 기준으로 사용됩니다.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------
# 1. 임상 필드 정의
#    build_clarification_prompt()가 쓰는 11개 필드와 동일한 어휘를 쓰고,
#    AI-Hub 시나리오에 자주 등장하는 2개를 추가합니다.
# ---------------------------------------------------------------

ClinicalField = Literal[
    "age",
    "sex",
    "chief_complaint",
    "symptoms",
    "onset",
    "duration",
    "pain_character",
    "radiation",
    "vital_signs",
    "medical_history",
    "medications",
    # AI-Hub 시나리오 특유의 항목 (환자가 알 수 없는 정보)
    "exam_results",
    "diagnosis",
]

# 환자 발화에는 절대 남으면 안 되는 필드.
# 원본 시나리오는 대부분 "MRI에서 ~ 확인되었다" 처럼 진단이 끝난 상태이므로
# 이 두 항목은 강제로 제거합니다.
FORCED_REMOVAL: tuple[str, ...] = ("exam_results", "diagnosis")

FIELD_LABELS_KO: dict[str, str] = {
    "age": "나이",
    "sex": "성별",
    "chief_complaint": "주호소",
    "symptoms": "동반 증상",
    "onset": "발생 시점",
    "duration": "지속 시간",
    "pain_character": "통증 양상",
    "radiation": "방사 여부",
    "vital_signs": "활력징후",
    "medical_history": "과거력",
    "medications": "복용약",
    "exam_results": "검사 결과",
    "diagnosis": "확정 진단",
}


# ---------------------------------------------------------------
# 2. 훼손 결과 스키마
# ---------------------------------------------------------------

class DamagedCase(BaseModel):
    """LLM이 생성해야 하는 훼손 결과."""

    damaged: str = Field(
        description="환자가 직접 말하는 것처럼 바꾼, 정보가 일부 빠진 발화",
    )
    removed: list[ClinicalField] = Field(
        default_factory=list,
        description="원본에는 있었지만 damaged에서 의도적으로 제거한 임상 필드",
    )
    kept: list[ClinicalField] = Field(
        default_factory=list,
        description="damaged에 그대로 남겨둔 임상 필드",
    )

    @field_validator("damaged")
    @classmethod
    def _strip(cls, value: str) -> str:
        return (value or "").strip()


class DamageRecord(BaseModel):
    """훼손 스크립트의 최종 출력 1건 (JSONL 한 줄)."""

    qa_id: str
    domain: str
    q_type: str
    original_question: str
    scenario: str            # 원본에서 시나리오 부분만 추출한 것
    damaged: str
    removed: list[str]
    kept: list[str]
    valid: bool = True
    reject_reason: str = ""


# ---------------------------------------------------------------
# 3. 시나리오 추출
#    원본 question = [임상 시나리오] + [시험문제 지시문/선택지]
#    우리는 앞부분만 필요합니다.
# ---------------------------------------------------------------

# "1)" "2." 같은 객관식 선택지 시작
_OPTION_START = re.compile(r"(?:^|\n)\s*[1-9]\s*[\)\.]")

# "다음 중 ~은?" / "~설명하시오" / "~논하시오" 같은 시험 지시문
_INSTRUCTION_PATTERNS = [
    r"다음\s*중",
    r"가장\s*(적절|올바른|알맞은|적합)",
    r"옳은\s*것은",
    r"옳지\s*않은",
    r"적절하지\s*않은",
    r"무엇인가",
    r"설명하시오",
    r"서술하시오",
    r"논하시오",
    r"기술하시오",
    r"쓰시오",
    r"제시하시오",
]
_INSTRUCTION_RE = re.compile("|".join(_INSTRUCTION_PATTERNS))

# "N세 남성/여성" 으로 시작하는지 (시나리오 문항 판별)
SCENARIO_START_RE = re.compile(
    r"^\s*\d{1,3}\s*(세|개월|살)\s*(남성|여성|남자|여자|남아|여아|환자)"
)


def is_scenario_question(question: str) -> bool:
    """환자 시나리오로 시작하는 문항인지 판별합니다."""
    return bool(SCENARIO_START_RE.match(question or ""))


def extract_scenario(question: str) -> str:
    """
    question에서 임상 시나리오 부분만 잘라냅니다.

    1) 객관식 선택지가 시작되면 그 앞까지
    2) 시험 지시문이 나오면 그 문장 앞까지
    """
    text = (question or "").strip()

    # 1) 선택지 앞에서 자르기
    option_match = _OPTION_START.search(text)
    if option_match:
        text = text[: option_match.start()].strip()

    # 2) 시험 지시문이 포함된 문장 제거
    #    문장 단위로 쪼갠 뒤, 지시문이 있는 문장부터 버립니다.
    sentences = re.split(r"(?<=[.?!])\s+|\n+", text)
    kept: list[str] = []
    for sentence in sentences:
        if _INSTRUCTION_RE.search(sentence):
            break
        if sentence.strip():
            kept.append(sentence.strip())

    return " ".join(kept).strip()


# ---------------------------------------------------------------
# 4. 훼손용 프롬프트
# ---------------------------------------------------------------

DAMAGE_SYSTEM_PROMPT = """
당신은 임상 시나리오를 실제 환자의 발화로 바꾸는 데이터 변환기다.

목적:
의료 AI가 "정보가 부족할 때 되묻는 능력"을 학습하도록,
완성된 임상 시나리오에서 정보 일부를 의도적으로 제거한
불완전한 환자 발화를 만든다.

반드시 지켜야 할 규칙:
1. 검사 결과(exam_results)와 확정 진단(diagnosis)은 항상 제거한다.
   초진 시점에는 아직 검사와 진단이 이루어지지 않은 상태이기 때문이다.
2. 그 외에 2~4개의 임상 필드를 추가로 제거한다.
3. 원본에 없던 정보를 새로 만들어내지 않는다.
4. damaged는 원본의 임상 서술체를 그대로 유지한다.
   문체를 바꾸지 말고, 정보만 제거한다.
   제거한 정보는 damaged에서 어떤 형태로도 언급하지 않는다.
5. removed에는 제거한 필드명을, kept에는 남겨둔 필드명을 정확히 기록한다.
6. removed와 kept에 같은 필드를 동시에 넣지 않는다.
7. JSON 객체 하나만 출력한다.
"""


def build_damage_prompt(scenario: str) -> str:
    schema = DamagedCase.model_json_schema()
    field_list = "\n".join(
        f"- {name}: {label}"
        for name, label in FIELD_LABELS_KO.items()
    )

    return f"""
[원본 임상 시나리오]
{scenario}

[사용 가능한 임상 필드]
{field_list}

[과제]
위 시나리오에서 아래 조건에 맞게 정보를 일부 제거하라.
문체는 원본 그대로 유지한다.

- exam_results와 diagnosis는 반드시 removed에 포함한다.
  (원본에 없더라도 removed에 넣는다)
- 그 외 2~4개 필드를 추가로 제거한다.
- 남긴 정보는 kept에 기록한다.
- damaged는 원본과 같은 임상 서술체로 쓴다. 구어체로 바꾸지 않는다.

[예시]
원본: "60세 남성이 만성 요통과 다리 저림을 호소하며 내원했다. MRI에서 요추 추간판 탈출증이 확인되었다."
출력:
{{
  "damaged": "60세 남성이 요통과 다리 저림을 호소하며 내원했다.",
  "removed": ["exam_results", "diagnosis", "onset", "duration"],
  "kept": ["age", "sex", "chief_complaint", "symptoms"]
}}

[출력 JSON Schema]
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


# ---------------------------------------------------------------
# 5. 자기검증
#    LLM이 removed/kept에 적은 내용이 damaged와 실제로 일치하는지 확인합니다.
#
#    중요: 원본에 애초에 없던 필드는 검증 대상에서 제외합니다.
#    (원본에 활력징후가 없는데 kept에 vital_signs를 적었다면
#     그건 모델의 라벨 실수일 뿐, damaged 자체는 멀쩡할 수 있습니다.)
# ---------------------------------------------------------------

_AGE_RE = re.compile(r"\d{1,3}\s*(세|살|개월)")
_SEX_RE = re.compile(r"(남성|여성|남자|여자|남아|여아)")
_VITAL_RE = re.compile(r"(혈압|맥박|체온|호흡수|산소포화도|\d+\s*/\s*\d+\s*mmHg)")

# 검사 시행/소견을 나타내는 표현
_EXAM_RE = re.compile(
    r"(MRI|CT|엑스레이|X-?ray|초음파|내시경|심전도|근전도|조직\s*검사|생검|"
    r"혈액\s*검사|소변\s*검사|신체\s*검진|신체\s*검사|이학적\s*검사|도플러|"
    r"검사\s*(결과|에서)|영상에서|촬영\s*결과|관찰되었|촉지되었|촉진되었)"
)

# 확정 진단이 내려졌음을 나타내는 표현
_DIAGNOSIS_RE = re.compile(
    r"(진단(을)?\s*(받|되|하)|확진|으로\s*진단|진단된|진단받은|"
    r"확인되었|의심되었|소견이\s*관찰)"
)

_MEDICATION_RE = re.compile(r"(복용|투여|처방|정제|mg|밀리그램)")

_FIELD_DETECTORS: dict[str, re.Pattern[str]] = {
    "age": _AGE_RE,
    "sex": _SEX_RE,
    "vital_signs": _VITAL_RE,
    "exam_results": _EXAM_RE,
    "diagnosis": _DIAGNOSIS_RE,
    "medications": _MEDICATION_RE,
}


def detect_fields(text: str) -> set[str]:
    """텍스트에서 탐지 가능한 임상 필드를 찾아냅니다."""
    return {
        field
        for field, detector in _FIELD_DETECTORS.items()
        if detector.search(text or "")
    }


def validate_damage(
    damaged_case: DamagedCase,
    scenario: str = "",
    *,
    min_removed: int = 2,
    max_removed: int = 8,
    min_reduction: float = 0.10,
) -> tuple[bool, str]:
    """
    훼손 결과가 쓸 만한지 규칙으로 검사합니다.

    scenario를 함께 주면 "원본에 실제로 있던 필드"만 검증하므로
    훨씬 정확합니다. (권장)

    반환: (통과 여부, 탈락 사유)
    """
    text = damaged_case.damaged
    removed = set(damaged_case.removed)
    kept = set(damaged_case.kept)

    if len(text) < 5:
        return False, "damaged가 너무 짧음"

    # 원본에서 실제로 탐지되는 필드 (scenario를 안 주면 전부 허용)
    origin_fields = detect_fields(scenario) if scenario else None

    # --- 기본 형식 검사 ---
    overlap = removed & kept
    if overlap:
        return False, f"removed와 kept가 겹침: {sorted(overlap)}"

    if not (min_removed <= len(removed) <= max_removed):
        return False, f"removed 개수 부적절 ({len(removed)}개)"

    # --- 강제 제거 항목: 원본에 있었다면 반드시 removed에 있어야 함 ---
    for field in FORCED_REMOVAL:
        if origin_fields is not None and field not in origin_fields:
            continue  # 원본에 없었으므로 검사 생략
        if field not in removed:
            return False, f"{field}가 원본에 있는데 removed에 없음"

    # --- removed라고 했는데 damaged에 흔적이 남아있는지 ---
    for field in removed:
        detector = _FIELD_DETECTORS.get(field)
        if not detector:
            continue
        if origin_fields is not None and field not in origin_fields:
            continue  # 원본에 없던 필드는 검사 생략
        if detector.search(text):
            return False, f"{field}를 지웠다고 했으나 damaged에 흔적이 남음"

    # --- kept라고 했는데 damaged에 실제로 없는지 ---
    for field in kept:
        detector = _FIELD_DETECTORS.get(field)
        if not detector:
            continue
        if origin_fields is not None and field not in origin_fields:
            continue  # 원본에 없던 필드는 검사 생략
        if not detector.search(text):
            return False, f"{field}를 남겼다고 했으나 damaged에 없음"

    # --- 실제로 훼손이 일어났는지 (길이 축소) ---
    if scenario:
        if len(text) >= len(scenario):
            return False, "damaged가 원본보다 짧아지지 않음 (훼손 미수행)"
        reduction = 1 - len(text) / len(scenario)
        if reduction < min_reduction:
            return False, f"훼손 정도가 너무 작음 (축소율 {reduction:.0%})"

    return True, ""


# 정규식으로 존재를 확인할 수 없는 필드들.
# (원본에 있었는지 판정이 불가능하므로 trim에서 건드리지 않습니다.)
UNDETECTABLE_FIELDS: frozenset[str] = frozenset(
    {
        "chief_complaint",
        "symptoms",
        "onset",
        "duration",
        "pain_character",
        "radiation",
        "medical_history",
    }
)


def trim_removed(removed: list[str], scenario: str) -> list[str]:
    """
    removed에서 "원본에 애초에 없던 필드"를 제거합니다.

    모델이 removed에 13개 필드를 전부 나열하는 경향이 있는데,
    그대로 두면 이후 채점 단계에서 원본에 없던 정보까지
    "되물었어야 한다"고 기대하게 되어 채점 기준이 망가집니다.

    탐지 불가능한 필드(onset, duration 등)는 판정할 수 없으므로
    보수적으로 그대로 남깁니다.
    """
    origin_fields = detect_fields(scenario)

    return [
        field
        for field in removed
        if field in UNDETECTABLE_FIELDS or field in origin_fields
    ]