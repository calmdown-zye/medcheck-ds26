"""
행동 카드 미리보기용 데모 앱 (모델·인덱스 불필요)

실행:
    streamlit run scripts/demo_action_card.py --server.port 8501

목적:
    - Qwen이나 인덱스를 로드하지 않고 행동 카드 UI만 즉시 확인
    - 발표 시연 전에 레이아웃/문구를 빠르게 다듬기 위함
    - 4가지 시나리오(고위험/판단보류/추가질문/정상)를 전환하며 비교
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medcheck.action_card import render_action_card, should_show_action_card

st.set_page_config(page_title="행동 카드 미리보기", layout="wide")


class Obj:
    """실제 pydantic 모델 대신 쓰는 최소 스텁"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------
# 시나리오 정의
# ---------------------------------------------------------------

SCENARIO_HIGH_RISK = Obj(
    abstain=False,
    abstain_reason="",
    red_flags=[
        Obj(finding="압박성 흉통과 식은땀 동반",
            reason="급성관상동맥증후군을 시사하는 전형적 소견입니다.",
            urgency="즉시", evidence_ids=["E1"]),
        Obj(finding="수축기 혈압 90mmHg, 맥박 110회/분",
            reason="쇼크 초기 상태일 가능성이 있어 즉각적인 평가가 필요합니다.",
            urgency="긴급", evidence_ids=["E1", "E3"]),
        Obj(finding="고혈압·당뇨 병력",
            reason="관상동맥질환의 주요 위험인자입니다.",
            urgency="주의", evidence_ids=["E2"]),
    ],
    missing_information=[
        Obj(question="12유도 심전도를 시행하셨습니까?",
            why_needed="ST 분절 상승 여부에 따라 재관류 치료 시점이 결정됩니다.",
            priority="필수"),
        Obj(question="흉통이 정확히 언제 시작되었습니까?",
            why_needed="증상 발현 시각은 재관류 치료 가능 시간을 좌우합니다.",
            priority="필수"),
        Obj(question="현재 복용 중인 항혈소판제가 있습니까?",
            why_needed="추가 투약 결정에 영향을 줍니다.",
            priority="중요"),
    ],
    structured_case={
        "age": "67", "sex": "남성", "chief_complaint": "압박성 흉통",
        "symptoms": ["압박성 흉통", "식은땀", "좌측 팔 방사통"],
        "onset": "30분 전", "duration": "지속적",
        "pain_character": "압박감, 조이는 느낌",
        "radiation": "좌측 팔",
        "vital_signs": "혈압 90/60mmHg, 맥박 110회/분",
        "medical_history": "고혈압, 당뇨",
        "medications": "미상",
    },
)

SCENARIO_ABSTAIN = Obj(
    abstain=True,
    abstain_reason="검색된 의료지식과 직접 연결된 감별진단이 없습니다.",
    red_flags=[
        Obj(finding="호흡곤란과 산소포화도 88%",
            reason="저산소혈증으로 즉각적인 산소 공급 평가가 필요합니다.",
            urgency="즉시", evidence_ids=[]),
    ],
    missing_information=[
        Obj(question="흉부 영상 검사를 시행하셨습니까?",
            why_needed="폐렴·폐색전 감별에 필수적입니다.", priority="필수"),
        Obj(question="발열이 동반되었습니까?",
            why_needed="감염성 원인 여부를 가릅니다.", priority="중요"),
    ],
    structured_case={
        "age": "73", "sex": "남성", "chief_complaint": "호흡곤란",
        "symptoms": ["호흡곤란", "빈맥"],
        "vital_signs": "산소포화도 88%, 맥박 118회/분",
        "onset": "", "duration": "미상",
        "pain_character": None, "radiation": {},
        "medical_history": [], "medications": "정보 없음",
    },
)

SCENARIO_CLARIFY = Obj(
    urgent=True,
    urgent_message="흉통은 응급 질환일 가능성이 있어, 추가 질문 응답과 관계없이 "
                   "즉시 대면 의료 평가가 필요할 수 있습니다.",
    questions=[
        Obj(question="통증이 언제 시작되었고 얼마나 지속되고 있습니까?",
            why_needed="발생 시점은 응급도와 치료 시간을 좌우합니다.", priority="필수"),
        Obj(question="식은땀, 구역, 호흡곤란이 동반됩니까?",
            why_needed="동반 증상은 급성관상동맥증후군 가능성을 높입니다.", priority="필수"),
        Obj(question="과거 심장 질환으로 진료받은 적이 있습니까?",
            why_needed="기왕력은 위험도 평가에 중요합니다.", priority="중요"),
    ],
    structured_case={
        "age": "50대", "sex": "남성", "chief_complaint": "흉통",
        "symptoms": ["흉통"], "onset": "미상",
    },
)

SCENARIO_NORMAL = Obj(
    abstain=False, abstain_reason="", red_flags=[],
    missing_information=[],
    structured_case={"age": "28", "sex": "여성", "chief_complaint": "경미한 인후통"},
)

SCENARIOS = {
    "① 고위험 (red_flags 즉시)": ("decision", SCENARIO_HIGH_RISK),
    "② 판단 보류 (abstain=True)": ("decision", SCENARIO_ABSTAIN),
    "③ 추가 질문 단계 (urgent=True)": ("assessment", SCENARIO_CLARIFY),
    "④ 정상 (카드 미표시)": ("decision", SCENARIO_NORMAL),
}

# ---------------------------------------------------------------
# 화면
# ---------------------------------------------------------------

st.title("행동 카드 미리보기")
st.caption(
    "심사 피드백 대응 — 경고 이후의 행동(진료 필요 이유 / 방문 시 준비 정보 / "
    "의료진 전달용 요약)을 한 화면에 제공합니다. 모델 로딩 없이 UI만 확인합니다."
)

choice = st.radio("시나리오", list(SCENARIOS.keys()), horizontal=True)
kind, obj = SCENARIOS[choice]

kwargs = {kind: obj}
show = should_show_action_card(**kwargs)

st.divider()

if show:
    render_action_card(**kwargs)
else:
    st.success(
        "검색 근거와 현재 입력 정보를 바탕으로 구조화된 의사결정 결과를 생성했습니다."
    )
    st.info("고위험·판단보류 상황이 아니므로 행동 카드를 노출하지 않고 "
            "일반 분석 결과를 그대로 보여줍니다.")

st.divider()
with st.expander("이 시나리오에 들어간 원본 데이터"):
    st.json({k: str(v) for k, v in obj.__dict__.items()})
