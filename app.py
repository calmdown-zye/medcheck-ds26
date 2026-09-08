from __future__ import annotations

import base64
import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from medcheck.constants import DOMAIN_CODE_TO_NAME
from medcheck.demo_modes import run_base_llm, run_rag_llm
from medcheck.index import MedicalHybridIndex
from medcheck.llm import LocalTransformersClient
from medcheck.models import ClarificationAssessment, PipelineResult
from medcheck.pipeline import MedCheckPipeline



from medcheck import ui
from medcheck.action_card import (
    render_action_card,
    should_show_action_card,
    infer_urgency,
)

# ============================================================
# 1. 기본 설정
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="XDoctors | MedCheck AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_LABEL_TO_KEY = {
    "Base LLM": "base",
    "LLM + RAG": "rag",
    "Structured RAG": "structured",
    "MedCheck AI": "medcheck",
}

# 결과 화면에서 내부 키 대신 사람이 읽는 이름을 표시하기 위한 역매핑입니다.
MODE_KEY_TO_LABEL = {key: label for label, key in MODE_LABEL_TO_KEY.items()}

MODE_FEATURES = {
    "base": {
        "RAG": False,
        "구조화": False,
        "추가 질문": False,
        "근거 검증": False,
    },
    "rag": {
        "RAG": True,
        "구조화": False,
        "추가 질문": False,
        "근거 검증": False,
    },
    "structured": {
        "RAG": True,
        "구조화": True,
        "추가 질문": False,
        "근거 검증": False,
    },
    "medcheck": {
        "RAG": True,
        "구조화": True,
        "추가 질문": True,
        "근거 검증": True,
    },
}

DEFAULT_CASE = (
    "67세 남성이 30분 전부터 압박성 흉통과 식은땀을 호소한다."
)

EXAMPLE_CASES = {
    "직접 입력": "",
    "급성 흉통": DEFAULT_CASE,
    "급성 복통": (
        "42세 여성이 6시간 전부터 시작된 우하복부 통증과 구역을 호소한다. "
        "체온은 38.1도이며 식욕이 감소했다."
    ),
    "호흡곤란": (
        "73세 남성이 갑자기 숨이 차고 가슴이 답답하다고 호소한다. "
        "산소포화도는 88%이고 맥박은 118회/분이다."
    ),
}

NOTICE_MD = """
### 의료 고지

- 본 서비스는 **연구·교육용 임상 의사결정 지원 프로토타입**입니다.
- 결과는 확정 진단, 처방 또는 의료진의 직접적인 판단을 대체하지 않습니다.
- 응급 증상이 의심되면 시스템 결과를 기다리지 말고 즉시 의료기관의 평가를 받아야 합니다.
- 이름, 연락처, 주민등록번호 등 개인을 직접 식별할 수 있는 정보는 입력하지 마세요.
"""


# ------------------------------------------------------------
# 표시용 리소스 (로고 · 데이터 출처)
# 로고 파일이 없어도 앱이 중단되지 않도록 모두 선택적으로 취급합니다.
# ------------------------------------------------------------

ASSETS_DIR = Path(__file__).parent / "assets"
KDS_LOGO_PATH = ASSETS_DIR / "kds-consortium-logo.png"
AIHUB_LOGO_PATH = ASSETS_DIR / "aihub-logo.png"

AIHUB_DATASET_URL = (
    "https://aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115"
    "&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EC%A0%84%EB%AC%B8"
    "%EC%9D%98%ED%95%99&srchDetailCnd=DETAILCND001&srchOrder=ORDER001"
    "&srchPagePer=20&aihubDataSe=data&dataSetSn=71874"
)

AIHUB_SOURCE_NOTE = (
    "검색 지식베이스는 AI Hub의 '전문 의학지식 데이터'를 가공·인덱싱하여 "
    "구성했습니다. AI Hub는 본 서비스가 생성한 의료 판단을 보증하지 않습니다."
)

# 화면 흐름 표시용. session_state에 새로운 상태값을 만들지 않고,
# 기존 stage 값(input | clarification | result)을 읽어 표시만 합니다.
STAGE_STEPS = (
    "환자 사례 입력",
    "추가 정보 확인",
    "근거 기반 분석",
    "결과와 다음 단계",
)

STAGE_TO_STEP_INDEX = {
    "input": 0,
    "clarification": 1,
    "result": 3,
}

# 단계 표시를 실행 중에 다시 그리기 위한 자리. 화면 표시 전용입니다.
_STAGE_SLOT: Any = None


# ============================================================
# 2. 스타일
# ============================================================

st.markdown(
    """
    <style>
    /* ============================================================
       디자인 토큰
       - 기존 --xd-* 변수명은 medcheck/ui.py와 action_card.py가
         참조하므로 이름을 유지하고 값만 역할 중심으로 정리합니다.
       ============================================================ */
    :root {
        /* 브랜드 */
        --xd-navy: #18344F;
        --xd-navy-soft: #EDF1F5;

        /* 상호작용 = 파랑 */
        --xd-blue: #137BC1;
        --xd-blue-dark: #0F6299;
        --xd-blue-soft: #E8F2FA;

        /* 상태 = 역할당 하나씩만 */
        --xd-success: #07883F;
        --xd-success-soft: #E7F5ED;
        --xd-warn: #F36F21;
        --xd-warn-text: #B24A0C;
        --xd-warn-soft: #FDF0E6;
        --xd-danger: #D71920;
        --xd-danger-soft: #FDECEC;

        /* 표면 */
        --xd-bg: #F6F7F9;
        --xd-surface: #FFFFFF;
        --xd-border: #DDE3E8;
        --xd-border-soft: #EDF1F4;
        --xd-text: #17191C;
        --xd-muted: #647080;

        /* 하위 호환 별칭 (민트·보라 장식 → 역할색/중립으로 대체) */
        --xd-mint: var(--xd-success-soft);
        --xd-mint-text: var(--xd-success);
        --xd-lavender: #EEF2F6;
        --xd-red: var(--xd-danger-soft);
        --xd-red-text: var(--xd-danger);
        --xd-amber: var(--xd-warn-soft);
        --xd-amber-text: var(--xd-warn-text);

        /* 형태 */
        --xd-radius: 16px;
        --xd-radius-sm: 12px;
        --xd-radius-xs: 8px;
        --xd-shadow: 0 1px 2px rgba(23,25,28,.04),
                     0 6px 16px rgba(23,25,28,.05);

        /* 폰트 — 외부 CDN 없이 OS 내장 폰트만 사용합니다. */
        --xd-font: Pretendard, "Apple SD Gothic Neo", "Noto Sans KR",
                   "Malgun Gothic", -apple-system, BlinkMacSystemFont,
                   "Segoe UI", sans-serif,
                   "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji";
        --xd-font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo,
                        Consolas, "D2Coding", monospace;
        /* 워드마크 전용. 라틴 전용 문자열이라 한글 폴백이 필요 없습니다.
           macOS Avenir Next → Windows Segoe UI → Linux Roboto 순. */
        --xd-font-brand: "Avenir Next", "Helvetica Neue", "Segoe UI",
                         Roboto, system-ui, sans-serif;
    }

    /* ============================================================
       타이포그래피 — 한글 가독성
       ============================================================ */
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"],
    [data-testid="stHeader"],
    [data-testid="stDialog"] {
        font-family: var(--xd-font);
        word-break: keep-all;
        overflow-wrap: break-word;
    }

    [data-testid="stAppViewContainer"] *,
    [data-testid="stSidebar"] *,
    [data-testid="stDialog"] * {
        font-family: inherit;
    }

    /* Streamlit 아이콘은 앱에 번들된 로컬 아이콘 폰트를 씁니다.
       위 규칙이 아이콘 글리프를 글자로 깨뜨리지 않도록 되돌립니다. */
    [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] *,
    [data-testid="stTooltipIcon"],
    [data-testid="stTooltipIcon"] *,
    [data-testid="stExpanderIcon"],
    span[class*="material-symbols"],
    .material-icons {
        font-family: "Material Symbols Rounded" !important;
    }

    code, pre, kbd, samp,
    [data-testid="stCode"] *,
    [data-testid="stJson"] * {
        font-family: var(--xd-font-mono) !important;
        word-break: normal;
    }

    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] li {
        line-height: 1.65;
    }

    /* Streamlit 기본 규칙이 word-break를 덮어써서 한글이 어절 중간에서
       잘리므로, 텍스트 요소에만 keep-all을 명시합니다. */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stAlert"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] summary p {
        word-break: keep-all !important;
        overflow-wrap: break-word !important;
    }

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: var(--xd-navy);
        font-weight: 800;
        letter-spacing: -.01em;
        line-height: 1.35;
    }

    /* 본문·입력·결과는 16px 이상을 유지합니다. */
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        font-size: 1rem;
    }

    /* 접근성: 키보드 포커스 */
    [data-testid="stAppViewContainer"] :focus-visible,
    [data-testid="stSidebar"] :focus-visible {
        outline: 2px solid var(--xd-blue);
        outline-offset: 2px;
        border-radius: 6px;
    }

    /* ============================================================
       레이아웃
       ============================================================ */
    [data-testid="stAppViewContainer"] {
        background: var(--xd-bg);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* ============================================================
       사이드바
       ============================================================ */
    [data-testid="stSidebar"] > div:first-child {
        background: var(--xd-surface);
        border-right: 1px solid var(--xd-border);
    }

    [data-testid="stSidebar"] {
        color: var(--xd-text);
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.15rem;
        margin-bottom: 0;
    }

    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
        color: var(--xd-navy);
        font-weight: 700;
        font-size: .88rem;
    }

    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: var(--xd-surface);
        border: 1px solid var(--xd-border);
        border-radius: var(--xd-radius-sm);
    }

    [data-testid="stSidebar"] [data-testid="stTextInput"] input,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        border-radius: var(--xd-radius-sm);
        border-color: var(--xd-border);
    }

    /* ============================================================
       앱 헤더
       ============================================================ */
    .xd-appbar {
        background: var(--xd-surface);
        border: 1px solid var(--xd-border);
        border-radius: 20px;
        box-shadow: var(--xd-shadow);
        padding: 1.25rem 1.6rem 1.35rem;
        margin-bottom: .9rem;
    }

    .xd-appbar-body {
        min-width: 0;
    }

    /* 대회명 줄. K-DS 로고가 이 줄에 작게 들어갑니다. */
    .xd-appbar-kicker {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: .55rem;
        font-size: .8rem;
        line-height: 1.5;
        margin-bottom: .55rem;
    }

    /* 원본 비율 유지: 높이만 지정하고 width는 auto */
    .xd-appbar-logo {
        height: 34px;
        width: auto;
        display: block;
        flex: none;
        margin-right: .35rem;
    }

    .xd-appbar-event {
        color: var(--xd-muted);
        font-weight: 600;
        letter-spacing: .01em;
    }

    .xd-appbar-team {
        color: var(--xd-navy);
        font-weight: 700;
    }

    .xd-appbar-sep {
        color: var(--xd-border);
    }

    .xd-appbar-title {
        font-family: var(--xd-font-brand) !important;
        color: var(--xd-navy);
        font-size: 2.05rem;
        font-weight: 700;
        letter-spacing: -.035em;
        line-height: 1.15;
        margin: 0 0 .35rem 0;
    }

    .xd-appbar-desc {
        color: var(--xd-muted);
        font-size: .95rem;
        line-height: 1.55;
        margin: 0;
        max-width: 62ch;
    }

    .xd-appbar-badges {
        display: flex;
        flex-wrap: wrap;
        gap: .4rem;
        margin-top: .7rem;
    }

    .xd-badge {
        display: inline-block;
        padding: .2rem .55rem;
        border-radius: 999px;
        border: 1px solid var(--xd-border);
        background: var(--xd-navy-soft);
        color: var(--xd-navy);
        font-size: .76rem;
        font-weight: 700;
        letter-spacing: .01em;
    }

    /* 연구용 프로토타입 표시는 다른 배지와 같은 중립색을 쓰되,
       문구 자체는 헤더에서 숨기지 않습니다. */
    .xd-badge-research {
        background: var(--xd-navy-soft);
        border-color: var(--xd-border);
        color: var(--xd-navy);
    }

    /* ============================================================
       단계 표시 — session_state.stage를 읽어 시각적으로만 표현
       ============================================================ */
    .xd-steps {
        display: flex;
        align-items: stretch;
        gap: .5rem;
        margin: 0 0 1.1rem 0;
    }

    .xd-step {
        flex: 1;
        min-width: 0;
        display: flex;
        align-items: center;
        gap: .5rem;
        padding: .55rem .8rem;
        border: 1px solid var(--xd-border);
        border-radius: var(--xd-radius-sm);
        background: var(--xd-surface);
    }

    .xd-step-num {
        flex: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.35rem;
        height: 1.35rem;
        border-radius: 50%;
        background: #EDF1F4;
        color: #8B97A3;
        font-size: .74rem;
        font-weight: 800;
    }

    .xd-step-label {
        color: #8B97A3;
        font-size: .84rem;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .xd-step-done .xd-step-num {
        background: var(--xd-success-soft);
        color: var(--xd-success);
    }

    .xd-step-done .xd-step-label {
        color: var(--xd-navy);
    }

    .xd-step-current {
        border-color: var(--xd-blue);
        background: var(--xd-blue-soft);
        box-shadow: inset 0 0 0 1px var(--xd-blue);
    }

    .xd-step-current .xd-step-num {
        background: var(--xd-blue);
        color: #FFFFFF;
    }

    .xd-step-current .xd-step-label {
        color: var(--xd-blue-dark);
        font-weight: 700;
    }

    /* 분석이 실행되는 동안 해당 단계에 회전 표시와 '진행 중'을 붙입니다. */
    .xd-step-running {
        border-color: var(--xd-blue);
        background: var(--xd-blue-soft);
        box-shadow: inset 0 0 0 1px var(--xd-blue);
    }

    .xd-step-running .xd-step-num {
        background: transparent;
        color: transparent;
        border: 2px solid rgba(19,123,193,.25);
        border-top-color: var(--xd-blue);
        animation: xd-spin .8s linear infinite;
    }

    .xd-step-running .xd-step-label {
        color: var(--xd-blue-dark);
        font-weight: 700;
    }

    .xd-step-tag {
        flex: none;
        margin-left: auto;
        padding: .08rem .4rem;
        border-radius: 999px;
        background: var(--xd-blue);
        color: #FFFFFF;
        font-size: .68rem;
        font-weight: 700;
        letter-spacing: .02em;
    }

    @keyframes xd-spin {
        to { transform: rotate(360deg); }
    }

    /* 움직임을 줄이도록 설정한 사용자에게는 회전을 멈춥니다. */
    @media (prefers-reduced-motion: reduce) {
        .xd-step-running .xd-step-num {
            animation: none;
            background: var(--xd-blue);
            color: #FFFFFF;
            border-color: var(--xd-blue);
        }
    }

    /* ============================================================
       카드 — 모든 카드가 같은 규칙을 따릅니다.
       ============================================================ */
    .xd-card {
        background: var(--xd-surface);
        border: 1px solid var(--xd-border);
        border-radius: var(--xd-radius);
        padding: 1.1rem 1.25rem;
        box-shadow: var(--xd-shadow);
        margin-bottom: .85rem;
    }

    .xd-card-title {
        color: var(--xd-navy);
        font-weight: 800;
        font-size: 1.02rem;
        line-height: 1.45;
        margin-bottom: .3rem;
    }

    .xd-muted {
        color: var(--xd-muted);
        font-size: .9rem;
        line-height: 1.6;
    }

    /* ---- 칩 ---- */
    .xd-chip-row {
        display: flex;
        gap: .4rem;
        flex-wrap: wrap;
        margin-top: .6rem;
    }

    .xd-chip {
        display: inline-block;
        padding: .26rem .58rem;
        border-radius: 999px;
        background: var(--xd-navy-soft);
        border: 1px solid var(--xd-border);
        color: var(--xd-navy);
        font-size: .8rem;
        font-weight: 650;
    }

    /* 기능 ON/OFF는 상태색이 아니라 중립색 + 점으로 구분합니다.
       초록은 '검증 통과' 같은 실제 판정에만 남겨 둡니다. */
    .xd-chip-on {
        background: var(--xd-navy-soft);
        border-color: var(--xd-border);
        color: var(--xd-navy);
    }

    .xd-chip-on::before,
    .xd-chip-off::before {
        content: "";
        display: inline-block;
        width: .38rem;
        height: .38rem;
        border-radius: 50%;
        margin-right: .35rem;
        vertical-align: .04rem;
        background: var(--xd-navy);
    }

    .xd-chip-off {
        background: #F4F6F8;
        border-color: var(--xd-border-soft);
        color: #8B97A3;
    }

    .xd-chip-off::before {
        background: #C2CBD3;
    }

    /* ---- 근거 연결 블록: 상태는 좌측 선으로만 ---- */
    .xd-claim {
        padding: .85rem 1rem;
        border: 1px solid var(--xd-border);
        border-left: 4px solid var(--xd-border);
        border-radius: var(--xd-radius-sm);
        background: var(--xd-surface);
        margin-bottom: .6rem;
        line-height: 1.65;
    }

    .xd-claim-supported   { border-left-color: var(--xd-success); }
    .xd-claim-partial     { border-left-color: var(--xd-warn); }
    .xd-claim-unsupported { border-left-color: var(--xd-danger); }

    .xd-evidence-id {
        display: inline-block;
        margin-right: .25rem;
        padding: .16rem .44rem;
        border-radius: 6px;
        background: var(--xd-navy-soft);
        border: 1px solid var(--xd-border);
        color: var(--xd-navy);
        font-size: .76rem;
        font-weight: 700;
        font-family: var(--xd-font-mono) !important;
    }

    /* [발표자료 반영] 추가 질문에 의한 판단 유예 상태
       응급 경고와 구분하면서도, 감별진단 전에 중요한 정보를 먼저
       확인한다는 제품 원칙과 질문 우선순위를 화면에 드러냅니다. */
    .xd-clarify-hold {
        background: var(--xd-surface);
        border: 1px solid var(--xd-border);
        border-left: 4px solid var(--xd-warn);
        border-radius: var(--xd-radius);
        box-shadow: var(--xd-shadow);
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
    }

    .xd-clarify-hold-head,
    .xd-question-title {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: .5rem;
    }

    .xd-clarify-hold-title {
        color: var(--xd-navy);
        font-size: 1.05rem;
        font-weight: 800;
    }

    .xd-clarify-hold-desc {
        color: var(--xd-muted);
        line-height: 1.65;
        margin-top: .4rem;
    }

    .xd-priority {
        display: inline-block;
        flex: 0 0 auto;
        min-width: 2.7rem;
        padding: .15rem .48rem;
        border-radius: 6px;
        text-align: center;
        font-size: .75rem;
        font-weight: 800;
        line-height: 1.35;
    }

    .xd-priority-required {
        background: var(--xd-danger-soft);
        color: var(--xd-danger);
    }

    .xd-priority-important {
        background: var(--xd-warn-soft);
        color: var(--xd-warn-text);
    }

    .xd-priority-optional {
        background: var(--xd-blue-soft);
        color: var(--xd-blue-dark);
    }

    /* ============================================================
       Streamlit 기본 컴포넌트 정돈
       ============================================================ */
    div.stButton > button {
        border-radius: var(--xd-radius-sm);
        font-weight: 700;
        border: 1px solid var(--xd-border);
    }

    div.stButton > button[kind="primary"],
    div.stFormSubmitButton > button[kind="primary"] {
        background: var(--xd-blue);
        border: 1px solid var(--xd-blue);
        border-radius: var(--xd-radius-sm);
        font-weight: 750;
    }

    div.stButton > button[kind="primary"]:hover,
    div.stFormSubmitButton > button[kind="primary"]:hover {
        background: var(--xd-blue-dark);
        border-color: var(--xd-blue-dark);
    }

    div.stFormSubmitButton > button {
        border-radius: var(--xd-radius-sm);
        font-weight: 700;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        border-radius: var(--xd-radius-sm);
        line-height: 1.65;
    }

    div[data-testid="stMetric"] {
        background: var(--xd-surface);
        border: 1px solid var(--xd-border);
        border-radius: var(--xd-radius-sm);
        padding: .75rem .85rem;
    }

    div[data-testid="stExpander"] {
        background: var(--xd-surface);
        border: 1px solid var(--xd-border);
        border-radius: var(--xd-radius-sm);
        box-shadow: none;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 700;
        color: var(--xd-navy);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--xd-border);
        border-radius: var(--xd-radius-sm);
        overflow: hidden;
    }

    /* 탭: 선택 상태는 Streamlit 기본 빨강 대신 인터랙션 블루 */
    button[data-baseweb="tab"] p {
        font-weight: 700;
        font-size: .93rem;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--xd-blue) !important;
    }

    [data-baseweb="tab-highlight"] {
        background-color: var(--xd-blue) !important;
    }

    /* 알림 박스 — 역할색과 어긋나지 않게 톤만 맞춥니다. */
    [data-testid="stAlert"] {
        border-radius: var(--xd-radius-sm);
    }

    /* ============================================================
       푸터
       ============================================================ */
    .xd-footer {
        margin-top: 2.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--xd-border);
        color: var(--xd-muted);
        font-size: .84rem;
        line-height: 1.65;
    }

    .xd-footer-brand {
        color: var(--xd-navy);
        font-weight: 700;
        font-size: .9rem;
        margin-bottom: .15rem;
    }

    .xd-footer-note {
        margin-top: .5rem;
        max-width: 80ch;
    }

    .xd-footer a {
        color: var(--xd-blue);
        text-decoration: underline;
    }

    /* ============================================================
       결과 화면 (st.container(key="result_panel"))
       라이트 테마를 .streamlit/config.toml에서 고정했으므로
       기존의 광범위한 !important 덮어쓰기는 최소한만 남깁니다.
       ============================================================ */
    .st-key-result_panel,
    .st-key-result_panel [data-testid="stMarkdownContainer"] p,
    .st-key-result_panel [data-testid="stMarkdownContainer"] li {
        color: var(--xd-text);
    }

    .st-key-result_panel [data-testid="stMetricLabel"] * {
        color: var(--xd-muted);
    }

    .st-key-result_panel [data-testid="stMetricValue"] * {
        color: var(--xd-navy);
    }

    .st-key-result_panel [data-testid="stCaptionContainer"] p {
        color: var(--xd-muted);
    }

    /* 디버그·원시 JSON은 핵심 결과보다 낮은 위계로 낮춥니다. */
    .st-key-result_panel [data-testid="stExpander"]:has(summary) {
        background: var(--xd-surface);
    }

    /* ============================================================
       반응형
       ============================================================ */
    @media (max-width: 900px) {
        .xd-steps {
            flex-wrap: wrap;
        }
        .xd-step {
            flex: 1 1 45%;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }
        /* 모바일에서는 로고가 제목 위로 올라갑니다. */
        .xd-appbar {
            flex-direction: column;
            align-items: flex-start;
            gap: .9rem;
            padding: 1.1rem 1.15rem;
            border-radius: var(--xd-radius);
        }
        .xd-appbar-logo {
            padding-right: 0;
            padding-bottom: .85rem;
            border-right: none;
            border-bottom: 1px solid var(--xd-border);
            width: 100%;
        }
        .xd-appbar-logo img {
            height: 44px;
        }
        /* 좁은 화면에서 구분자만 줄 끝에 남지 않도록 감춥니다. */
        .xd-appbar-sep {
            display: none;
        }
        .xd-appbar-kicker {
            gap: .1rem .5rem;
        }
        .xd-appbar-title {
            font-size: 1.6rem;
        }
        .xd-appbar-desc {
            font-size: .9rem;
        }
        .xd-step {
            flex: 1 1 100%;
        }
        .xd-card {
            padding: .95rem 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. 세션 상태와 공통 유틸리티
# ============================================================

def initialize_state() -> None:
    defaults = {
        "notice_confirmed": False,
        "stage": "input",  # input | clarification | result
        "original_case": "",
        "clarification": None,
        "analysis_result": None,
        "analysis_mode": None,
        "analysis_domain": None,
        "analysis_top_k": None,
        "analysis_dense_weight": None,
        "analysis_index_dir": None,
        "feedback": None,
        "case_text": DEFAULT_CASE,
        "example_selector": "급성 흉통",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_analysis() -> None:
    st.session_state.stage = "input"
    st.session_state.original_case = ""
    st.session_state.clarification = None
    st.session_state.analysis_result = None
    st.session_state.analysis_mode = None
    st.session_state.analysis_domain = None
    st.session_state.analysis_top_k = None
    st.session_state.analysis_dense_weight = None
    st.session_state.analysis_index_dir = None
    st.session_state.feedback = None

    for key in list(st.session_state.keys()):
        if key.startswith("clarify_"):
            del st.session_state[key]


def to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def escaped(text: Any) -> str:
    return html.escape(str(text or ""))


@st.cache_data(show_spinner=False)
def logo_data_uri(path_str: str) -> str:
    """
    로고 파일을 data URI로 변환합니다. 표시 전용 헬퍼입니다.

    외부 CDN 없이 오프라인에서 동작하도록 파일을 인라인합니다.
    파일이 없거나 읽을 수 없으면 빈 문자열을 반환하며,
    호출부는 로고 없이 텍스트만으로 정상 렌더링합니다.
    """
    try:
        path = Path(path_str)
        if not path.is_file():
            return ""
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""

    return f"data:image/png;base64,{encoded}"


def show_notice_once() -> None:
    if st.session_state.notice_confirmed:
        return

    if hasattr(st, "dialog"):
        @st.dialog("MedCheck AI 이용 전 안내")
        def _notice_dialog() -> None:
            st.markdown(NOTICE_MD)
            confirmed = st.checkbox(
                "위 내용을 확인했으며 연구·교육용 프로토타입임을 이해했습니다.",
                key="notice_checkbox",
            )
            if st.button(
                "확인하고 시작",
                type="primary",
                use_container_width=True,
                disabled=not confirmed,
            ):
                st.session_state.notice_confirmed = True
                st.rerun()

        _notice_dialog()
        st.stop()

    st.markdown("## MedCheck AI 이용 전 안내")
    st.markdown(NOTICE_MD)
    confirmed = st.checkbox(
        "위 내용을 확인했습니다.",
        key="notice_checkbox_fallback",
    )
    if st.button(
        "확인하고 시작",
        type="primary",
        disabled=not confirmed,
    ):
        st.session_state.notice_confirmed = True
        st.rerun()
    st.stop()


@st.cache_resource(show_spinner="의학지식 인덱스를 불러오는 중...")
def load_index(path: str) -> MedicalHybridIndex:
    return MedicalHybridIndex.load(path)


@st.cache_resource(show_spinner="로컬 의료 언어모델을 불러오는 중...")
def load_llm(model_cache_key: str) -> LocalTransformersClient:
    _ = model_cache_key
    return LocalTransformersClient()


def get_runtime(
    *,
    index_dir: str,
    top_k: int,
    dense_weight: float,
) -> tuple[MedicalHybridIndex, LocalTransformersClient, MedCheckPipeline]:
    resolved_index = str(Path(index_dir).expanduser().resolve())
    model_cache_key = "|".join(
        [
            os.getenv("LOCAL_LLM_MODEL", "models/Qwen2.5-3B-Instruct"),
            os.getenv("LOCAL_LLM_ADAPTER", ""),
            os.getenv("MEDCHECK_DEVICE", "auto"),
        ]
    )

    index = load_index(resolved_index)
    llm = load_llm(model_cache_key)
    pipeline = MedCheckPipeline(
        index,
        llm,
        top_k=top_k,
        dense_weight=dense_weight,
        bm25_weight=1.0 - dense_weight,
    )
    return index, llm, pipeline


def render_feature_chips(mode_key: str) -> None:
    features = MODE_FEATURES[mode_key]
    chips = []
    for label, enabled in features.items():
        cls = "xd-chip xd-chip-on" if enabled else "xd-chip xd-chip-off"
        value = "ON" if enabled else "OFF"
        chips.append(
            f'<span class="{cls}">{escaped(label)} · {value}</span>'
        )

    st.markdown(
        '<div class="xd-chip-row">' + "".join(chips) + "</div>",
        unsafe_allow_html=True,
    )


def render_stage_indicator(running_step: int | None = None) -> None:
    """
    입력 → 추가 정보 → 분석 → 결과의 4단계를 시각적으로만 표시합니다.

    session_state.stage를 읽기만 하며, 새로운 상태값이나 전환 조건을
    만들지 않습니다. '근거 기반 분석'은 st.status가 실행되는 동안에만
    존재하는 단계이므로, 결과 화면에서 완료로 표시합니다.

    running_step을 주면 같은 자리(placeholder)를 다시 그려, 분석이
    실행되는 동안 해당 단계에 회전 표시와 '진행 중' 배지를 붙입니다.
    화면 표시 전용이며 상태 전이에는 관여하지 않습니다.
    """
    global _STAGE_SLOT

    if running_step is None or _STAGE_SLOT is None:
        _STAGE_SLOT = st.empty()

    stage = st.session_state.get("stage", "input")
    current = STAGE_TO_STEP_INDEX.get(stage, 0)

    parts = ['<div class="xd-steps">']
    for index, label in enumerate(STAGE_STEPS):
        tag = ""
        if running_step is not None and index == running_step:
            # 실행 중인 단계
            state_cls = "xd-step-running"
            tag = '<span class="xd-step-tag">진행 중</span>'
        elif running_step is not None:
            # 실행 중에는 이미 지나온 단계만 완료로 표시합니다.
            # (입력에서 바로 분석에 들어가면 '추가 정보 확인'은 건너뜁니다.)
            state_cls = "xd-step-done" if index <= current else ""
        elif index == current:
            state_cls = "xd-step-current"
        elif index < current:
            state_cls = "xd-step-done"
        else:
            state_cls = ""

        parts.append(
            f'<div class="xd-step {state_cls}">'
            f'<span class="xd-step-num">{index + 1}</span>'
            f'<span class="xd-step-label">{escaped(label)}</span>'
            f"{tag}"
            "</div>"
        )
    parts.append("</div>")

    _STAGE_SLOT.markdown("".join(parts), unsafe_allow_html=True)


# def infer_urgency(decision: Any) -> str:
#     levels = {
#         str(getattr(item, "urgency", "일반"))
#         for item in getattr(decision, "red_flags", [])
#     }

#     if "즉시" in levels:
#         return "즉시"
#     if "긴급" in levels:
#         return "긴급"
#     if levels:
#         return "주의"
#     return "일반"


def evidence_badges(evidence_ids: list[str]) -> str:
    if not evidence_ids:
        return '<span class="xd-muted">연결된 문서 근거 없음</span>'

    return "".join(
        f'<span class="xd-evidence-id">{escaped(evidence_id)}</span>'
        for evidence_id in evidence_ids
    )


# ============================================================
# 4. 사이드바
# ============================================================

def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("## MedCheck AI")
        st.caption("XDoctors · eXplainable Doctors")
        st.caption(
            "근거를 연결하고 불확실성을 드러내는 임상 의사결정 지원 시스템"
        )
        st.divider()

        settings_tab, guide_tab, faq_tab = st.tabs(
            ["⚙️ 설정", "📖 가이드", "❓ FAQ"]
        )

        with settings_tab:
            mode_label = st.selectbox(
                "분석 모드",
                list(MODE_LABEL_TO_KEY.keys()),
                index=3,
                help=(
                    "MedCheck AI는 추가 질문, 구조화된 감별진단, "
                    "근거 검증과 답변 보류를 모두 사용합니다."
                ),
            )
            mode_key = MODE_LABEL_TO_KEY[mode_label]

            domain_options = ["전체"] + sorted(
                set(DOMAIN_CODE_TO_NAME.values())
            )
            domain = st.selectbox(
                "진료과 필터",
                domain_options,
                index=0,
            )

            top_k = st.slider(
                "검색 근거 수",
                min_value=1,
                max_value=10,
                value=3,
                disabled=mode_key == "base",
            )

            dense_weight = st.slider(
                "Dense 검색 가중치",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.05,
                disabled=mode_key == "base",
                help="BM25 가중치는 자동으로 1 - Dense 가중치로 설정됩니다.",
            )

            with st.expander("고급 설정"):
                index_dir = st.text_input(
                    "인덱스 경로",
                    value=os.getenv(
                        "MEDCHECK_INDEX_DIR",
                        "artifacts/medical_index",
                    ),
                )
                show_raw_json = st.checkbox(
                    "원본 JSON 표시",
                    value=False,
                )

            st.divider()
            st.markdown("**현재 모드 기능**")
            render_feature_chips(mode_key)

            if st.session_state.stage != "input":
                st.write("")
                if st.button(
                    "새 분석 시작",
                    use_container_width=True,
                ):
                    reset_analysis()
                    st.rerun()

        with guide_tab:
            st.markdown(
                """
                ### 사례 입력 권장 항목

                - 나이와 성별
                - 주호소와 동반 증상
                - 발생 시점과 지속 시간
                - 활력징후
                - 과거력과 복용약
                - 이용 가능한 검사 결과

                ### MedCheck AI 흐름

                1. 환자 사례 구조화
                2. 중요 정보 누락 확인
                3. 필요한 경우 추가 질문
                4. 의료지식 검색
                5. 감별진단 비교
                6. 근거 검증 및 답변 보류
                """
            )

        with faq_tab:
            with st.expander("확정 진단을 제공하나요?"):
                st.write(
                    "아닙니다. 감별 가능한 후보와 그 근거, "
                    "반대 또는 누락 근거, 다음 확인 단계를 제공합니다."
                )

            with st.expander("왜 추가 질문을 하나요?"):
                st.write(
                    "중요 정보가 부족한 상태에서 바로 진단 후보를 생성하는 "
                    "위험을 줄이기 위해서입니다."
                )

            with st.expander("근거 ID는 무엇인가요?"):
                st.write(
                    "검색된 AI Hub 전문 의학지식 문서를 E1, E2와 같이 "
                    "표시한 식별자입니다."
                )

        st.divider()
        st.warning(
            "연구·교육용 프로토타입입니다. "
            "확정 진단과 처방을 대체하지 않습니다."
        )

        st.divider()
        st.caption("**데이터 출처**")

        if AIHUB_LOGO_PATH.is_file():
            st.image(str(AIHUB_LOGO_PATH), width=84)

        st.caption(AIHUB_SOURCE_NOTE)
        st.caption(f"[AI Hub 전문 의학지식 데이터셋 바로가기]({AIHUB_DATASET_URL})")

    return {
        "mode_label": mode_label,
        "mode_key": mode_key,
        "domain": None if domain == "전체" else domain,
        "top_k": top_k,
        "dense_weight": dense_weight,
        "index_dir": index_dir,
        "show_raw_json": show_raw_json,
    }


# ============================================================
# 5. 입력 화면
# ============================================================

def render_hero() -> None:
    """앱 헤더. 로고 파일이 없으면 로고 영역 없이 텍스트만 렌더링합니다."""
    logo_uri = logo_data_uri(str(KDS_LOGO_PATH))

    logo_block = (
        f'<img class="xd-appbar-logo" src="{logo_uri}" '
        'alt="K-DS 컨소시엄 로고">'
        if logo_uri
        else ""
    )

    st.markdown(
        f"""
        <div class="xd-appbar">
            <div class="xd-appbar-body">
                <div class="xd-appbar-kicker">
                    {logo_block}
                    <span class="xd-appbar-event">
                        2026 K-Data Science Hackathon
                    </span>
                    <span class="xd-appbar-sep">|</span>
                    <span class="xd-appbar-team">
                        XDoctors · eXplainable Doctors
                    </span>
                </div>
                <h1 class="xd-appbar-title">MedCheck AI</h1>
                <p class="xd-appbar-desc">
                    근거를 연결하고 불확실성을 드러내는 임상 의사결정 지원 시스템
                </p>
                <div class="xd-appbar-badges">
                    <span class="xd-badge">Local LLM</span>
                    <span class="xd-badge">Evidence-grounded</span>
                    <span class="xd-badge">Safety-aware</span>
                    <span class="xd-badge xd-badge-research">
                        연구·교육용 프로토타입
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_input_stage(config: dict[str, Any]) -> None:
    mode_key = config["mode_key"]

    st.markdown(
        """
        <div class="xd-card">
            <div class="xd-card-title">1단계 · 환자 사례 입력</div>
            <div class="xd-muted">
                임상 판단에 필요한 정보만 입력하세요.
                이름, 연락처, 주민등록번호 등 개인 식별정보는 입력하지 않습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_example, col_mode = st.columns([1.2, 1])

    with col_example:
        selected_example = st.selectbox(
            "데모 사례",
            list(EXAMPLE_CASES.keys()),
            key="example_selector",
        )

        if selected_example != "직접 입력":
            selected_text = EXAMPLE_CASES[selected_example]
            if st.session_state.get("_last_example") != selected_example:
                st.session_state.case_text = selected_text
                st.session_state._last_example = selected_example

    with col_mode:
        st.markdown("**선택된 분석 구성**")
        st.write(config["mode_label"])
        st.caption("분석 모드는 왼쪽 사이드바에서 변경할 수 있습니다.")
        render_feature_chips(mode_key)

    patient_case = st.text_area(
        "환자 사례",
        key="case_text",
        height=190,
        placeholder=(
            "예: 67세 남성이 30분 전부터 압박성 흉통과 "
            "식은땀을 호소한다..."
        ),
        help=(
            "나이, 성별, 주호소, 발생 시점, 활력징후, 과거력, "
            "복용약, 검사 결과를 가능한 범위에서 입력하세요."
        ),
    )

    with st.expander("입력에 포함하면 좋은 항목"):
        st.markdown(
            """
            <div class="xd-chip-row" style="margin-top:0;">
                <span class="xd-chip">나이·성별</span>
                <span class="xd-chip">주호소</span>
                <span class="xd-chip">발생 시점</span>
                <span class="xd-chip">활력징후</span>
                <span class="xd-chip">과거력</span>
                <span class="xd-chip">복용약</span>
                <span class="xd-chip">검사 결과</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "항목이 많을수록 추가 질문이 줄고 근거 연결이 정확해집니다. "
            "모르는 항목은 비워 두어도 됩니다."
        )

    st.write("")

    if st.button(
        "분석 시작",
        type="primary",
        use_container_width=True,
    ):
        if len(patient_case.strip()) < 10:
            st.error("환자 사례를 10자 이상 입력하세요.")
            return

        execute_initial_analysis(
            patient_case=patient_case.strip(),
            config=config,
        )


# ============================================================
# 6. 분석 실행
# ============================================================

def execute_initial_analysis(
    *,
    patient_case: str,
    config: dict[str, Any],
) -> None:
    mode_key = config["mode_key"]

    st.session_state.original_case = patient_case
    st.session_state.analysis_mode = mode_key
    st.session_state.analysis_domain = config["domain"]
    st.session_state.analysis_top_k = config["top_k"]
    st.session_state.analysis_dense_weight = config["dense_weight"]
    st.session_state.analysis_index_dir = config["index_dir"]

    render_stage_indicator(running_step=2)  # 표시 전용

    try:
        model_cache_key = os.getenv(
            "LOCAL_LLM_MODEL",
            "default-local-model",
        )
        llm = load_llm(model_cache_key)

        if mode_key == "base":
            with st.status(
                "Base LLM 답변을 생성하고 있습니다.",
                expanded=True,
            ) as status:
                status.write("환자 사례를 로컬 언어모델에 전달")
                result = run_base_llm(
                    llm=llm,
                    patient_case=patient_case,
                )
                status.update(
                    label="Base LLM 답변 생성 완료",
                    state="complete",
                )

        else:
            index, llm, pipeline = get_runtime(
                index_dir=config["index_dir"],
                top_k=config["top_k"],
                dense_weight=config["dense_weight"],
            )

            if mode_key == "rag":
                with st.status(
                    "관련 의학지식을 검색하고 답변을 생성하고 있습니다.",
                    expanded=True,
                ) as status:
                    status.write("Hybrid Retrieval 실행")
                    status.write("검색 근거를 로컬 언어모델에 전달")
                    result = run_rag_llm(
                        index=index,
                        llm=llm,
                        patient_case=patient_case,
                        domain=config["domain"],
                        top_k=config["top_k"],
                        dense_weight=config["dense_weight"],
                        bm25_weight=1.0 - config["dense_weight"],
                    )
                    status.update(
                        label="RAG 답변 생성 완료",
                        state="complete",
                    )

            elif mode_key == "structured":
                with st.status(
                    "구조화된 임상 판단을 생성하고 있습니다.",
                    expanded=True,
                ) as status:
                    status.write("관련 의료지식 검색")
                    status.write("감별진단과 반대·누락 근거 생성")
                    result = pipeline.run(
                        patient_case,
                        domain=config["domain"],
                        verify=False,
                        skip_clarification=True,
                    )
                    status.update(
                        label="Structured RAG 분석 완료",
                        state="complete",
                    )

            elif mode_key == "medcheck":
                with st.status(
                    "환자 사례의 중요 누락 정보를 확인하고 있습니다.",
                    expanded=True,
                ) as status:
                    status.write("환자 사례 구조화")
                    status.write("추가 질문 필요 여부 판단")
                    result = pipeline.run(
                        patient_case,
                        domain=config["domain"],
                        verify=True,
                        skip_clarification=False,
                    )

                    if isinstance(result, ClarificationAssessment):
                        status.update(
                            label="추가 확인 정보가 필요합니다.",
                            state="complete",
                        )
                    else:
                        status.update(
                            label="MedCheck AI 분석 완료",
                            state="complete",
                        )

            else:
                raise ValueError(
                    f"지원하지 않는 분석 모드입니다: {mode_key}"
                )

        if isinstance(result, ClarificationAssessment):
            st.session_state.clarification = result
            st.session_state.stage = "clarification"
            st.session_state.analysis_result = None
        else:
            st.session_state.analysis_result = result
            st.session_state.stage = "result"

        st.rerun()

    except Exception as exc:
        st.exception(exc)


def execute_clarified_analysis(
    *,
    answers: dict[str, str],
) -> None:
    clarification = st.session_state.clarification
    patient_case = st.session_state.original_case

    if not isinstance(clarification, ClarificationAssessment):
        st.error("추가 질문 상태가 유효하지 않습니다.")
        return

    render_stage_indicator(running_step=2)  # 표시 전용

    try:
        _, _, pipeline = get_runtime(
            index_dir=st.session_state.analysis_index_dir,
            top_k=st.session_state.analysis_top_k,
            dense_weight=st.session_state.analysis_dense_weight,
        )

        with st.status(
            "추가 정보를 반영하여 MedCheck AI 분석을 진행합니다.",
            expanded=True,
        ) as status:
            status.write("원래 환자 사례와 추가 답변 병합")
            status.write("관련 의료지식 검색")
            status.write("감별진단 비교")
            status.write("주장과 검색 근거 검증")

            result = pipeline.run(
                patient_case,
                domain=st.session_state.analysis_domain,
                verify=True,
                clarification=clarification,
                clarification_answers=answers,
            )

            if isinstance(result, ClarificationAssessment):
                raise RuntimeError(
                    "추가 답변을 반영한 뒤에도 ClarificationAssessment가 "
                    "반환되었습니다. pipeline.run()의 답변 병합 분기를 확인하세요."
                )

            status.update(
                label="MedCheck AI 분석 완료",
                state="complete",
            )

        st.session_state.analysis_result = result
        st.session_state.stage = "result"
        st.rerun()

    except Exception as exc:
        st.exception(exc)


# ============================================================
# 7. 추가 질문 화면
# ============================================================

def render_clarification_hold_notice(
    assessment: ClarificationAssessment,
) -> None:
    """일반 추가 질문 단계의 판단 유예 상태와 우선순위를 표시합니다.

    [발표자료 반영] 모델 출력이나 상태 전이를 변경하지 않고, 이미 생성된
    ``requires_clarification`` 및 질문 ``priority``를 사용자에게 설명하는
    표현 계층 전용 함수입니다. 응급 상황은 기존 행동 카드가 담당합니다.
    """
    counts = {"필수": 0, "중요": 0, "보조": 0}
    for question in assessment.questions:
        if question.priority in counts:
            counts[question.priority] += 1

    count_badges = "".join(
        f'<span class="xd-priority {css_class}">'
        f'{escaped(label)} {counts[label]}</span>'
        for label, css_class in (
            ("필수", "xd-priority-required"),
            ("중요", "xd-priority-important"),
            ("보조", "xd-priority-optional"),
        )
        if counts[label]
    )

    st.markdown(
        f"""
        <div class="xd-clarify-hold">
            <div class="xd-clarify-hold-head">
                <span class="xd-clarify-hold-title">
                    현재는 판단을 유예하고 추가 정보를 확인합니다
                </span>
                {count_badges}
            </div>
            <div class="xd-clarify-hold-desc">
                현재 정보만으로 바로 감별진단을 생성하지 않고,
                결과와 안전성에 영향을 주는 항목부터 우선 확인합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_clarification_stage() -> None:
    assessment = st.session_state.clarification

    if not isinstance(assessment, ClarificationAssessment):
        st.error("추가 질문 결과를 불러올 수 없습니다.")
        if st.button("처음으로 돌아가기"):
            reset_analysis()
            st.rerun()
        return

    st.markdown(
        """
        <div class="xd-card">
            <div class="xd-card-title">2단계 · 추가 정보 확인</div>
            <div class="xd-muted">
                중요한 정보가 부족한 상태에서 바로 감별진단을 생성하지 않고,
                분석 결과를 바꿀 수 있는 핵심 정보부터 확인합니다.
                모르는 항목은 '모름'으로 답해도 분석을 이어갈 수 있습니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # version 1
    # if assessment.urgent:
    #     st.error(
    #         assessment.urgent_message
    #         or (
    #             "응급 가능성이 있습니다. 추가 질문 응답과 관계없이 "
    #             "즉시 대면 의료 평가가 필요할 수 있습니다."
    #         )
    #     )

    if should_show_action_card(assessment=assessment):
        render_action_card(assessment=assessment)
    elif assessment.requires_clarification:
        render_clarification_hold_notice(assessment)

    if assessment.reason:
        st.info(assessment.reason)

    if assessment.structured_case:
        with st.expander("현재까지 구조화된 환자 정보"):
            st.json(to_plain(assessment.structured_case))

    with st.form("clarification_form", clear_on_submit=False):
        answers: dict[str, str] = {}

        for index, question in enumerate(
            assessment.questions,
            start=1,
        ):
            priority_class = {
                "필수": "xd-priority-required",
                "중요": "xd-priority-important",
                "보조": "xd-priority-optional",
            }.get(question.priority, "xd-priority-optional")
            st.markdown(
                f"""
                <div class="xd-card">
                    <div class="xd-card-title xd-question-title">
                        <span class="xd-priority {priority_class}">
                            {escaped(question.priority)}
                        </span>
                        <span>Q{index}. {escaped(question.question)}</span>
                    </div>
                    <div class="xd-muted">
                        {escaped(question.why_needed)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            widget_key = f"clarify_{question.question_id}"

            if question.answer_type == "yes_no":
                answer = st.radio(
                    "답변",
                    ["예", "아니오", "모름"],
                    key=widget_key,
                    horizontal=True,
                )

            elif question.answer_type == "choice":
                options = question.options or ["모름"]
                if "모름" not in options:
                    options = [*options, "모름"]
                answer = st.selectbox(
                    "답변",
                    options,
                    key=widget_key,
                )

            elif question.answer_type == "number":
                answer = st.text_input(
                    "답변",
                    placeholder="숫자 또는 측정값을 입력하세요. 모르면 '모름'으로 입력하세요.",
                    key=widget_key,
                )

            else:
                answer = st.text_input(
                    "답변",
                    placeholder="모르면 '모름'으로 입력하세요.",
                    key=widget_key,
                )

            answers[question.question_id] = str(answer).strip()

        submit_col, back_col = st.columns([3, 1])

        with submit_col:
            submitted = st.form_submit_button(
                "추가 정보를 반영하여 분석",
                type="primary",
                use_container_width=True,
            )

        with back_col:
            cancel = st.form_submit_button(
                "처음으로",
                use_container_width=True,
            )

    if cancel:
        reset_analysis()
        st.rerun()

    if submitted:
        required_unanswered = [
            question.question
            for question in assessment.questions
            if question.priority == "필수"
            and not answers.get(question.question_id, "").strip()
        ]

        if required_unanswered:
            st.warning(
                "필수 질문에는 답변하거나 '모름'이라고 입력하세요."
            )
            return

        execute_clarified_analysis(answers=answers)


# ============================================================
# 8. 결과 렌더링
# ============================================================

def render_text_result(result: Any, mode_key: str) -> None:
    title = (
        "Base LLM 답변"
        if mode_key == "base"
        else "LLM + RAG 답변"
    )

    st.subheader(title)

    st.markdown(
        f"""
        <div class="xd-card">
            <div class="xd-card-title">생성 답변</div>
            <div style="line-height:1.75; white-space:pre-wrap;">
                {escaped(getattr(result, "answer", ""))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence = getattr(result, "evidence", []) or []

    if evidence:
        st.subheader("검색된 의학 근거")
        render_evidence_documents(evidence)


def render_pipeline_result(
    result: PipelineResult,
    *,
    show_raw_json: bool,
) -> None:
    decision = result.decision
    verification = result.verification
    urgency = infer_urgency(decision)

    # if decision.abstain:
    #     st.error(
    #         "판단 보류: "
    #         + (
    #             decision.abstain_reason
    #             or "충분히 근거 있는 판단을 생성하지 못했습니다."
    #         )
    #     )
    # else:
    #     st.success(
    #         "검색 근거와 현재 입력 정보를 바탕으로 "
    #         "구조화된 의사결정 결과를 생성했습니다."
    #     )
    
    if should_show_action_card(decision=decision):
        render_action_card(decision=decision)
        
    else:
        st.success(
            "검색 근거와 현재 입력 정보를 바탕으로 "
            "구조화된 의사결정 결과를 생성했습니다."
        )
    
    
    

    # metric1, metric2, metric3, metric4 = st.columns(4)

    # metric1.metric("응급도", urgency)
    # metric2.metric(
    #     "감별진단",
    #     len(decision.differential_diagnoses),
    # )
    # metric3.metric(
    #     "검색 근거",
    #     len(result.evidence),
    # )
    # metric4.metric(
    #     "검증 결과",
    #     (
    #         "미실행"
    #         if verification is None
    #         else (
    #             "통과"
    #             if verification.overall_grounded
    #             and not verification.needs_abstention
    #             else "주의"
    #         )
    #     ),
    # )
    
    
    ui.render_metrics(
        urgency=urgency,
        n_diagnoses=len(decision.differential_diagnoses),
        n_evidence=len(result.evidence),
        verification=verification,
        has_red_flags=bool(decision.red_flags),
    )
    
    

    tabs = st.tabs(
        [
            "요약",
            "감별진단",
            "근거 연결",
            "검증 결과",
            "검색 문서",
        ]
    )

    with tabs[0]:
        render_summary_tab(decision)

    with tabs[1]:
        render_differential_tab(decision)

    with tabs[2]:
        render_evidence_link_tab(decision, verification)

    with tabs[3]:
        render_verification_tab(verification)

    with tabs[4]:
        render_evidence_documents(result.evidence)

    st.info(decision.safety_notice)

    if show_raw_json:
        with st.expander("원본 PipelineResult JSON"):
            st.json(result.model_dump())


# def render_summary_tab(decision: Any) -> None:
#     st.markdown(
#         f"""
#         <div class="xd-card">
#             <div class="xd-card-title">임상 문제 요약</div>
#             <div>{escaped(decision.case_summary)}</div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     left, right = st.columns(2)

#     with left:
#         st.subheader("구조화된 환자 정보")
#         st.json(to_plain(decision.structured_case))

#         st.subheader("응급징후")
#         if decision.red_flags:
#             red_flag_df = pd.DataFrame(
#                 [item.model_dump() for item in decision.red_flags]
#             )
#             st.dataframe(
#                 red_flag_df,
#                 use_container_width=True,
#                 hide_index=True,
#             )
#         else:
#             st.info("명시적으로 탐지된 응급징후가 없습니다.")

#     with right:
#         st.subheader("추가 확인이 필요한 정보")
#         if decision.missing_information:
#             missing_df = pd.DataFrame(
#                 [
#                     item.model_dump()
#                     for item in decision.missing_information
#                 ]
#             )
#             st.dataframe(
#                 missing_df,
#                 use_container_width=True,
#                 hide_index=True,
#             )
#         else:
#             st.info("추가 확인 정보가 없습니다.")

#         st.subheader("다음 단계")
#         if decision.next_steps:
#             step_df = pd.DataFrame(
#                 [item.model_dump() for item in decision.next_steps]
#             )
#             st.dataframe(
#                 step_df,
#                 use_container_width=True,
#                 hide_index=True,
#             )
#         else:
#             st.info("생성된 다음 단계가 없습니다.")

def render_summary_tab(decision: Any) -> None:
    st.markdown(
        f"""
        <div class="xd-card">
            <div class="xd-card-title">임상 문제 요약</div>
            <div>{escaped(decision.case_summary)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        ui.section_title("구조화된 환자 정보")
        ui.render_structured_case(decision.structured_case)

        ui.section_title("응급징후")
        ui.render_key_value_list(
            decision.red_flags,
            title_attr="finding",
            note_attr="reason",
            tag_attr="urgency",
            empty_text="탐지된 응급징후가 없습니다.",
        )

    with right:
        ui.section_title("추가 확인이 필요한 정보")
        ui.render_key_value_list(
            decision.missing_information,
            title_attr="question",
            note_attr="why_needed",
            tag_attr="priority",
            empty_text="추가로 확인할 정보가 없습니다.",
        )

        ui.section_title("다음 단계")
        ui.render_key_value_list(
            decision.next_steps,
            title_attr="action",
            note_attr="rationale",
            tag_attr="urgency",
            empty_text="제시된 다음 단계가 없습니다.",
        )


# def render_differential_tab(decision: Any) -> None:
#     diagnoses = sorted(
#         decision.differential_diagnoses,
#         key=lambda item: item.rank,
#     )

#     if not diagnoses:
#         st.warning("생성된 감별진단이 없습니다.")
#         return

#     for diagnosis in diagnoses:
#         with st.expander(
#             f"{diagnosis.rank}. {diagnosis.name}",
#             expanded=diagnosis.rank == 1,
#         ):
#             support_col, counter_col = st.columns(2)

#             with support_col:
#                 st.markdown("#### 지지 근거")
#                 if diagnosis.supporting_evidence:
#                     for item in diagnosis.supporting_evidence:
#                         st.write(f"- {item}")
#                 else:
#                     st.caption("명시된 지지 근거가 없습니다.")

#             with counter_col:
#                 st.markdown("#### 반대 또는 누락 근거")
#                 if diagnosis.contradicting_or_missing_evidence:
#                     for item in diagnosis.contradicting_or_missing_evidence:
#                         st.write(f"- {item}")
#                 else:
#                     st.caption("명시된 반대 또는 누락 근거가 없습니다.")

#             st.markdown("#### 추가 확인사항")
#             if diagnosis.recommended_checks:
#                 for item in diagnosis.recommended_checks:
#                     st.write(f"- {item}")
#             else:
#                 st.caption("추가 확인사항이 없습니다.")

#             st.markdown("#### 연결 문서")
#             st.markdown(
#                 evidence_badges(diagnosis.evidence_ids),
#                 unsafe_allow_html=True,
#             )

def render_differential_tab(decision: Any) -> None:
    ui.render_differential_diagnoses(decision.differential_diagnoses)




def render_evidence_link_tab(
    decision: Any,
    verification: Any | None,
) -> None:
    st.caption(
        "현재 구현은 감별진단·다음 단계 단위로 문서 근거를 연결합니다. "
        "검증기가 생성한 verdict는 주장 단위로 표시합니다."
    )

    st.subheader("감별진단별 근거 연결")

    for diagnosis in sorted(
        decision.differential_diagnoses,
        key=lambda item: item.rank,
    ):
        st.markdown(
            f"""
            <div class="xd-claim xd-claim-supported">
                <strong>{diagnosis.rank}. {escaped(diagnosis.name)}</strong><br>
                <span class="xd-muted">연결 근거</span><br>
                {evidence_badges(diagnosis.evidence_ids)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("다음 단계별 근거 연결")

    if decision.next_steps:
        for step in decision.next_steps:
            st.markdown(
                f"""
                <div class="xd-claim">
                    <strong>[{escaped(step.urgency)}] {escaped(step.action)}</strong><br>
                    <span class="xd-muted">{escaped(step.rationale)}</span><br><br>
                    {evidence_badges(step.evidence_ids)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("표시할 다음 단계가 없습니다.")

    if verification is not None and verification.verdicts:
        st.subheader("검증기 주장 단위 판정")

        for verdict in verification.verdicts:
            status = str(verdict.status)
            if status == "supported":
                css_class = "xd-claim-supported"
                status_label = "지지됨"
            elif status == "partially_supported":
                css_class = "xd-claim-partial"
                status_label = "부분 지지"
            else:
                css_class = "xd-claim-unsupported"
                status_label = "지지되지 않음"

            st.markdown(
                f"""
                <div class="xd-claim {css_class}">
                    <strong>{escaped(verdict.claim)}</strong><br>
                    <span class="xd-muted">판정: {status_label}</span><br>
                    <span>{escaped(verdict.reason)}</span><br><br>
                    {evidence_badges(verdict.evidence_ids)}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_verification_tab(verification: Any | None) -> None:
    if verification is None:
        st.info(
            "이 분석 모드에서는 근거 검증기를 실행하지 않았습니다."
        )
        return

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Grounded",
        "통과" if verification.overall_grounded else "주의",
    )
    c2.metric(
        "Unsupported claim rate",
        f"{verification.unsupported_claim_rate:.1%}",
    )
    c3.metric(
        "Verifier abstention",
        "필요" if verification.needs_abstention else "불필요",
    )

    if verification.reason:
        st.info(verification.reason)

    if verification.verdicts:
        verdict_df = pd.DataFrame(
            [item.model_dump() for item in verification.verdicts]
        )
        st.dataframe(
            verdict_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        ui.empty_note("생성된 주장별 검증 결과가 없습니다.")

    if verification.unsafe_recommendations:
        st.error(
            "위험 가능 권고: "
            + "; ".join(verification.unsafe_recommendations)
        )
    else:
        st.success("검증기가 탐지한 위험 가능 권고가 없습니다.")


def render_evidence_documents(evidence_items: list[Any]) -> None:
    if not evidence_items:
        st.info("표시할 검색 근거가 없습니다.")
        return

    for evidence in evidence_items:
        title = (
            f"[{evidence.evidence_id}] "
            f"{evidence.domain} · qa_id={evidence.qa_id}"
        )

        with st.expander(title):
            st.markdown("**질문**")
            st.write(evidence.question)

            st.markdown("**정답 또는 설명**")
            st.write(evidence.answer)

            st.caption(
                f"유형: {evidence.q_type} | "
                f"검색점수: {evidence.score:.6f}"
            )


def render_result_stage(config: dict[str, Any]) -> None:
    result = st.session_state.analysis_result
    mode_key = st.session_state.analysis_mode

    if result is None:
        st.error("분석 결과를 불러올 수 없습니다.")
        return

    # 이 key가 DOM의 .st-key-result_panel 클래스를 생성합니다.
    with st.container(key="result_panel"):
        st.markdown(
            f"""
            <div class="xd-card">
                <div class="xd-card-title">4단계 · 결과와 다음 단계</div>
                <div class="xd-muted">
                    실행 모드
                    <span class="xd-chip" style="margin-left:.25rem;">
                        {escaped(MODE_KEY_TO_LABEL.get(mode_key, mode_key))}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if isinstance(result, PipelineResult):
            render_pipeline_result(
                result,
                show_raw_json=config["show_raw_json"],
            )
        elif hasattr(result, "answer"):
            render_text_result(result, mode_key)
            if config["show_raw_json"] and hasattr(result, "model_dump"):
                with st.expander("원본 결과 JSON"):
                    st.json(result.model_dump())
        else:
            st.error(
                "지원하지 않는 결과 타입입니다: "
                f"{type(result).__name__}"
            )

        render_feedback()

        action_col1, action_col2 = st.columns(2)

        with action_col1:
            if st.button(
                "같은 사례로 다시 분석",
                use_container_width=True,
                key="result_rerun_same_case",
            ):
                st.session_state.stage = "input"
                st.session_state.analysis_result = None
                st.session_state.clarification = None
                st.rerun()

        with action_col2:
            if st.button(
                "새 사례 입력",
                type="primary",
                use_container_width=True,
                key="result_new_case",
            ):
                reset_analysis()
                st.session_state.case_text = ""
                st.session_state.example_selector = "직접 입력"
                st.rerun()


def render_feedback() -> None:
    st.divider()
    st.subheader("서비스 평가")
    st.caption(
        "라이브 데모 품질 개선을 위해 결과가 유용했는지 알려주세요."
    )

    like_col, dislike_col, spacer = st.columns([1, 1, 4])

    with like_col:
        if st.button(
            "유용해요 👍",
            use_container_width=True,
        ):
            st.session_state.feedback = "positive"

    with dislike_col:
        if st.button(
            "개선이 필요해요 👎",
            use_container_width=True,
        ):
            st.session_state.feedback = "negative"

    with spacer:
        if st.session_state.feedback == "positive":
            st.success("피드백이 기록되었습니다.")
        elif st.session_state.feedback == "negative":
            st.info("피드백이 기록되었습니다.")


# ============================================================
# 9. 메인
# ============================================================

def main() -> None:
    initialize_state()
    show_notice_once()

    config = render_sidebar()
    ui.inject_css()
    render_hero()
    render_stage_indicator()

    if st.session_state.stage == "input":
        render_input_stage(config)

    elif st.session_state.stage == "clarification":
        render_clarification_stage()

    elif st.session_state.stage == "result":
        render_result_stage(config)

    else:
        st.error(
            f"알 수 없는 화면 상태입니다: {st.session_state.stage}"
        )
        reset_analysis()

    st.markdown(
        f"""
        <div class="xd-footer">
            <div class="xd-footer-brand">
                MedCheck AI · XDoctors — eXplainable Doctors
            </div>
            <div>
                Evidence-grounded Clinical Decision Support ·
                2026 K-Data Science Hackathon
            </div>
            <div class="xd-footer-note">
                본 서비스는 연구·교육용 임상 의사결정 지원 프로토타입이며,
                확정 진단이나 처방을 대체하지 않습니다.
            </div>
            <div class="xd-footer-note">
                {escaped(AIHUB_SOURCE_NOTE)}
                <a href="{escaped(AIHUB_DATASET_URL)}"
                   target="_blank" rel="noopener noreferrer">데이터셋 바로가기</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
