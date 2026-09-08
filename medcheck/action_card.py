"""
행동 카드(Action Card) 렌더링 모듈

심사 피드백 대응:
  "판단 보류나 응급 경고가 안내 문구를 제시하는 데서 끝나면 사용자가 실제
   다음 행동을 결정하기 어려울 수 있음. 고위험 사례에서는 일반 분석보다
   경고와 행동 안내를 우선 노출하고, 진료가 필요한 이유, 방문 시 준비할
   정보, 의료진에게 바로 보여줄 수 있는 증상 요약을 한 화면에 제공하여
   경고 이후의 행동까지 자연스럽게 연결할 필요가 있음."

설계 원칙:
  - 새로운 LLM 호출을 추가하지 않는다.
    이미 생성되는 red_flags / missing_information / structured_case를
    재조합하기만 한다.
  - 추가 질문 단계(ClarificationAssessment)와 결과 단계(ClinicalDecision)의
    두 경고 경로를 하나의 진입점으로 통합한다.
  - 기존 CSS 토큰(--xd-red, --xd-amber 등)을 그대로 쓴다.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


# ---------------------------------------------------------------
# 1. 트리거 판정
# ---------------------------------------------------------------

URGENCY_RANK = {"즉시": 3, "긴급": 2, "주의": 1, "일반": 0}


def infer_urgency(decision: Any) -> str:
    """red_flags의 urgency 중 가장 높은 단계를 반환합니다."""
    levels = {
        str(getattr(item, "urgency", "일반"))
        for item in getattr(decision, "red_flags", [])
    }
    # red_flags가 비어도 next_steps의 긴급도를 반영합니다.
    # 3B 모델이 red_flags를 빠뜨리는 경우가 있어, 조치의 긴급도로 보완합니다.
    levels |= {
        str(getattr(item, "urgency", "일반"))
        for item in getattr(decision, "next_steps", [])
    }
    if "즉시" in levels: return "즉시"
    if "긴급" in levels: return "긴급"
    if levels - {"일반", "선택"}: return "주의"
    return "일반"


def should_show_action_card(
    decision: Any = None,
    assessment: Any = None,
) -> bool:
    """
    행동 카드를 우선 노출해야 하는 상황인지 판정합니다.

    입력 예: decision.abstain=True            → True
            decision.red_flags=[urgency"즉시"] → True
            assessment.urgent=True            → True
            둘 다 없거나 평범한 경우          → False
    """
    if assessment is not None and getattr(assessment, "urgent", False):
        return True

    if decision is not None:
        if getattr(decision, "abstain", False):
            return True
        if URGENCY_RANK.get(infer_urgency(decision), 0) >= 2:  # 즉시 또는 긴급
            return True

    return False


# ---------------------------------------------------------------
# 2. 3요소 추출 — 새 LLM 호출 없이 기존 필드를 재조합
# ---------------------------------------------------------------

# structured_case의 영문 키를 화면에 보여줄 한국어 라벨로 변환
FIELD_LABELS = {
    "age": "나이",
    "sex": "성별",
    "chief_complaint": "주호소",
    "symptoms": "증상",
    "onset": "발생 시점",
    "duration": "지속 기간",
    "pain_character": "통증 양상",
    "radiation": "방사 여부",
    "vital_signs": "활력징후",
    "medical_history": "과거력",
    "medications": "복용약",
}

# 모델이 스키마와 다른 키 이름을 쓰는 경우가 있어 정규화합니다.
# 예: structured_case에 "pain_radiation": "모름" 이 나온 사례
FIELD_ALIASES = {
    "pain_radiation": "radiation",
    "radiation_pattern": "radiation",
    "pain_quality": "pain_character",
    "pain_characteristics": "pain_character",
    "characteristics": "pain_character",
    "current_medication": "medications",
    "current_medications": "medications",
    "medication": "medications",
    "past_history": "medical_history",
    "history": "medical_history",
    "gender": "sex",
    "vitals": "vital_signs",
    "vital_sign": "vital_signs",
    "symptom": "symptoms",
    "onset_time": "onset",
}


def canonical_field(key: str) -> str:
    """모델이 쓴 키를 표준 필드명으로 정규화합니다."""
    return FIELD_ALIASES.get(key, key)

# "정보 없음"을 나타내는 여러 표기 (모델마다 제각각이라 전부 처리)
EMPTY_VALUES = {"", "미상", "정보 없음", "없음", "N/A", "불명", "미지수", None}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in EMPTY_VALUES
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value if not _is_empty(v))
    if isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items() if not _is_empty(v))
    return str(value)


def extract_reasons(decision: Any, assessment: Any = None) -> list[dict[str, str]]:
    """① 진료가 필요한 이유 — red_flags + abstain_reason + urgent_message"""
    reasons: list[dict[str, str]] = []

    for flag in getattr(decision, "red_flags", []) or []:
        reasons.append({
            "finding": str(getattr(flag, "finding", "")),
            "reason": str(getattr(flag, "reason", "")),
            "urgency": str(getattr(flag, "urgency", "일반")),
            "evidence_ids": list(getattr(flag, "evidence_ids", []) or []),
        })

    # urgency 높은 순으로 정렬
    reasons.sort(key=lambda r: -URGENCY_RANK.get(r["urgency"], 0))

    if assessment is not None and getattr(assessment, "urgent", False):
        message = getattr(assessment, "urgent_message", "")
        if message:
            reasons.insert(0, {
                "finding": "응급 가능성 감지",
                "reason": str(message),
                "urgency": "즉시",
                "evidence_ids": [],
            })

    return reasons


def extract_preparations(decision: Any, assessment: Any = None) -> list[dict[str, str]]:
    """
    ② 방문 시 준비할 정보

    결과 단계에서는 missing_information을, 추가 질문 단계에서는
    아직 답하지 않은 questions를 체크리스트로 씁니다.
    이미 우선순위(필수/중요/보조)가 붙어 있어 그대로 정렬만 하면 됩니다.
    """
    priority_rank = {"필수": 0, "중요": 1, "보조": 2}
    items: list[dict[str, str]] = []

    for info in getattr(decision, "missing_information", []) or []:
        items.append({
            "question": str(getattr(info, "question", "")),
            "why_needed": str(getattr(info, "why_needed", "")),
            "priority": str(getattr(info, "priority", "보조")),
        })

    if assessment is not None:
        for q in getattr(assessment, "questions", []) or []:
            items.append({
                "question": str(getattr(q, "question", "")),
                "why_needed": str(getattr(q, "why_needed", "")),
                "priority": str(getattr(q, "priority", "보조")),
            })

    # 중복 제거 (질문 텍스트 기준)
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    for item in items:
        if item["question"] and item["question"] not in seen:
            seen.add(item["question"])
            unique.append(item)

    unique.sort(key=lambda i: priority_rank.get(i["priority"], 3))
    return unique


def build_symptom_summary(structured_case: Any) -> list[tuple[str, str]]:
    """
    ③ 의료진에게 보여줄 증상 요약

    raw JSON이 아니라 읽을 수 있는 (라벨, 값) 목록으로 변환합니다.
    비어 있는 필드는 제외하고, 모델이 쓴 변형 키는 표준명으로 정규화합니다.

    입력 예: {"age": "60세", "pain_radiation": "모름", "vital_signs": {}}
    출력 예: [("나이", "60세")]
             (pain_radiation → radiation으로 정규화되나 값이 "모름"이라 제외,
              vital_signs는 빈 dict라 제외)
    """
    if not structured_case:
        return []

    if hasattr(structured_case, "model_dump"):
        structured_case = structured_case.model_dump()
    if not isinstance(structured_case, dict):
        return []

    # 변형 키를 표준명으로 정규화 (먼저 온 값 우선)
    normalized: dict[str, Any] = {}
    for key, value in structured_case.items():
        canon = canonical_field(key)
        if canon not in normalized or _is_empty(normalized[canon]):
            normalized[canon] = value

    rows: list[tuple[str, str]] = []
    for key, label in FIELD_LABELS.items():
        value = normalized.get(key)
        if _is_empty(value):
            continue
        rows.append((label, _stringify(value)))

    # 스키마에 없는 추가 필드도 뒤에 붙임
    for key, value in normalized.items():
        if key in FIELD_LABELS or _is_empty(value):
            continue
        rows.append((key, _stringify(value)))

    return rows


def format_handoff_text(
    symptom_rows: list[tuple[str, str]],
    reasons: list[dict[str, str]],
    preparations: list[dict[str, str]],
) -> str:
    """의료진에게 그대로 보여주거나 복사할 수 있는 평문 요약"""
    lines = ["[환자 상태 요약]"]
    lines.extend(f"- {label}: {value}" for label, value in symptom_rows)

    if reasons:
        lines.append("")
        lines.append("[확인된 위험 신호]")
        for r in reasons:
            finding = r["finding"] or r["reason"]
            lines.append(f"- ({r['urgency']}) {finding}")

    if preparations:
        lines.append("")
        lines.append("[미확인 정보]")
        for p in preparations:
            lines.append(f"- [{p['priority']}] {p['question']}")

    lines.append("")
    lines.append("※ 본 요약은 임상 의사결정 지원 도구가 생성한 참고 자료이며,")
    lines.append("   확정 진단이나 처방이 아닙니다.")
    return "\n".join(lines)


# ---------------------------------------------------------------
# 3. 렌더링
# ---------------------------------------------------------------

_CARD_CSS = """
<style>
/* 행동 카드 — app.py의 --xd-* 토큰을 그대로 사용합니다.
   카드 전체를 강한 상태색으로 채우지 않고, 좌측 상태선과 배지로만
   위험도를 표현합니다. */
.xd-action {
    background: var(--xd-surface);
    border: 1px solid var(--xd-border);
    border-left: 4px solid var(--xd-danger);
    border-radius: var(--xd-radius);
    padding: 1.2rem 1.3rem 1.05rem;
    box-shadow: var(--xd-shadow);
    margin-bottom: 1rem;
}
.xd-action-head {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: .6rem;
    margin-bottom: .35rem;
}
.xd-action-title {
    color: var(--xd-danger);
    font-weight: 800;
    font-size: 1.14rem;
    line-height: 1.4;
    letter-spacing: -.01em;
}
.xd-action-badge {
    display: inline-block;
    padding: .16rem .62rem;
    border-radius: 999px;
    background: var(--xd-danger);
    color: #fff;
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .02em;
}
/* 판단 보류도 위험 계열로 표시합니다(보라 장식 제거). */
.xd-badge-hold {
    background: var(--xd-danger-soft);
    color: var(--xd-danger);
    border: 1px solid #F3C9CB;
}
.xd-action-sub {
    color: var(--xd-muted);
    font-size: 1rem;
    line-height: 1.65;
    margin-bottom: 1rem;
}
.xd-action-note {
    background: var(--xd-warn-soft);
    color: var(--xd-warn-text);
    border: 1px solid #F7D9BF;
    border-radius: var(--xd-radius-xs);
    padding: .55rem .75rem;
    font-size: .89rem;
    line-height: 1.6;
    margin: -.5rem 0 1rem;
}
.xd-action-sec {
    background: var(--xd-surface);
    border: 1px solid var(--xd-border);
    border-radius: var(--xd-radius-sm);
    padding: .9rem 1rem;
    margin-bottom: .65rem;
}
.xd-action-sec:last-child { margin-bottom: 0; }
.xd-action-sec-title {
    color: var(--xd-navy);
    font-weight: 700;
    font-size: .95rem;
    margin-bottom: .65rem;
    padding-bottom: .45rem;
    border-bottom: 1px solid var(--xd-border-soft);
}
.xd-action-item {
    padding: .5rem 0;
    font-size: 1rem;
    line-height: 1.65;
    color: var(--xd-text);
}
.xd-action-item + .xd-action-item {
    border-top: 1px solid var(--xd-border-soft);
}
.xd-action-why {
    color: var(--xd-muted);
    font-size: .87rem;
    line-height: 1.6;
    margin-top: .15rem;
    padding-left: 2.9rem;
}
.xd-pill {
    display: inline-block;
    min-width: 2.5rem;
    text-align: center;
    padding: .12rem .44rem;
    border-radius: 6px;
    font-size: .75rem;
    font-weight: 700;
    margin-right: .5rem;
    vertical-align: .06rem;
}
.xd-pill-now   { background: var(--xd-danger-soft); color: var(--xd-danger); }
.xd-pill-soon  { background: var(--xd-warn-soft); color: var(--xd-warn-text); }
.xd-pill-watch { background: var(--xd-blue-soft); color: var(--xd-blue-dark); }
.xd-kv {
    display: grid;
    grid-template-columns: 6rem 1fr;
    gap: .42rem .9rem;
    font-size: 1rem;
    line-height: 1.6;
}
.xd-kv-key {
    color: var(--xd-muted);
    font-weight: 600;
}
.xd-kv-val {
    color: var(--xd-navy);
}

@media (max-width: 768px) {
    .xd-action {
        padding: 1rem 1rem .9rem;
        border-radius: var(--xd-radius-sm);
    }
    .xd-action-title { font-size: 1.05rem; }
    .xd-action-why { padding-left: 0; }
    .xd-kv { grid-template-columns: 5rem 1fr; font-size: .92rem; }
}
</style>
"""

_PILL_CLASS = {
    "즉시": "xd-pill-now",
    "긴급": "xd-pill-now",
    "주의": "xd-pill-soon",
    "필수": "xd-pill-now",
    "중요": "xd-pill-soon",
    "보조": "xd-pill-watch",
}


def _pill(text: str) -> str:
    cls = _PILL_CLASS.get(text, "xd-pill-watch")
    return f'<span class="xd-pill {cls}">{escape(text)}</span>'


def render_action_card(
    decision: Any = None,
    assessment: Any = None,
    *,
    show_copy_box: bool = True,
) -> None:
    """
    행동 카드를 렌더링합니다.

    결과 단계:      render_action_card(decision=result.decision)
    추가 질문 단계: render_action_card(assessment=st.session_state.clarification)
    """
    st.markdown(_CARD_CSS, unsafe_allow_html=True)

    reasons = extract_reasons(decision, assessment)
    preparations = extract_preparations(decision, assessment)

    structured_case = None
    if decision is not None:
        structured_case = getattr(decision, "structured_case", None)
    if not structured_case and assessment is not None:
        structured_case = getattr(assessment, "structured_case", None)
    symptom_rows = build_symptom_summary(structured_case)

    urgency = infer_urgency(decision) if decision is not None else "즉시"
    if assessment is not None and getattr(assessment, "urgent", False):
        urgency = "즉시"

    is_abstained = bool(decision is not None and getattr(decision, "abstain", False))
    abstain_reason = (
        str(getattr(decision, "abstain_reason", "")) if is_abstained else ""
    )

    # 카드가 뜬 이유에 따라 제목·배지·설명을 다르게 씁니다.
    #   판단 보류  → "판단을 보류했습니다" + 보류 사유
    #   응급 경고  → "지금 필요한 조치" + 응급도 배지
    if is_abstained:
        title = "판단을 보류했습니다"
        badge = "보류"
        badge_cls = "xd-action-badge xd-badge-hold"
        subtitle = abstain_reason or "충분히 근거 있는 판단을 생성하지 못했습니다."
    else:
        title = "지금 필요한 조치"
        badge = urgency
        badge_cls = "xd-action-badge"
        if reasons :
            subtitle = "아래 위험 신호가 확인되었습니다. 의료진의 직접 평가가 필요합니다."
        else : 
            subtitle = "긴급도가 높은 조치가 필요합니다. 의료진의 직접 평가가 필요합니다."

    # --- HTML을 조각내지 않고 한 번에 조립합니다 ---
    # streamlit은 st.markdown 호출마다 HTML을 독립적으로 처리하므로,
    # 여러 번 나눠 부르면 열어둔 <div>가 자동으로 닫혀 레이아웃이 깨집니다.
    parts: list[str] = [
        '<div class="xd-action">',
        '<div class="xd-action-head">',
        f'<span class="xd-action-title">{escape(title)}</span>',
        f'<span class="{badge_cls}">{escape(badge)}</span>',
        "</div>",
        f'<div class="xd-action-sub">{escape(subtitle)}</div>',
    ]

    # 판단 보류인데 위험 신호도 있으면 응급도를 함께 알립니다.
    if is_abstained and urgency in {"즉시", "긴급"}:
        parts.append(
            f'<div class="xd-action-note">'
            f'다만 <b>{escape(urgency)}</b> 수준의 위험 신호가 확인되었습니다. '
            f'아래 내용을 확인하세요.</div>'
        )

    # --- ① 진료가 필요한 이유 ---
    if reasons:
        parts.append('<div class="xd-action-sec">')
        parts.append('<div class="xd-action-sec-title">① 진료가 필요한 이유</div>')
        for r in reasons:
            finding = escape(r["finding"] or r["reason"])
            why = escape(r["reason"]) if r["finding"] else ""
            ev = "".join(
                f'<span class="xd-evidence-id">{escape(e)}</span>'
                for e in r["evidence_ids"]
            )
            parts.append(
                f'<div class="xd-action-item">{_pill(r["urgency"])}'
                f"<b>{finding}</b> {ev}"
                + (f'<div class="xd-action-why">{why}</div>' if why else "")
                + "</div>"
            )
        parts.append("</div>")

    # --- ② 방문 시 준비할 정보 ---
    if preparations:
        parts.append('<div class="xd-action-sec">')
        parts.append('<div class="xd-action-sec-title">② 방문 시 준비할 정보</div>')
        for p in preparations:
            why = escape(p["why_needed"])
            parts.append(
                f'<div class="xd-action-item">{_pill(p["priority"])}'
                f'{escape(p["question"])}'
                + (f'<div class="xd-action-why">{why}</div>' if why else "")
                + "</div>"
            )
        parts.append("</div>")

    # --- ③ 의료진에게 보여줄 증상 요약 ---
    if symptom_rows:
        parts.append('<div class="xd-action-sec">')
        parts.append(
            '<div class="xd-action-sec-title">③ 의료진에게 보여줄 증상 요약</div>'
        )
        parts.append('<div class="xd-kv">')
        for label, value in symptom_rows:
            parts.append(f'<div class="xd-kv-key">{escape(label)}</div>')
            parts.append(f'<div class="xd-kv-val">{escape(value)}</div>')
        parts.append("</div>")
        parts.append("</div>")

    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)

    # --- 복사용 평문 ---
    if show_copy_box and symptom_rows:
        with st.expander("📋 의료진에게 보여줄 텍스트 (복사용)"):
            st.code(
                format_handoff_text(symptom_rows, reasons, preparations),
                language="text",
            )