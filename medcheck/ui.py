"""
결과 화면 렌더링 모듈

app.py의 raw JSON 출력과 기본 metric을, 의료진이 빠르게 스캔할 수 있는
형태로 다듬습니다. 기존 --xd-* 토큰을 그대로 쓰되 다음을 바꿉니다.

1. 구조화된 환자 정보 — raw JSON → 한국어 라벨 표
2. metric 4개 — 값만 큰 카드 → 응급도 색상, 상태 판독 가능
3. 감별진단 — 좌우 컬럼 → 순위·근거 밀도가 보이는 카드
4. 빈 상태 — 파란 info 박스 남발 → 조용한 회색 문구

설계 원칙:
  - 응급도 색은 트리아지 관례를 따릅니다 (즉시=적, 긴급=황, 주의=청, 일반=회)
  - 빈 상태는 강조하지 않습니다. 정보 부재를 파란 박스로 알리면
    판단 보류 화면이 오히려 '시스템이 실패했다'처럼 읽힙니다.
"""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st

from .action_card import (
    FIELD_LABELS,
    _is_empty,
    _stringify,
    canonical_field,
)


# ---------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------

_UI_CSS = """
<style>
/* ---- 지표 카드 ---- */
.xd-metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .7rem;
    margin: .2rem 0 1.1rem;
}
.xd-metric {
    background: var(--xd-surface);
    border: 1px solid var(--xd-border);
    border-radius: var(--xd-radius-sm);
    padding: .8rem .95rem .8rem 1.05rem;
    position: relative;
    overflow: hidden;
}
/* 상태는 카드를 채우지 않고 좌측 상태선으로만 표현합니다. */
.xd-metric::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--xd-border);
}
.xd-metric-now::before   { background: var(--xd-danger); }
.xd-metric-soon::before  { background: var(--xd-warn); }
.xd-metric-watch::before { background: var(--xd-blue); }
.xd-metric-ok::before    { background: var(--xd-success); }

.xd-metric-label {
    color: var(--xd-muted);
    font-size: .82rem;
    font-weight: 600;
    margin-bottom: .18rem;
}
.xd-metric-value {
    color: var(--xd-navy);
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.15;
}
.xd-metric-now   .xd-metric-value { color: var(--xd-danger); }
.xd-metric-soon  .xd-metric-value { color: var(--xd-warn-text); }
.xd-metric-ok    .xd-metric-value { color: var(--xd-success); }
.xd-metric-note {
    color: var(--xd-muted);
    font-size: .78rem;
    margin-top: .2rem;
}

/* ---- 환자 정보 표 ---- */
.xd-facts {
    display: grid;
    grid-template-columns: 7rem 1fr;
    gap: .1rem .9rem;
    font-size: 1rem;
    line-height: 1.6;
}
.xd-facts-key {
    color: var(--xd-muted);
    font-weight: 600;
    padding: .34rem 0;
    border-bottom: 1px solid var(--xd-border-soft);
}
.xd-facts-val {
    color: var(--xd-navy);
    padding: .34rem 0;
    border-bottom: 1px solid var(--xd-border-soft);
}
.xd-facts-key:last-of-type,
.xd-facts-val:last-of-type { border-bottom: none; }

/* ---- 감별진단 카드 ---- */
.xd-dx {
    background: var(--xd-surface);
    border: 1px solid var(--xd-border);
    border-radius: var(--xd-radius-sm);
    padding: 1rem 1.1rem;
    margin-bottom: .7rem;
}
.xd-dx-head {
    display: flex;
    align-items: baseline;
    gap: .55rem;
    margin-bottom: .7rem;
    padding-bottom: .55rem;
    border-bottom: 1px solid var(--xd-border-soft);
}
.xd-dx-rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.5rem;
    height: 1.5rem;
    border-radius: 50%;
    background: var(--xd-navy);
    color: #fff;
    font-size: .8rem;
    font-weight: 800;
    flex: none;
}
.xd-dx-rank-2 { background: #4E6E8A; }
.xd-dx-rank-3 { background: #8598A8; }
.xd-dx-name {
    color: var(--xd-navy);
    font-weight: 800;
    font-size: 1.04rem;
    line-height: 1.45;
    flex: 1;
}
.xd-dx-body {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: .9rem;
}
.xd-dx-col-title {
    font-size: .84rem;
    font-weight: 700;
    margin-bottom: .35rem;
}
.xd-dx-support  .xd-dx-col-title { color: var(--xd-success); }
.xd-dx-counter  .xd-dx-col-title { color: var(--xd-warn-text); }
.xd-dx-list {
    margin: 0;
    padding-left: 1.05rem;
    font-size: .98rem;
    line-height: 1.65;
    color: var(--xd-text);
}
.xd-dx-list li { margin-bottom: .18rem; }
.xd-dx-foot {
    margin-top: .75rem;
    padding-top: .6rem;
    border-top: 1px solid var(--xd-border-soft);
    display: flex;
    align-items: center;
    gap: .5rem;
    flex-wrap: wrap;
}
.xd-dx-foot-label {
    color: var(--xd-muted);
    font-size: .84rem;
    font-weight: 600;
}
/* 추가 확인사항 제목 — 보라 대신 중립 네이비 */
.xd-dx-checks .xd-dx-col-title { color: var(--xd-navy); }

/* ---- 조용한 빈 상태 ---- */
.xd-empty {
    color: #8B97A3;
    font-size: .9rem;
    line-height: 1.6;
    padding: .55rem .1rem;
    font-style: normal;
}
.xd-section-title {
    color: var(--xd-navy);
    font-weight: 700;
    font-size: .97rem;
    margin: 1rem 0 .5rem;
}

/* ---- 반응형 ---- */
@media (max-width: 1100px) {
    .xd-metrics { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 768px) {
    .xd-metrics { grid-template-columns: repeat(2, 1fr); gap: .5rem; }
    .xd-metric-value { font-size: 1.35rem; }
    .xd-dx-body { grid-template-columns: 1fr; gap: .7rem; }
    .xd-facts { grid-template-columns: 5.5rem 1fr; font-size: .92rem; }
}
</style>
"""


def inject_css() -> None:
    """페이지당 한 번만 호출하면 됩니다."""
    st.markdown(_UI_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------
# 1. 구조화된 환자 정보
# ---------------------------------------------------------------

def render_structured_case(structured_case: Any) -> None:
    """
    raw JSON 대신 한국어 라벨 표로 렌더링합니다.

    입력 예: {"age": "68세", "pain_radiation": "우측", "vital_signs": {}}
    출력 예: 나이 | 68세
             방사 여부 | 우측
             (vital_signs는 빈 값이라 표시하지 않음)
    """
    rows = _to_fact_rows(structured_case)

    if not rows:
        st.markdown(
            '<div class="xd-empty">구조화된 정보가 없습니다.</div>',
            unsafe_allow_html=True,
        )
        return

    cells = "".join(
        f'<div class="xd-facts-key">{escape(label)}</div>'
        f'<div class="xd-facts-val">{escape(value)}</div>'
        for label, value in rows
    )
    st.markdown(f'<div class="xd-facts">{cells}</div>', unsafe_allow_html=True)


def _to_fact_rows(structured_case: Any) -> list[tuple[str, str]]:
    if not structured_case:
        return []

    if hasattr(structured_case, "model_dump"):
        structured_case = structured_case.model_dump()
    if not isinstance(structured_case, dict):
        return []

    # 모델이 쓴 변형 키를 표준명으로 모읍니다 (pain_radiation → radiation 등)
    normalized: dict[str, Any] = {}
    for key, value in structured_case.items():
        canon = canonical_field(key)
        if canon not in normalized or _is_empty(normalized[canon]):
            normalized[canon] = value

    rows: list[tuple[str, str]] = []
    for key, label in FIELD_LABELS.items():
        value = normalized.get(key)
        if not _is_empty(value):
            rows.append((label, _stringify(value)))

    # 스키마에 없는 키도 뒤에 붙입니다 (모델이 임의로 만든 필드)
    for key, value in normalized.items():
        if key not in FIELD_LABELS and not _is_empty(value):
            rows.append((key, _stringify(value)))

    return rows


# ---------------------------------------------------------------
# 2. 지표 카드
# ---------------------------------------------------------------

_URGENCY_TONE = {
    "즉시": "xd-metric-now",
    "긴급": "xd-metric-soon",
    "주의": "xd-metric-watch",
    "일반": "",
}


def render_metrics(
    *,
    urgency: str,
    n_diagnoses: int,
    n_evidence: int,
    verification: Any | None,
    has_red_flags: bool = True,
) -> None:
    """응급도에 트리아지 색을 입힌 지표 4개를 그립니다."""
    if verification is None:
        verify_value, verify_tone, verify_note = "미실행", "", "검증기 미사용 모드"
    elif verification.overall_grounded and not verification.needs_abstention:
        verify_value, verify_tone, verify_note = "통과", "xd-metric-ok", "근거 일치 확인"
    else:
        verify_value, verify_tone, verify_note = "주의", "xd-metric-soon", "근거 불일치 발견"

    cards = [
        (
            "응급도",
            urgency,
            _URGENCY_TONE.get(urgency, ""),
            
            "위험 신호 기준" if has_red_flags and urgency != "일반" 
            else "조치 긴급도 기준" if urgency != "일반"
            else "위험 신호 없음",
        ),
        (
            "감별진단",
            str(n_diagnoses),
            "" if n_diagnoses else "xd-metric-soon",
            "근거와 연결된 후보" if n_diagnoses else "생성되지 않음",
        ),
        (
            "검색 근거",
            str(n_evidence),
            "",
            "의학지식 문서",
        ),
        ("검증 결과", verify_value, verify_tone, verify_note),
    ]

    html_parts = ['<div class="xd-metrics">']
    for label, value, tone, note in cards:
        html_parts.append(
            f'<div class="xd-metric {tone}">'
            f'<div class="xd-metric-label">{escape(label)}</div>'
            f'<div class="xd-metric-value">{escape(value)}</div>'
            f'<div class="xd-metric-note">{escape(note)}</div>'
            "</div>"
        )
    html_parts.append("</div>")

    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ---------------------------------------------------------------
# 3. 감별진단 카드
# ---------------------------------------------------------------

def render_differential_diagnoses(diagnoses: list[Any]) -> None:
    """
    순위 배지와 근거를 한 카드에 담아, 펼치지 않아도 내용이 보이게 합니다.

    기존 expander는 접혀 있어 2·3순위를 놓치기 쉬웠습니다.
    """
    if not diagnoses:
        st.markdown(
            '<div class="xd-empty">'
            "이 사례에서는 근거와 연결된 감별진단을 생성하지 않았습니다."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for dx in sorted(diagnoses, key=lambda d: d.rank):
        rank_cls = f"xd-dx-rank-{dx.rank}" if dx.rank in (2, 3) else ""

        support = _bullet_list(dx.supporting_evidence)
        counter = _bullet_list(dx.contradicting_or_missing_evidence)
        checks = _bullet_list(dx.recommended_checks)

        badges = (
            "".join(
                f'<span class="xd-evidence-id">{escape(e)}</span>'
                for e in dx.evidence_ids
            )
            if dx.evidence_ids
            else '<span class="xd-empty">연결된 문서 근거 없음</span>'
        )

        parts = [
            '<div class="xd-dx">',
            '<div class="xd-dx-head">',
            f'<span class="xd-dx-rank {rank_cls}">{dx.rank}</span>',
            f'<span class="xd-dx-name">{escape(dx.name)}</span>',
            "</div>",
            '<div class="xd-dx-body">',
            '<div class="xd-dx-support">',
            '<div class="xd-dx-col-title">지지 근거</div>',
            support,
            "</div>",
            '<div class="xd-dx-counter">',
            '<div class="xd-dx-col-title">반대 또는 누락 근거</div>',
            counter,
            "</div>",
            "</div>",
        ]

        if dx.recommended_checks:
            parts.extend([
                '<div class="xd-dx-body xd-dx-checks" '
                'style="grid-template-columns:1fr;margin-top:.7rem;">',
                "<div>",
                '<div class="xd-dx-col-title">추가 확인사항</div>',
                checks,
                "</div>",
                "</div>",
            ])

        parts.extend([
            '<div class="xd-dx-foot">',
            '<span class="xd-dx-foot-label">연결 문서</span>',
            badges,
            "</div>",
            "</div>",
        ])

        st.markdown("".join(parts), unsafe_allow_html=True)


def _bullet_list(items: list[str] | None) -> str:
    if not items:
        return '<div class="xd-empty">해당 없음</div>'
    lis = "".join(f"<li>{escape(str(item))}</li>" for item in items)
    return f'<ul class="xd-dx-list">{lis}</ul>'


# ---------------------------------------------------------------
# 4. 조용한 빈 상태 / 섹션 제목
# ---------------------------------------------------------------

def section_title(text: str) -> None:
    st.markdown(
        f'<div class="xd-section-title">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def empty_note(text: str) -> None:
    """
    st.info 대신 씁니다.

    파란 박스는 시선을 끌어서, 판단 보류 화면에서 정보 부재만
    여러 개 강조되면 시스템이 실패한 것처럼 읽힙니다.
    """
    st.markdown(
        f'<div class="xd-empty">{escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_key_value_list(
    items: list[Any],
    *,
    title_attr: str,
    note_attr: str,
    tag_attr: str | None = None,
    empty_text: str = "해당 항목이 없습니다.",
) -> None:
    """
    red_flags / missing_information / next_steps를 공통 형식으로 그립니다.
    dataframe보다 한국어 가독성이 낫고 열 잘림이 없습니다.
    """
    if not items:
        empty_note(empty_text)
        return

    tone = {
        "즉시": "xd-pill-now", "긴급": "xd-pill-now",
        "필수": "xd-pill-now", "주의": "xd-pill-soon",
        "중요": "xd-pill-soon", "일반": "xd-pill-watch",
        "보조": "xd-pill-watch", "선택": "xd-pill-watch",
    }

    parts = ['<div class="xd-dx" style="padding:.8rem 1rem;">']
    for i, item in enumerate(items):
        tag = str(getattr(item, tag_attr, "")) if tag_attr else ""
        pill = (
            f'<span class="xd-pill {tone.get(tag, "xd-pill-watch")}">'
            f"{escape(tag)}</span>"
            if tag
            else ""
        )
        border = (
            ""
            if i == 0
            else "border-top:1px solid var(--xd-border-soft);"
        )
        parts.append(
            f'<div style="padding:.5rem 0;{border}">'
            f'{pill}<b>{escape(str(getattr(item, title_attr, "")))}</b>'
            f'<div style="color:var(--xd-muted);font-size:.88rem;'
            f'line-height:1.6;margin-top:.15rem;">'
            f'{escape(str(getattr(item, note_attr, "")))}</div>'
            "</div>"
        )
    parts.append("</div>")

    st.markdown("".join(parts), unsafe_allow_html=True)