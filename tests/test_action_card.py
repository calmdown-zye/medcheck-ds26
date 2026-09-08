"""streamlit 없이 action_card의 추출 로직만 검증합니다."""
import sys, types

# streamlit 스텁 (렌더링은 테스트하지 않고 추출 로직만 봄)
st_stub = types.ModuleType("streamlit")
st_stub.markdown = lambda *a, **k: None
st_stub.code = lambda *a, **k: None
class _Exp:
    def __enter__(self): return self
    def __exit__(self, *a): return False
st_stub.expander = lambda *a, **k: _Exp()
sys.modules["streamlit"] = st_stub

# 프로젝트 루트에서 실행하면 로컬 medcheck 패키지를 그대로 사용합니다.
# 특정 개발자의 서버 절대경로에 의존하지 않도록 별도 경로를 추가하지 않습니다.
from medcheck.action_card import (
    should_show_action_card, extract_reasons, extract_preparations,
    build_symptom_summary, format_handoff_text, infer_urgency,
)


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# --- 실제 파이프라인이 만드는 형태의 모의 객체 ---
decision_urgent = Obj(
    abstain=False,
    abstain_reason="",
    red_flags=[
        Obj(finding="압박성 흉통 + 식은땀", reason="급성관상동맥증후군 시사",
            urgency="즉시", evidence_ids=["E1"]),
        Obj(finding="수축기 혈압 90mmHg", reason="쇼크 초기 가능성",
            urgency="긴급", evidence_ids=[]),
        Obj(finding="흡연력", reason="위험인자", urgency="주의", evidence_ids=["E2"]),
    ],
    missing_information=[
        Obj(question="심전도 결과가 있습니까?", why_needed="ST 분절 상승 확인", priority="필수"),
        Obj(question="흉통이 언제 시작되었습니까?", why_needed="재관류 시간 결정", priority="필수"),
        Obj(question="과거 심질환 병력이 있습니까?", why_needed="위험도 평가", priority="중요"),
    ],
    structured_case={
        "age": "67", "sex": "남성", "chief_complaint": "흉통",
        "symptoms": ["압박성 흉통", "식은땀"],
        "onset": "30분 전", "duration": "지속",
        "pain_character": "미상",          # 빈 값 표기 1
        "radiation": "",                   # 빈 값 표기 2
        "vital_signs": {},                 # 빈 값 표기 3
        "medical_history": None,           # 빈 값 표기 4
        "medications": [],                 # 빈 값 표기 5
    },
)

decision_abstain = Obj(
    abstain=True,
    abstain_reason="검색된 의료지식과 직접 연결된 감별진단이 없습니다.",
    red_flags=[],
    missing_information=[],
    structured_case={"age": "50대", "sex": "남성", "chief_complaint": "흉통"},
)

decision_normal = Obj(
    abstain=False, abstain_reason="", red_flags=[],
    missing_information=[], structured_case={"age": "30", "sex": "여성"},
)

assessment_urgent = Obj(
    urgent=True,
    urgent_message="흉통은 즉시 대면 평가가 필요할 수 있습니다.",
    questions=[
        Obj(question="통증이 언제 시작되었나요?", why_needed="응급도 판단", priority="필수"),
        Obj(question="식은땀이 동반되나요?", why_needed="ACS 시사 소견", priority="중요"),
    ],
    structured_case={"age": "50대", "sex": "남성", "chief_complaint": "흉통"},
)

print("=" * 70)
print("1. 트리거 판정")
print("=" * 70)
cases = [
    ("고위험 red_flags(즉시)", dict(decision=decision_urgent), True),
    ("abstain=True", dict(decision=decision_abstain), True),
    ("평범한 결과", dict(decision=decision_normal), False),
    ("assessment.urgent=True", dict(assessment=assessment_urgent), True),
]
for name, kwargs, expected in cases:
    got = should_show_action_card(**kwargs)
    mark = "✅" if got == expected else "❌"
    print(f"  {mark} {name}: {got} (기대 {expected})")

print("\n" + "=" * 70)
print("2. ① 진료가 필요한 이유 — urgency 순 정렬 확인")
print("=" * 70)
for r in extract_reasons(decision_urgent):
    print(f"  [{r['urgency']}] {r['finding']} — {r['reason']} {r['evidence_ids']}")

print("\n  (assessment 경로: urgent_message가 맨 앞에 삽입되는지)")
for r in extract_reasons(None, assessment_urgent):
    print(f"  [{r['urgency']}] {r['finding']} — {r['reason']}")

print("\n" + "=" * 70)
print("3. ② 방문 시 준비할 정보 — 우선순위 정렬")
print("=" * 70)
for p in extract_preparations(decision_urgent):
    print(f"  [{p['priority']}] {p['question']}")

print("\n" + "=" * 70)
print("4. ③ 증상 요약 — 빈 값 5종이 제외되는지")
print("=" * 70)
rows = build_symptom_summary(decision_urgent.structured_case)
for label, value in rows:
    print(f"  {label}: {value}")
print(f"\n  → 11개 필드 중 {len(rows)}개만 표시 (빈 값 5종 제외 확인)")

print("\n" + "=" * 70)
print("5. 복사용 평문")
print("=" * 70)
print(format_handoff_text(
    rows,
    extract_reasons(decision_urgent),
    extract_preparations(decision_urgent),
))
