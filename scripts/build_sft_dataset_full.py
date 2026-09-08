# %% [markdown]
# # SFT 학습 데이터 변환 — full 버전 (질문 개수 자르기 없음)
#
# 입력:  results/generated_assessments.jsonl  (2단계 산출물)
# 출력:  results/sft_full/train.jsonl, eval.jsonl
#
# 프롬프트는 medcheck.prompts에서 import하지 않고 이 파일 안에 직접 정의합니다.
# (학습 시점의 프롬프트를 파일에 박아두어, 나중에 prompts.py가 바뀌어도
#  이 데이터가 어떤 프롬프트로 만들어졌는지 추적할 수 있게 하기 위함)

# %%
import sys, os, json, random
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/home1/zye614/projects/medcheck")
os.chdir("/home1/zye614/projects/medcheck")

SRC_PATH = Path("results/generated_assessments_clean.jsonl")
OUT_DIR = Path("results/sft_full_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EVAL_RATIO = 0.1

print("작업 디렉토리:", os.getcwd())

# %% [markdown]
# ## 1. 2단계 산출물 로드

# %%
rows = [
    json.loads(line)
    for line in SRC_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
ok_rows = [r for r in rows if r.get("schema_ok")]

print(f"전체 {len(rows)}건 / 스키마 통과 {len(ok_rows)}건")
print(f"진료과 분포: {dict(Counter(r['domain'] for r in ok_rows))}")
print(f"질문 수 분포: {dict(Counter(r['n_questions'] for r in ok_rows))}")

# %% [markdown]
# ## 2. 프롬프트 정의 (모듈 import 없이 여기서 직접 정의)
#
# 중요: 학습에 쓴 프롬프트와 추론 시 프롬프트가 **반드시 동일**해야 합니다.
# 아래 두 값을 나중에 pipeline에서 쓸 때도 그대로 사용하세요.

# %%
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
8. 서술은 한국어로 작성한다. 중국어, 일본어를 절대 사용하지 않는다. 의학 용어, 약물명, 검사명은 한국어를 우선하되 필요하면 괄호 안에 영어 언어만 병기한다. 
예 : 급성관상동맥증후군(ACS), 아스피린(aspirin), 심전도(electrocardiogram, ECG)
"""



def build_clarification_prompt(patient_case: str) -> str:
    """
    학습/추론에 공통으로 쓰는 사용자 프롬프트.

    입력 예:  "남성이 체중 감소와 목의 덩어리를 호소하며 병원을 방문하였다."
    출력 예:  "[환자 사례]\n남성이 체중 감소와...\n\n[과제]\n환자 사례에서 ..."
    """
    from medcheck.models import ClarificationAssessment  # 스키마 생성용

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
아래 두 예시의 판단 기준을 그대로 적용하라.

[질문 생성 규칙]
- 나이, 성별, 주호소, 주요 증상이 확인되고
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


# 프롬프트가 제대로 만들어지는지 확인
_preview = build_clarification_prompt("테스트 사례")
print("예시 포함:", "[예시 1]" in _preview)
print("프롬프트 길이:", len(_preview), "자")

# %% [markdown]
# ## 3. messages 형식으로 변환
#
# assistant 응답은 compact JSON(공백 최소)으로 넣습니다.
# 시퀀스 길이를 줄이고, 모델이 일관된 포맷을 배우게 하기 위함입니다.

# %%
def to_messages(row: dict) -> dict:
    assistant_json = json.dumps(
        row["output"],
        ensure_ascii=False,
        separators=(",", ":"),   # compact
    )
    return {
        "qa_id": row["qa_id"],
        "domain": row["domain"],
        "messages": [
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": build_clarification_prompt(row["damaged"]).strip()},
            {"role": "assistant", "content": assistant_json},
        ],
    }


samples = [to_messages(r) for r in ok_rows]
print(f"변환 완료: {len(samples)}건")

# 첫 샘플 확인
_s = samples[0]
print("\n--- system (앞 200자) ---")
print(_s["messages"][0]["content"][:200])
print("\n--- user (앞 300자) ---")
print(_s["messages"][1]["content"][:300])
print("\n--- assistant (앞 400자) ---")
print(_s["messages"][2]["content"][:400])

# %% [markdown]
# ## 4. train / eval 분할
#
# 진료과 비율을 유지하도록 층화 분할합니다.

# %%
random.seed(SEED)

by_domain = {}
for s in samples:
    by_domain.setdefault(s["domain"], []).append(s)

train, evalset = [], []
for domain, items in by_domain.items():
    random.shuffle(items)
    n_eval = max(1, int(len(items) * EVAL_RATIO))
    evalset.extend(items[:n_eval])
    train.extend(items[n_eval:])

random.shuffle(train)
random.shuffle(evalset)

print(f"train {len(train)}건 / eval {len(evalset)}건")
print("\ntrain 진료과:", dict(Counter(s['domain'] for s in train)))
print("eval  진료과:", dict(Counter(s['domain'] for s in evalset)))

# qa_id 누수 확인
overlap = {s["qa_id"] for s in train} & {s["qa_id"] for s in evalset}
print(f"\nqa_id 중복(누수): {len(overlap)}건  ← 0이어야 정상")

# %% [markdown]
# ## 5. 저장

# %%
def save_jsonl(path: Path, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"저장: {path}  ({len(items)}건, {path.stat().st_size/1024/1024:.1f}MB)")


save_jsonl(OUT_DIR / "train.jsonl", train)
save_jsonl(OUT_DIR / "eval.jsonl", evalset)

# %% [markdown]
# ## 6. 통계 확인

# %%
q_counts = Counter(len(json.loads(s["messages"][2]["content"])["questions"]) for s in samples)
rc_counts = Counter(json.loads(s["messages"][2]["content"])["requires_clarification"] for s in samples)

print(f"질문 수 분포          : {dict(sorted(q_counts.items()))}")
print(f"requires_clarification: {dict(rc_counts)}")

# 토큰 길이 대략 확인 (학습 시 max_seq_length 결정에 필요)
lens = [
    len(s["messages"][0]["content"]) + len(s["messages"][1]["content"]) + len(s["messages"][2]["content"])
    for s in samples
]
lens.sort()
print(f"\n문자 길이 — 중앙값 {lens[len(lens)//2]:,} / 최대 {lens[-1]:,}")
print(f"  (한국어는 대략 문자수 ≈ 토큰수의 1.5~2배로 잡으면 안전)")

# %%
