# %% [markdown]
# # 질문 품질 평가 — removed 필드 적중률 (3-way)
#
# 지금까지의 평가는 **형식**만 봤습니다 (JSON 파싱, 스키마 통과, 필드 표기).
# 이 스크립트는 **내용**을 봅니다 — 훼손 때 지운 항목을 실제로 되물었는가.
#
# ## 채점 구조
#
# ```
# 원본 시나리오 (Validation split)
#   ↓ 1단계 훼손 (14B)
# damaged  +  removed = ["onset", "duration", "exam_results"]   ← 정답지
#   ↓ assess_clarification
#   ├─ base-3b   → questions
#   ├─ trimmed   → questions
#   └─ 14B       → questions   (상한선)
#   ↓ 판정 (14B): "이 질문들이 13개 필드 중 무엇을 묻는가"
# predicted_fields
#   ↓ 코드로 집합 비교
# precision / recall / F1
# ```
#
# 판정 시 `removed`(정답지)를 보여주지 않습니다. 판정이 정답에 오염되면
# 채점이 무의미해집니다.

# %%
import sys, os, json, gc, time, random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, "/home1/zye614/projects/medcheck")
os.chdir("/home1/zye614/projects/medcheck")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_3B = "/home1/zye614/projects/medcheck/sft_model/qwen2.5-3b-instruct"
ADAPTER_TRIMMED = "sft_model/adapters/qwen3b-clarify-trimmed-v2"
MODEL_14B = "/home1/zye614/projects/medcheck/sft_model/qwen2.5-14b-instruct"

DAMAGED_PATH = Path("results/val_damaged.jsonl")
GEN_PATH = Path("results/val_questions_3way_v2.jsonl")
JUDGE_PATH = Path("results/val_field_judgment_v2.jsonl")

for p in [DAMAGED_PATH, GEN_PATH, JUDGE_PATH]:
    p.parent.mkdir(parents=True, exist_ok=True)

print("작업 디렉토리:", os.getcwd())

# %% [markdown]
# ## 1. 평가 대상 문항 확정
#
# 기존 3-way 평가(형식 지표)와 **같은 문항**을 씁니다.
# 그래야 형식 지표와 내용 지표를 같은 표본에서 비교할 수 있습니다.

# %%
import medcheck.damage as dmg
from medcheck.data import load_dataset

prev_path = Path("results/eval_3way.jsonl")
target_ids: list[str] = []

if prev_path.exists():
    seen = set()
    for line in prev_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qa_id = json.loads(line)["qa_id"]
        if qa_id not in seen:
            seen.add(qa_id)
            target_ids.append(qa_id)
    print(f"기존 3-way 평가 문항 재사용: {len(target_ids)}건")

records, _ = load_dataset("data")
validation = {r.qa_id: r for r in records if r.split == "Validation"}

if target_ids:
    targets = [(validation[i], dmg.extract_scenario(validation[i].question))
               for i in target_ids if i in validation]
else:
    # 기존 파일이 없으면 새로 뽑음
    pool = []
    for r in validation.values():
        if not dmg.is_scenario_question(r.question):
            continue
        s = dmg.extract_scenario(r.question)
        if len(s) >= 60:          # 훼손할 여지가 있으려면 좀 길어야 함
            pool.append((r, s))
    random.seed(123)
    random.shuffle(pool)
    targets = pool[:84]

print(f"평가 대상: {len(targets)}건")
print(f"진료과: {dict(Counter(r.domain_name for r, _ in targets))}")

# %% [markdown]
# ## 2. 14B 로드

# %%
import torch
from medcheck.llm import LocalTransformersClient

os.environ["LOCAL_LLM_MODEL"] = MODEL_14B
os.environ["LOCAL_LLM_MAX_NEW_TOKENS"] = "1024"
os.environ.pop("LOCAL_LLM_ADAPTER", None)

llm14 = LocalTransformersClient()
print("14B 로드 완료:", llm14.model_path)

# %% [markdown]
# ## 3. Validation 훼손 — 정답지(removed) 생성
#
# Training 때와 **동일한 프롬프트·검증 규칙**을 씁니다.

# %%
done = set()
if DAMAGED_PATH.exists():
    for line in DAMAGED_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["qa_id"])

remaining = [(r, s) for r, s in targets if r.qa_id not in done]
print(f"이미 처리됨 {len(done)}건 / 남은 작업 {len(remaining)}건")

stats = Counter()
started = time.perf_counter()

with DAMAGED_PATH.open("a", encoding="utf-8") as out:
    for i, (r, scenario) in enumerate(remaining, 1):
        row = {"qa_id": r.qa_id, "domain": r.domain_name, "scenario": scenario}
        try:
            raw = llm14.chat_json(
                system_prompt=dmg.DAMAGE_SYSTEM_PROMPT,
                user_prompt=dmg.build_damage_prompt(scenario),
                temperature=0.0,
                json_schema=dmg.DamagedCase.model_json_schema(),
            )
            case = dmg.DamagedCase.model_validate(raw)
            ok, reason = dmg.validate_damage(case, scenario)
            row.update({
                "damaged": case.damaged,
                "removed": dmg.trim_removed(case.removed, scenario),
                "removed_raw": case.removed,
                "kept": case.kept,
                "valid": ok,
                "reject_reason": reason,
            })
            stats["pass" if ok else "fail"] += 1
        except Exception as e:
            row.update({"damaged": "", "removed": [], "valid": False,
                        "reject_reason": f"{type(e).__name__}: {str(e)[:150]}"})
            stats["error"] += 1

        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

        if i % 20 == 0:
            el = time.perf_counter() - started
            print(f"  [{i}/{len(remaining)}] 통과 {stats['pass']} / 탈락 {stats['fail']} "
                  f"/ 오류 {stats['error']} | {el/i:.1f}초/건")

print(f"\n완료: {dict(stats)}")

# %% [markdown]
# ## 4. 훼손 결과 확인
#
# 정답지가 쓸 만한지 눈으로 봅니다.

# %%
damaged_rows = [json.loads(l) for l in DAMAGED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
valid_rows = [r for r in damaged_rows if r.get("valid")]

print(f"전체 {len(damaged_rows)}건 / 통과 {len(valid_rows)}건 ({len(valid_rows)/len(damaged_rows):.1%})")

field_dist = Counter()
for r in valid_rows:
    field_dist.update(r["removed"])
print(f"\nremoved 필드 분포 (정답지가 어디 쏠려있는지):")
for f, n in field_dist.most_common():
    print(f"  {f}: {n}건 ({n/len(valid_rows):.0%})")

print(f"\n샘플 3건:")
for r in valid_rows[:3]:
    print(f"\n  원본: {r['scenario'][:100]}")
    print(f"  훼손: {r['damaged'][:100]}")
    print(f"  정답지(removed): {r['removed']}")

# %% [markdown]
# ## 5. 판정 스키마와 프롬프트
#
# 13개 필드 목록을 **전부 보여주고** 고르게 합니다.
# `removed`는 절대 보여주지 않습니다.

# %%
from pydantic import BaseModel, Field
from typing import Literal
from medcheck.damage import ClinicalField, FIELD_LABELS_KO


class QuestionFieldMapping(BaseModel):
    question: str
    fields: list[ClinicalField] = Field(
        default_factory=list,
        description="이 질문이 확인하려는 임상 필드 (해당 없으면 빈 배열)",
    )


class FieldJudgment(BaseModel):
    per_question: list[QuestionFieldMapping] = Field(default_factory=list)


JUDGE_SYSTEM_PROMPT = """
당신은 의료 질문을 임상 필드로 분류하는 분류기다.

주어진 각 질문이 아래 임상 필드 중 무엇을 확인하려는 것인지 판정한다.

규칙:
1. 한 질문이 여러 필드를 물을 수 있다. 해당하는 것을 모두 고른다.
2. 목록에 없는 내용을 묻는 질문이면 fields를 빈 배열로 둔다.
3. 질문의 표현이 달라도 의미가 같으면 같은 필드로 판정한다.
   예: "언제부터 아프셨나요", "증상 발생 시점은" → 둘 다 onset
4. 추측하지 말고 질문에 실제로 드러난 것만 판정한다.
5. JSON 객체 하나만 출력한다.
"""


def build_judge_prompt(questions: list[str]) -> str:
    schema = FieldJudgment.model_json_schema()
    field_list = "\n".join(
        f"- {name}: {label}" for name, label in FIELD_LABELS_KO.items()
    )
    q_list = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))

    return f"""
[임상 필드 목록]
{field_list}

[판정할 질문들]
{q_list}

[과제]
각 질문이 위 목록 중 어느 필드를 확인하려는 것인지 판정하라.
질문 텍스트는 원문 그대로 question에 넣는다.

[출력 JSON Schema]
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


# 프롬프트 확인
_demo = build_judge_prompt([
    "통증이 언제 시작되었습니까?",
    "영상 검사를 받아보셨나요?",
    "가족 중에 비슷한 증상을 겪은 분이 있나요?",
])
print(_demo[:1500])

# %% [markdown]
# ## 6. 판정 파일럿 — 3건만
#
# 판정이 제대로 되는지 먼저 확인합니다.

# %%
pilot_questions = [
    "통증이 언제 시작되었습니까?",
    "통증이 어느 부위로 퍼지는지 알려주세요.",
    "MRI나 CT 같은 영상 검사를 받아보셨나요?",
    "현재 복용 중인 약물이 있습니까?",
    "가족 중에 비슷한 증상을 겪은 분이 있나요?",
]

raw = llm14.chat_json(
    system_prompt=JUDGE_SYSTEM_PROMPT,
    user_prompt=build_judge_prompt(pilot_questions),
    temperature=0.0,
    json_schema=FieldJudgment.model_json_schema(),
)
judgment = FieldJudgment.model_validate(raw)

print("판정 결과:")
for m in judgment.per_question:
    print(f"  {m.question}")
    print(f"    → {m.fields}")

# %% [markdown]
# 위 결과가 상식적이면 (onset / radiation / exam_results / medications / 빈 배열)
# 다음 셀로 진행합니다. 이상하면 프롬프트를 조정하세요.

# %% [markdown]
# ## 7. 3-way 질문 생성
#
# 같은 damaged 입력에 대해 세 모델이 각각 질문을 만듭니다.
# 14B는 상한선(교사 성능) 역할입니다.

# %%
from medcheck.models import ClarificationAssessment
from medcheck.prompts import CLARIFICATION_SYSTEM_PROMPT, build_clarification_prompt


def generate_questions(model_name: str, model_path: str, adapter: str | None,
                       rows: list[dict], reuse_llm=None) -> list[dict]:
    """한 모델로 전체 damaged 입력에 대해 질문을 생성합니다."""
    print(f"\n{'='*60}\n[{model_name}] 시작")

    if reuse_llm is not None:
        client = reuse_llm
    else:
        os.environ["LOCAL_LLM_MODEL"] = model_path
        if adapter:
            os.environ["LOCAL_LLM_ADAPTER"] = adapter
        else:
            os.environ.pop("LOCAL_LLM_ADAPTER", None)
        client = LocalTransformersClient()

    results = []
    started = time.perf_counter()

    for i, row in enumerate(rows, 1):
        out = {"model": model_name, "qa_id": row["qa_id"],
               "domain": row["domain"], "damaged": row["damaged"],
               "removed": row["removed"]}
        try:
            raw = client.chat_json(
                system_prompt=CLARIFICATION_SYSTEM_PROMPT,
                user_prompt=build_clarification_prompt(row["damaged"]),
                temperature=0.0,
                json_schema=ClarificationAssessment.model_json_schema(),
            )
            a = ClarificationAssessment.model_validate(raw)
            out.update({
                "questions": [q.question for q in a.questions],
                "schema_ok": True,
            })
        except Exception as e:
            out.update({"questions": [], "schema_ok": False,
                        "error": f"{type(e).__name__}: {str(e)[:150]}"})
        results.append(out)

        if i % 20 == 0:
            el = time.perf_counter() - started
            print(f"  [{i}/{len(rows)}] {el/i:.1f}초/건")

    if reuse_llm is None:
        del client
        gc.collect()
        torch.cuda.empty_cache()

    print(f"[{model_name}] 완료 — {time.perf_counter()-started:.0f}초")
    return results


# 14B는 이미 로드돼 있으므로 재사용
gen_14b = generate_questions("14b", MODEL_14B, None, valid_rows, reuse_llm=llm14)

# %% [markdown]
# ## 8. 14B 해제 후 3B 두 개 실행

# %%
del llm14
gc.collect()
torch.cuda.empty_cache()
print("14B 해제 완료")

gen_base = generate_questions("base-3b", BASE_3B, None, valid_rows)
gen_trimmed = generate_questions("trimmed", BASE_3B, ADAPTER_TRIMMED, valid_rows)

all_gen = gen_14b + gen_base + gen_trimmed
with GEN_PATH.open("w", encoding="utf-8") as f:
    for row in all_gen:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"저장: {GEN_PATH} ({len(all_gen)}행)")

# %% [markdown]
# ## 9. 판정 — 14B 재로드
#
# 세 모델의 질문을 모두 같은 판정자로 분류합니다.

# %%
os.environ["LOCAL_LLM_MODEL"] = MODEL_14B
os.environ.pop("LOCAL_LLM_ADAPTER", None)
llm_judge = LocalTransformersClient()
print("판정용 14B 로드 완료")

# %%
judged = []
started = time.perf_counter()
targets_to_judge = [r for r in all_gen if r.get("questions")]

with JUDGE_PATH.open("w", encoding="utf-8") as out:
    for i, row in enumerate(targets_to_judge, 1):
        rec = dict(row)
        try:
            raw = llm_judge.chat_json(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=build_judge_prompt(row["questions"]),
                temperature=0.0,
                json_schema=FieldJudgment.model_json_schema(),
            )
            j = FieldJudgment.model_validate(raw)
            predicted = sorted({f for m in j.per_question for f in m.fields})
            rec.update({
                "predicted_fields": predicted,
                "per_question": [m.model_dump() for m in j.per_question],
                "judge_ok": True,
            })
        except Exception as e:
            rec.update({"predicted_fields": [], "judge_ok": False,
                        "judge_error": f"{type(e).__name__}: {str(e)[:150]}"})
        judged.append(rec)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()

        if i % 30 == 0:
            el = time.perf_counter() - started
            print(f"  [{i}/{len(targets_to_judge)}] {el/i:.1f}초/건")

print(f"\n판정 완료: {len(judged)}건")

# %% [markdown]
# ## 10. 채점 ★
#
# `removed`(정답지)와 `predicted_fields`(판정 결과)를 집합 비교합니다.
#
# - **recall**: 지운 항목 중 몇 개를 되물었나  ← 가장 중요
# - **precision**: 되물은 것 중 몇 개가 지운 항목이었나
# - **F1**: 둘의 조화평균

# %%
import pandas as pd


def score(removed: list[str], predicted: list[str]) -> dict:
    R, P = set(removed), set(predicted)
    if not R:
        return {}
    hit = len(R & P)
    recall = hit / len(R)
    precision = hit / len(P) if P else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1,
            "hit": hit, "n_removed": len(R), "n_predicted": len(P)}


rows_scored = []
for r in judged:
    if not r.get("judge_ok"):
        continue
    s = score(r["removed"], r["predicted_fields"])
    if s:
        rows_scored.append({**r, **s})

df = pd.DataFrame(rows_scored)

summary = df.groupby("model").agg(
    n=("qa_id", "count"),
    recall=("recall", "mean"),
    precision=("precision", "mean"),
    f1=("f1", "mean"),
    질문수=("n_predicted", "mean"),
).round(3)

# 14B가 맨 위로 오도록 정렬
order = ["14b", "base-3b", "trimmed"]
summary = summary.reindex([m for m in order if m in summary.index])
summary

# %% [markdown]
# ## 11. 필드별 적중률
#
# 어떤 항목을 잘 되묻고 어떤 걸 놓치는지 봅니다.

# %%
for model in order:
    sub = df[df["model"] == model]
    if sub.empty:
        continue
    hit_c, miss_c = Counter(), Counter()
    for _, r in sub.iterrows():
        R, P = set(r["removed"]), set(r["predicted_fields"])
        hit_c.update(R & P)
        miss_c.update(R - P)

    print(f"\n[{model}]")
    fields = sorted(set(hit_c) | set(miss_c),
                    key=lambda f: -(hit_c[f] + miss_c[f]))
    for f in fields:
        total = hit_c[f] + miss_c[f]
        print(f"  {f:16s} {hit_c[f]:3d}/{total:3d}  ({hit_c[f]/total:.0%})")

# %% [markdown]
# ## 12. 실제 사례 확인
#
# 수치만 보면 오해할 수 있으니, 실제 질문과 판정을 눈으로 봅니다.

# %%
sample_ids = df[df["model"] == "trimmed"].nlargest(2, "recall")["qa_id"].tolist()
sample_ids += df[df["model"] == "trimmed"].nsmallest(2, "recall")["qa_id"].tolist()

for qa_id in sample_ids:
    print(f"\n{'='*70}\nqa_id={qa_id}")
    for model in order:
        sub = df[(df["model"] == model) & (df["qa_id"] == qa_id)]
        if sub.empty:
            continue
        r = sub.iloc[0]
        print(f"\n  [{model}] recall={r['recall']:.0%}")
        print(f"    정답지: {r['removed']}")
        print(f"    판정  : {r['predicted_fields']}")
        for q in r["questions"]:
            print(f"    - {q}")

# %%
