# %% [markdown]
# # 3-way 비교 평가 — 베이스 3B vs full 어댑터 vs trimmed 어댑터
#
# 테스트 세트: **Validation split**에서 100건 (학습에 전혀 쓰이지 않은 데이터)
#
# 학습 데이터는 Training split에서만 만들었으므로, Validation은 완전한
# held-out입니다. qa_id 누수 검사도 아래에서 수행합니다.
#
# 주의: 모델을 하나씩 로드/해제하며 순차 실행합니다. 동시에 올리면 OOM입니다.

# %%
import sys, os, json, gc, time, random
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, "/home1/zye614/projects/medcheck")
os.chdir("/home1/zye614/projects/medcheck")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL = "/home1/zye614/projects/medcheck/sft_model/qwen2.5-3b-instruct"
ADAPTER_FULL = "sft_model/adapters/qwen3b-clarify-full-v2"
ADAPTER_TRIMMED = "sft_model/adapters/qwen3b-clarify-trimmed-v2"

OUT_PATH = Path("results/eval_3way_v2.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

N_TEST = 100
SEED = 123

print("작업 디렉토리:", os.getcwd())

# %% [markdown]
# ## 1. 테스트 세트 구성 (Validation split)

# %%
import medcheck.damage as dmg
from medcheck.data import load_dataset

records, _ = load_dataset("data")
validation = [r for r in records if r.split == "Validation"]
print(f"Validation 전체: {len(validation)}건")

# 시나리오 문항만, 길이 필터는 학습 데이터와 동일 기준
candidates = []
for r in validation:
    if not dmg.is_scenario_question(r.question):
        continue
    scenario = dmg.extract_scenario(r.question)
    if len(scenario) < 40:
        continue
    candidates.append((r, scenario))

print(f"시나리오 문항: {len(candidates)}건")

# 진료과 균형 샘플링
by_dom = defaultdict(list)
for r, s in candidates:
    by_dom[r.domain_name].append((r, s))

random.seed(SEED)
test_set = []
cap = max(1, N_TEST // len(by_dom))
for domain, items in by_dom.items():
    random.shuffle(items)
    test_set.extend([(domain, r, s) for r, s in items[:cap]])

random.shuffle(test_set)
test_set = test_set[:N_TEST]

print(f"\n테스트 세트: {len(test_set)}건")
for d, n in sorted(Counter(d for d, _, _ in test_set).items()):
    print(f"  {d}: {n}건")

# %% [markdown]
# ## 2. 학습 데이터 누수 검사 ★
#
# 테스트 qa_id가 학습에 쓰였는지 확인합니다. 0건이어야 정상입니다.

# %%
train_ids = set()
for name in ["results/sft_full/train.jsonl", "results/sft_full/eval.jsonl",
             "results/sft_trimmed/train.jsonl", "results/sft_trimmed/eval.jsonl"]:
    p = Path(name)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                train_ids.add(json.loads(line)["qa_id"])

test_ids = {r.qa_id for _, r, _ in test_set}
leak = test_ids & train_ids

print(f"학습에 쓰인 qa_id: {len(train_ids)}개")
print(f"테스트 qa_id: {len(test_ids)}개")
print(f"누수: {len(leak)}건  ← 0이어야 정상")
if leak:
    print(f"  누수된 id: {sorted(leak)[:10]}")

# %% [markdown]
# ## 3. 프롬프트 정의
#
# 학습 시점과 **완전히 동일**해야 합니다.
# build_sft_dataset_*.py에 넣은 것과 같은 내용입니다.

# %%
from medcheck.models import ClarificationAssessment

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
"""


def build_clarification_prompt(patient_case: str) -> str:
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


print("프롬프트 길이:", len(build_clarification_prompt("테스트")), "자")

# %% [markdown]
# ## 4. 평가 함수
#
# 모델 하나를 로드해서 100건을 돌리고, 끝나면 GPU에서 내립니다.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# # "정보 없음"을 나타내는 표기가 몇 가지나 섞여 나오는지 세기 위한 값들
# EMPTY_MARKERS = {None, "", "미상", "정보 없음", "없음", "N/A", "불명", {}, []}


def normalize_empty(value):
    """빈 값 표기를 문자열로 정규화 (일관성 측정용)"""
    if value is None:
        return "None"
    if isinstance(value, str):
        return f'"{value}"' if value in {"", "미상", "정보 없음", "없음", "N/A", "불명"} else None
    if isinstance(value, dict) and not value:
        return "{}"
    if isinstance(value, list) and not value:
        return "[]"
    return None


def run_eval(model_name: str, adapter_path: str | None):
    """모델 하나로 테스트 세트 전체를 돌립니다."""
    print(f"\n{'='*60}\n[{model_name}] 로딩 중...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map={"": 0},
        attn_implementation="sdpa",
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"  어댑터 적용: {adapter_path}")

    model.eval()
    model.config.use_cache = True

    results = []
    started = time.perf_counter()

    for i, (domain, r, scenario) in enumerate(test_set, 1):
        messages = [
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": build_clarification_prompt(scenario).strip()},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        latency = time.perf_counter() - t0

        generated = tokenizer.decode(
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        row = {
            "model": model_name,
            "qa_id": r.qa_id,
            "domain": domain,
            "scenario": scenario,
            "raw_output": generated[:4000],
            "latency": latency,
        }

        # 1) JSON 파싱
        parsed = None
        try:
            parsed = json.loads(generated)
            row["json_ok"] = True
        except Exception:
            # llm.py와 동일한 복구 시도
            try:
                from json_repair import repair_json
                parsed = json.loads(repair_json(generated))
                row["json_ok"] = True
                row["needed_repair"] = True
            except Exception as e:
                row["json_ok"] = False
                row["error"] = f"JSONError: {str(e)[:150]}"

        # 2) 스키마 검증
        if parsed is not None:
            try:
                a = ClarificationAssessment.model_validate(parsed)
                row.update({
                    "schema_ok": True,
                    "n_questions": len(a.questions),
                    "requires_clarification": a.requires_clarification,
                    "urgent": a.urgent,
                    "priorities": [q.priority for q in a.questions],
                    "empty_markers": sorted({
                        m for m in (
                            normalize_empty(v)
                            for v in (a.structured_case or {}).values()
                        ) if m
                    }),
                })
            except Exception as e:
                row.update({"schema_ok": False, "error": f"ValidationError: {str(e)[:150]}"})
        else:
            row["schema_ok"] = False

        results.append(row)

        if i % 20 == 0:
            elapsed = time.perf_counter() - started
            print(f"  [{i}/{len(test_set)}] {elapsed/i:.1f}초/건 "
                  f"| json {sum(x.get('json_ok', False) for x in results)} "
                  f"| schema {sum(x.get('schema_ok', False) for x in results)}")

    # GPU 정리
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[{model_name}] 완료 — {time.perf_counter()-started:.0f}초")
    return results


# %% [markdown]
# ## 5. 실행 — 베이스 모델 (파인튜닝 없음)
#
# 이게 하한선(baseline)입니다. 파인튜닝이 뭘 개선했는지 말하려면 반드시 필요합니다.

# %%
res_base = run_eval("base-3b", None)

# %% [markdown]
# ## 6. 실행 — full 어댑터

# %%
res_full = run_eval("full", ADAPTER_FULL)

# %% [markdown]
# ## 7. 실행 — trimmed 어댑터

# %%
res_trimmed = run_eval("trimmed", ADAPTER_TRIMMED)

# %% [markdown]
# ## 8. 결과 저장

# %%
all_results = res_base + res_full + res_trimmed

with OUT_PATH.open("w", encoding="utf-8") as f:
    for row in all_results:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"저장: {OUT_PATH} ({len(all_results)}행)")

# %% [markdown]
# ## 9. 비교표 ★
#
# 발표에 그대로 쓸 수 있는 수치입니다.

# %%
import pandas as pd

def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r.get("schema_ok")]
    return {
        "모델": name,
        "JSON 파싱": f"{sum(r.get('json_ok', False) for r in rows)}/{n}",
        "스키마 통과": f"{len(ok)}/{n}",
        "복구 필요": sum(r.get("needed_repair", False) for r in rows),
        "평균 질문 수": round(sum(r.get("n_questions", 0) for r in ok) / max(len(ok), 1), 2),
        "질문 0개": sum(1 for r in ok if r.get("n_questions") == 0),
        "clarification=False": sum(1 for r in ok if r.get("requires_clarification") is False),
        "urgent=True": sum(1 for r in ok if r.get("urgent")),
        "평균 지연(초)": round(sum(r["latency"] for r in rows) / n, 2),
    }

df = pd.DataFrame([
    summarize("base-3b (파인튜닝 없음)", res_base),
    summarize("full", res_full),
    summarize("trimmed", res_trimmed),
])
df

# %% [markdown]
# ## 10. 질문 개수 분포
#
# trimmed가 실제로 다양해졌는지 확인합니다.

# %%
for name, rows in [("base-3b", res_base), ("full", res_full), ("trimmed", res_trimmed)]:
    ok = [r for r in rows if r.get("schema_ok")]
    dist = dict(sorted(Counter(r["n_questions"] for r in ok).items()))
    print(f"{name:10s}: {dist}")

# %% [markdown]
# ## 11. 필드 표기 일관성
#
# "정보 없음"을 몇 가지 표기로 나타내는지 셉니다.
# 가짓수가 적을수록 다운스트림(app.py 렌더링)이 안전합니다.

# %%
for name, rows in [("base-3b", res_base), ("full", res_full), ("trimmed", res_trimmed)]:
    ok = [r for r in rows if r.get("schema_ok")]
    markers = Counter()
    for r in ok:
        markers.update(r.get("empty_markers", []))
    print(f"\n{name}: {len(markers)}가지 표기")
    for m, c in markers.most_common():
        print(f"    {m}: {c}건")

# %% [markdown]
# ## 12. 실패 사례 확인

# %%
for name, rows in [("base-3b", res_base), ("full", res_full), ("trimmed", res_trimmed)]:
    fails = [r for r in rows if not r.get("schema_ok")]
    print(f"\n{'='*60}\n{name}: 실패 {len(fails)}건")
    for r in fails[:3]:
        print(f"\n  qa_id={r['qa_id']} ({r['domain']})")
        print(f"  사유: {r.get('error', '')[:150]}")
        print(f"  출력 앞부분: {r['raw_output'][:200]}")
# %%
