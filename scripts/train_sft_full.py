# %% [markdown]
# # Qwen2.5-3B LoRA 파인튜닝 — full 버전
#
# 입력:  results/sft_full/train.jsonl, eval.jsonl
# 출력:  sft_model/adapters/qwen3b-clarify-full/
#
# 주의: 이 셀들을 실행하는 인터랙티브 창은 **빈 GPU**에서 띄워야 합니다.
#       14B가 올라가 있는 창과 같은 GPU면 OOM이 납니다.

# %%
import sys, os, json
from pathlib import Path

sys.path.insert(0, "/home1/zye614/projects/medcheck")
os.chdir("/home1/zye614/projects/medcheck")

# tokenizers 병렬 경고 억제
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL = "/home1/zye614/projects/medcheck/sft_model/qwen2.5-3b-instruct"
DATA_DIR = Path("results/sft_full_v2")                        # ← v2인지 확인
OUT_DIR = Path("sft_model/adapters/qwen3b-clarify-full-v2")   # ← v2인지 확인
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("작업 디렉토리:", os.getcwd())
print("베이스 모델 :", BASE_MODEL)

# %% [markdown]
# ## 1. GPU 확인 — 반드시 비어 있어야 합니다

# %%
import subprocess
import torch

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU 여유 메모리: {free/1024**3:.1f}GB / {total/1024**3:.1f}GB")

# %% [markdown]
# ## 2. 데이터 로드

# %%
from datasets import Dataset

def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

train_rows = load_jsonl(DATA_DIR / "train.jsonl")
eval_rows = load_jsonl(DATA_DIR / "eval.jsonl")

# trl SFTTrainer는 "messages" 컬럼만 있으면 chat template을 자동 적용합니다
train_ds = Dataset.from_list([{"messages": r["messages"]} for r in train_rows])
eval_ds = Dataset.from_list([{"messages": r["messages"]} for r in eval_rows])

print(f"train {len(train_ds)}건 / eval {len(eval_ds)}건")
print("\n첫 샘플 assistant 응답 (앞 300자):")
print(train_rows[0]["messages"][2]["content"][:300])

# %% [markdown]
# ## 3. 토크나이저 — 실제 토큰 길이 확인
#
# 문자 길이 4,279자가 실제로 몇 토큰인지 확인해서
# max_seq_length를 정합니다.

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lengths = []
for r in train_rows[:200]:
    text = tokenizer.apply_chat_template(r["messages"], tokenize=False)
    lengths.append(len(tokenizer(text)["input_ids"]))

lengths.sort()
p50 = lengths[len(lengths) // 2]
p95 = lengths[int(len(lengths) * 0.95)]
print(f"토큰 길이 — 중앙값 {p50} / 95퍼센타일 {p95} / 최대 {lengths[-1]}")

MAX_SEQ_LENGTH = 4096 if lengths[-1] <= 4000 else 6144
print(f"→ max_seq_length = {MAX_SEQ_LENGTH}")

# %% [markdown]
# ## 4. 모델 로드 (LoRA, 양자화 없음)
#
# A6000 48GB면 3B fp16 + LoRA 학습이 여유롭게 들어갑니다.
# bitsandbytes(QLoRA)는 torch 2.2.2와 충돌 위험이 있어 쓰지 않습니다.

# %%
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    device_map={"": 0},
    attn_implementation="sdpa",
)
model.config.use_cache = False          # gradient checkpointing과 충돌 방지
model.config.pretraining_tp = 1

print(f"모델 로드 완료: {model.config.model_type}")
print(f"파라미터 수: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# %% [markdown]
# ## 5. LoRA 설정
#
# doc1의 논문 설정(rank 1 / alpha 2)은 베이스가 Qwen2.5-3B가 아닌
# 다른 모델 기준이라 그대로 쓰지 않습니다.
# rank 16으로 시작해서, 과적합이 보이면 낮추는 쪽이 안전합니다.

# %%
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %% [markdown]
# ## 6. 학습 설정
#
# 1,130건 × 3 epoch ≒ 3,400 step (batch 1, grad_accum 8 기준 약 420 optimizer step)
# 데이터가 적으므로 epoch를 늘리기보다 적게 돌리고 eval loss를 봅니다.

# %%
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir=str(OUT_DIR),

    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # 실효 batch = 8
    per_device_eval_batch_size=1,

    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,

    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,                      # 샘플 경계를 유지 (형식 학습에 중요)

    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print("Trainer 준비 완료")

# %% [markdown]
# ## 7. 학습 실행
#
# 1,130건 × 3 epoch면 A6000에서 대략 1~2시간 예상됩니다.
# eval_loss가 epoch마다 찍히니, 2 epoch 이후 올라가면 과적합입니다.

# %%
train_result = trainer.train()

print("\n=== 학습 완료 ===")
for key, value in train_result.metrics.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## 8. 어댑터 저장

# %%
trainer.save_model(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))

print(f"저장 위치: {OUT_DIR.resolve()}")
print(subprocess.run(["ls", "-la", str(OUT_DIR)], capture_output=True, text=True).stdout)

# %% [markdown]
# ## 9. 학습 곡선 확인
#
# eval_loss가 계속 떨어지면 epoch를 늘려도 되고,
# 중간부터 올라가면 그 지점이 최적입니다.

# %%
history = trainer.state.log_history
train_losses = [(h["step"], h["loss"]) for h in history if "loss" in h]
eval_losses = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

print("train loss (마지막 10개):")
for step, loss in train_losses[-10:]:
    print(f"  step {step}: {loss:.4f}")

print("\neval loss:")
for step, loss in eval_losses:
    print(f"  step {step}: {loss:.4f}")

# %% [markdown]
# ## 10. 즉시 스모크 테스트
#
# 학습된 어댑터가 실제로 JSON을 뱉는지 확인합니다.
# eval 세트의 첫 샘플로 테스트합니다.

# %%
from medcheck.models import ClarificationAssessment

model.eval()
model.config.use_cache = True

sample = eval_rows[0]
messages = sample["messages"][:2]      # system + user (assistant는 정답이므로 제외)

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

print("=== 생성 결과 ===")
print(generated[:800])

print("\n=== 파싱 검증 ===")
try:
    parsed = json.loads(generated.strip())
    print("JSON 파싱: 성공")
    a = ClarificationAssessment.model_validate(parsed)
    print(f"스키마 검증: 성공 (질문 {len(a.questions)}개, "
          f"requires_clarification={a.requires_clarification})")
except Exception as e:
    print(f"실패: {type(e).__name__}: {str(e)[:300]}")

print("\n=== 정답(교사 출력) ===")
print(sample["messages"][2]["content"][:500])
# %%
import re
CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
found = CJK.findall(generated)
print(f"\n중국어/일본어: {'없음 ✅' if not found else f'발견 ❌ {found[:15]}'}")
# %%
