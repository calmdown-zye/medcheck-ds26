# MedCheck AI

MedCheck AI는 AI-Hub 전문 의학지식 데이터를 검색하고 로컬 언어모델로 구조화된
감별진단, 누락 정보, 다음 평가 단계를 제시하는 연구용 임상 의사결정 지원
프로토타입입니다.

> 연구·교육·시연 목적이며 확정 진단이나 처방을 대신하지 않습니다.

## 지원 환경

- Python 3.10~3.12
- macOS Apple Silicon(MPS), Linux NVIDIA GPU(CUDA), CPU
- 기본 모델: `Qwen/Qwen2.5-3B-Instruct`
- UI: Streamlit

CPU에서도 실행할 수 있지만 3B 모델 적재에 충분한 메모리가 필요하고 생성이 느릴 수
있습니다. Apple Silicon 또는 NVIDIA GPU 사용을 권장합니다.

## 빠른 시작

### 1. 설치

```bash
git clone <repository-url> medcheck
cd medcheck
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell의 활성화 명령은 `.venv\Scripts\Activate.ps1`입니다.

### 2. Qwen2.5-3B-Instruct 다운로드

```bash
python scripts/download_models.py
```

공식 Hugging Face 저장소에서 약 6.18GB를 받아
`models/Qwen2.5-3B-Instruct`에 저장합니다. 중단되면 같은 명령으로 이어받습니다.

### 3. 검색 인덱스 준비

배포물에 `artifacts/medical_index`가 포함되어 있으면 건너뜁니다. 직접 생성하려면:

```bash
python scripts/build_index.py \
  --data-root data/Training/02.라벨링데이터 \
  --output-dir artifacts/medical_index \
  --embedding-model intfloat/multilingual-e5-small \
  --batch-size 32
```

즉시 실행용 배포물에는 `models/multilingual-e5-small`도 포함합니다. 포함하지 않으면
첫 실행에 임베딩 모델 다운로드가 필요합니다.

### 4. 환경과 테스트 확인

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
```

실제 모델 스모크 테스트:

```bash
python scripts/run_pipeline_smoke_test.py \
  --index-dir artifacts/medical_index \
  --top-k 3 \
  --output results/smoke.json
```

### 5. 앱 실행

```bash
streamlit run app.py
```

## PEFT/LoRA 어댑터 배치

베이스 모델만 사용할 때 `.env`의 `LOCAL_LLM_ADAPTER`를 비워 둡니다.

```dotenv
LOCAL_LLM_ADAPTER=
```

두 어댑터를 모두 배포하며, 처음 실행할 때는 앞선 비교에서 질문 품질이 더
안정적이었던 `trimmed-v2`를 권장합니다. `full-v2`는 비교 및 재현용입니다.
어댑터는 디렉터리 전체를 다음 구조로 제공합니다.

```text
models/
├── Qwen2.5-3B-Instruct/
└── adapters/
    ├── qwen3b-clarify-trimmed-v2/  # 권장 기본값
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── qwen3b-clarify-full-v2/     # 비교·재현용
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── README.md                 # 선택
```

`.env`에는 상대경로를 사용합니다.

```dotenv
LOCAL_LLM_ADAPTER=models/adapters/qwen3b-clarify-trimmed-v2
```

Full 어댑터를 비교하려면 경로만 바꾼 뒤 앱을 다시 시작합니다.

```dotenv
LOCAL_LLM_ADAPTER=models/adapters/qwen3b-clarify-full-v2
```

추가 질문·감별진단·검증기를 위해 베이스 모델을 여러 번 적재하지 않습니다.
선택한 어댑터가 적용된 하나의 로컬 모델이 각 단계를 순차적으로 수행합니다.

체크포인트, optimizer 상태, 학습 로그, 베이스 모델 사본은 배포용 어댑터에 넣지
않습니다. 어댑터는 반드시 `Qwen/Qwen2.5-3B-Instruct`와 호환되어야 하며 현재 로더는
베이스 모델 토크나이저를 사용합니다.

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LOCAL_LLM_MODEL` | `models/Qwen2.5-3B-Instruct` | 베이스 모델 경로 |
| `LOCAL_LLM_ADAPTER` | 빈 값 | 선택적 PEFT 어댑터 경로 |
| `MEDCHECK_DEVICE` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `MEDCHECK_INDEX_DIR` | `artifacts/medical_index` | 검색 인덱스 경로 |
| `MEDCHECK_EMBEDDING_MODEL` | `models/multilingual-e5-small` | 검색 임베딩 모델 경로 |
| `MEDCHECK_TOP_K` | `3` | 검색할 문서 개수 |
| `MEDCHECK_OFFLINE` | `0` | `1`이면 임베딩 모델 다운로드 차단 |

전체 설정은 [.env.example](.env.example)을 참고하세요.

## 주요 구조

```text
medcheck/
├── app.py
├── medcheck/             # 파이프라인, 모델, 검색, 안전장치
├── scripts/              # 다운로드, 인덱싱, 점검, 평가
├── tests/                # 단위·회귀 테스트
├── requirements.txt
└── DEPLOYMENT.md
```

## CLI

```bash
python scripts/query_cli.py \
  --index-dir artifacts/medical_index \
  --case "60세 남성이 6개월 전부터 요통과 오른쪽 다리 저림을 호소한다."
```

검색만 확인하려면 `--retrieval-only`를 추가합니다.

## 알려진 평가 한계

- 3B 모델은 검색 문서 속 별도 환자 정보를 현재 환자 정보와 혼동할 수 있습니다.
- 구조화 근거 필드가 비거나 관련성이 낮은 문서를 인용할 수 있습니다.
- 동일한 소형 모델을 쓰는 검증기는 오류를 모두 포착하지 못할 수 있습니다.
- 테스트의 `PASS`는 의료적 정확성 보증이 아니라 소프트웨어 검사 통과를 뜻합니다.

배포 전에는 [DEPLOYMENT.md](DEPLOYMENT.md)의 체크리스트와 스모크 결과를 함께
검토하세요.

예선 제출 코드에서 어떤 모듈과 조건이 달라졌는지는
[docs/IMPLEMENTATION_CHANGES_FROM_PRELIMINARY.md](docs/IMPLEMENTATION_CHANGES_FROM_PRELIMINARY.md)를,
팀 발표 구성은 [docs/FINAL_ROUND_PRESENTATION_GUIDE.md](docs/FINAL_ROUND_PRESENTATION_GUIDE.md)를
참고하세요.
