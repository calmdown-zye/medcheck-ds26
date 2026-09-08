# MedCheck 배포 안내

이 문서는 macOS Apple Silicon, Linux NVIDIA GPU, CPU에서 동일한 배포 패키지를
사용하기 위한 기준입니다.

## 배포 패키지

```text
medcheck/
├── app.py
├── medcheck/
├── scripts/
├── tests/
├── sample_data/
├── artifacts/medical_index/       # 즉시 실행 배포라면 포함
├── .env.example
├── requirements.txt
├── pyproject.toml
├── README.md
├── DEPLOYMENT.md
└── LICENSE
```

AI-Hub 원본 데이터와 모델 가중치는 크기와 배포 권한을 확인한 뒤 별도로 전달합니다.
즉시 실행 패키지에는 최소한 검색 인덱스를 포함해야 합니다.

## 장치 선택

`MEDCHECK_DEVICE=auto`는 CUDA, Apple MPS, CPU 순서로 사용 가능한 장치를 선택합니다.
필요하면 `cuda`, `mps`, `cpu`로 고정합니다. CPU는 호환 경로이며 긴 입력이나 검증기
사용 시 응답 시간이 크게 증가합니다.

## 베이스 모델

```bash
python scripts/download_models.py
```

기본 저장 위치는 `models/Qwen2.5-3B-Instruct`입니다. 다른 위치는 다음처럼 지정합니다.

```bash
python scripts/download_models.py --output /path/to/Qwen2.5-3B-Instruct
```

```dotenv
LOCAL_LLM_MODEL=/path/to/Qwen2.5-3B-Instruct
```

## 어댑터 제공 규격

두 어댑터를 배포합니다. `trimmed-v2`는 권장 기본값이고 `full-v2`는 비교·재현용입니다.
서버에서 최종 어댑터를 다음처럼 내보냅니다.

```python
peft_model.save_pretrained(
    "qwen3b-clarify-trimmed-v2",
    safe_serialization=True,
)
```

최소 배포 구조:

```text
qwen3b-clarify-trimmed-v2/
├── adapter_config.json
└── adapter_model.safetensors
```

현재 배포 로더는 베이스 모델 토크나이저를 사용합니다. 토크나이저 어휘를 변경해 학습한
어댑터는 그대로 배포하지 말고, 임베딩 크기와 추가 토큰 로딩을 별도로 검증해야 합니다.
`checkpoint-*`, `optimizer.pt`, `scheduler.pt`, `trainer_state.json`,
`training_args.bin`, `rng_state.pth`는 추론 배포에 필요하지 않습니다.

전달 시 zip과 SHA-256 체크섬을 권장합니다.

```bash
zip -r qwen3b-clarify-trimmed-v2.zip qwen3b-clarify-trimmed-v2
shasum -a 256 qwen3b-clarify-trimmed-v2.zip
```

수신자는 두 압축파일을 `models/adapters` 아래에 풀고, 사용할 하나를 `.env`에
설정합니다. 어댑터를 바꾼 뒤에는 실행 중인 Streamlit 프로세스를 종료하고 다시
시작해야 합니다.

```dotenv
LOCAL_LLM_ADAPTER=models/adapters/qwen3b-clarify-trimmed-v2
```

비교·재현용 Full 어댑터:

```dotenv
LOCAL_LLM_ADAPTER=models/adapters/qwen3b-clarify-full-v2
```

배포된 압축파일과 같은 위치의 `.sha256` 파일로 압축 해제 전에 무결성을 확인합니다.

```bash
shasum -a 256 -c qwen3b-clarify-trimmed-v2.tar.gz.sha256
shasum -a 256 -c qwen3b-clarify-full-v2.tar.gz.sha256
```

## 검색 인덱스

즉시 실행 배포에는 다음 네 파일이 모두 필요합니다.

```text
artifacts/medical_index/
├── records.jsonl
├── embeddings.npy
├── tokens.pkl
└── index_config.json
```

즉시 실행 배포에는 `models/multilingual-e5-small`도 포함합니다. 프로젝트가 이 로컬
사본을 자동으로 우선 사용합니다. 포함하지 않으면 최초 검색 시 `index_config.json`에
기록된 임베딩 모델을 내려받습니다. 완전한 오프라인 배포에서는 모델을 포함하고
`MEDCHECK_OFFLINE=1`로 설정합니다.

## 배포 전 검증

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_pipeline_smoke_test.py \
  --index-dir artifacts/medical_index \
  --top-k 3 \
  --output results/smoke.json
```

검증기 경로는 `--verify`를 추가해 별도로 확인합니다. 스모크 테스트의 `REVIEW`는 실행
실패가 아니라 품질 검토가 필요하다는 뜻입니다.

## 실행

```bash
streamlit run app.py
```

내부 서버에서 외부 접속을 허용하려면:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

의료정보를 다루는 환경은 인증, TLS, 접근 로그, 보존 정책 없이 인터넷에 직접 노출하면
안 됩니다.

## 릴리스 체크리스트

- [ ] `.env`와 개인 절대경로가 배포물에 포함되지 않았다.
- [ ] Qwen 모델 shard가 완전하다.
- [ ] 어댑터의 베이스 모델이 Qwen2.5-3B-Instruct와 일치한다.
- [ ] 검색 인덱스 네 파일이 포함되어 있다.
- [ ] `scripts/check_environment.py`가 통과한다.
- [ ] 단위 테스트가 통과한다.
- [ ] Trimmed와 Full 어댑터 압축파일의 SHA-256이 일치한다.
- [ ] 권장 기본값인 Trimmed 어댑터의 스모크 결과를 검토했다.
- [ ] Full 어댑터가 정상 적재되는지 비교 실행으로 확인했다.
- [ ] 개인정보가 데이터와 로그에 포함되지 않았다.
- [ ] 외부 공개 전 인증·TLS·로그 정책을 적용했다.
