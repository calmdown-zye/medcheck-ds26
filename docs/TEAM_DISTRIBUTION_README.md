# MedCheck AI 팀 배포파일 안내

## 전달 파일

1. `medcheck-team-runtime-20260908.tar.gz`  
   앱 소스, 검색 인덱스, 문서, 로고, 실행·점검 스크립트와 테스트
2. `qwen3b-clarify-trimmed-v2.tar.gz`  
   권장 기본 LoRA 어댑터
3. `qwen3b-clarify-full-v2.tar.gz`  
   비교·재현용 LoRA 어댑터
4. 각 압축파일과 같은 이름의 `.sha256` 파일  
   전송 중 파일 손상 확인용

Qwen2.5-3B-Instruct 베이스 모델과 Python 가상환경은 용량·운영체제 차이 때문에
포함하지 않습니다. 베이스 모델은 압축을 푼 뒤 제공된 다운로드 스크립트로 받습니다.

## 가장 빠른 설치 순서

```bash
tar -xzf medcheck-team-runtime-20260908.tar.gz
cd medcheck
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/download_models.py
```

두 어댑터 압축파일을 프로젝트의 `models/adapters`에 풉니다.

```bash
mkdir -p models/adapters
tar -xzf ../qwen3b-clarify-trimmed-v2.tar.gz -C models/adapters
tar -xzf ../qwen3b-clarify-full-v2.tar.gz -C models/adapters
cp .env.example .env
```

`.env`에서 권장 기본 어댑터를 선택합니다.

```dotenv
LOCAL_LLM_MODEL=models/Qwen2.5-3B-Instruct
LOCAL_LLM_ADAPTER=models/adapters/qwen3b-clarify-trimmed-v2
MEDCHECK_DEVICE=auto
MEDCHECK_INDEX_DIR=artifacts/medical_index
```

환경 점검과 앱 실행:

```bash
python scripts/check_environment.py
python -m unittest discover -s tests -p 'test_*.py'
streamlit run app.py
```

브라우저가 자동으로 열리지 않으면 터미널에 표시된 `http://localhost:8501`을 직접
엽니다. Full 어댑터를 비교할 때는 `.env`의 `LOCAL_LLM_ADAPTER`만
`models/adapters/qwen3b-clarify-full-v2`로 바꾸고 앱을 재시작합니다.

## 주의사항

- `trimmed-v2`가 권장 기본값이며, 두 어댑터를 동시에 적용하지 않습니다.
- 테스트 통과는 의료적 정확성 보증이 아니라 소프트웨어 경로 점검 결과입니다.
- 이 프로젝트는 연구·교육·시연용이며 확정 진단이나 처방을 대신하지 않습니다.
- AI-Hub 원본 데이터는 포함하지 않았습니다. 포함된 검색 인덱스의 외부 재배포 전에는
  데이터 이용조건을 별도로 확인해야 합니다.
- 상세 설치와 문제 해결은 `README.md`, 배포 체크리스트는 `DEPLOYMENT.md`, 발표 내용은
  `docs/FINAL_ROUND_PRESENTATION_GUIDE.md`를 확인합니다.
- 예선 코드와 현재 코드의 함수·조건·호출 흐름 비교는
  `docs/IMPLEMENTATION_CHANGES_FROM_PRELIMINARY.md`를 확인합니다.
