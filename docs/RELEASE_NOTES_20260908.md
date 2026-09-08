# MedCheck AI 팀 배포본 — 2026-09-08

## 구성

- Qwen2.5-3B-Instruct 기반 로컬 추론
- Dense + BM25 전문 의학지식 검색 인덱스
- 추가 질문과 사용자 답변 병합
- 감별진단별 지지·반대·누락 근거 및 검색 근거 ID
- 검증기와 다섯 판단 보류 경로
- 고위험·판단 보류 상황의 행동 카드
- macOS MPS, Linux CUDA, CPU 선택 로더
- Trimmed/Full LoRA 어댑터 별도 배포

## 권장 구성

- 기본 어댑터: `qwen3b-clarify-trimmed-v2`
- 비교·재현용: `qwen3b-clarify-full-v2`
- 베이스 모델: `Qwen/Qwen2.5-3B-Instruct`
- 임베딩 모델: `intfloat/multilingual-e5-small`

## 검증 기록

- Python 3.10 프로젝트 가상환경에서 `tests/test_damage.py`의 8개 회귀 테스트 통과
- 행동 카드의 트리거·정렬·요약·복사용 인계문 점검 통과
- `scripts/check_environment.py` 통과
- Full/Trimmed 어댑터의 로컬 Streamlit 적재·실행 확인
- 예선 제출본 대비 파일·함수·조건 비교 명세 포함

테스트 통과는 의료적 정확성 보증이 아니라 배포 파일과 소프트웨어 경로 점검을 뜻합니다.

## 알려진 한계

- 소형 모델이 검색 문서의 다른 환자 정보를 현재 환자 사실로 혼동할 수 있습니다.
- 일부 출력에서 반대·누락 근거나 추가 확인사항이 비어 있을 수 있습니다.
- 동일 모델의 별도 검증 호출이 모든 생성 오류를 포착하지는 못합니다.
- Cross-Encoder 재순위화와 문서 내 근거 구간 연결은 아직 구현되지 않았습니다.
- 포함된 AI-Hub 기반 검색 인덱스를 외부에 재배포하기 전 데이터 이용조건을 확인해야 합니다.
