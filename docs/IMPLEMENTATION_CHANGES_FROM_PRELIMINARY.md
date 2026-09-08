# MedCheck AI 예선본 대비 구현 변경 명세

> 비교 기준  
> 예선 제출본: 프로젝트 내부 비교용 `K-DS2026-main/`  
> 현재 본선 준비본: 프로젝트 루트

이 문서는 발표 요약이 아니라 팀원이 현재 코드를 이해하고 유지보수하기 위한 구현 명세다. 예선에 이미 있던 기능, 본선에서 강화한 기능, 새로 추가한 모듈, 실험 후 운영 경로에서 제외한 코드를 구분한다.

## 1. 먼저 알아야 할 결론

### 예선부터 존재했던 핵심 기능

- 환자 사례 구조화
- 최대 3개의 추가 질문과 답변 병합
- Dense + BM25 하이브리드 검색
- 감별진단·다음 단계 단위의 `E1~Ek` 근거 ID
- 별도 LLM 호출을 이용한 검증기
- LLM 판단, 검증기, 규칙을 결합한 판단 보류
- 확정 진단·구체 처방 요구 탐지

따라서 위 기능 전체를 “본선에서 새로 만들었다”고 설명하면 안 된다.

### 본선 준비 과정에서 새로 추가하거나 강화한 부분

- 심사 피드백을 화면 행동으로 연결하는 `medcheck/action_card.py`
- 결과 화면의 정보 밀도와 가독성을 분리한 `medcheck/ui.py`
- 불완전 초진 사례를 만드는 `medcheck/damage.py`
- 추가 질문 단계의 판단 유예 배너와 질문 우선순위 배지
- 환자 사례와 검색 문서의 역할을 더 엄격히 나눈 결정·검증 프롬프트
- 잘못 배치된 `E숫자` 인용 보정, 중복 진단 제거, 최대 3개 제한
- 진단이 없는 결과와 강제 안전 보류의 후처리 강화
- LoRA 어댑터 적재와 모델 파일 무결성 검사
- macOS MPS·Linux CUDA·CPU를 포괄하는 로컬 실행 구조
- 손상 회귀 테스트, 스모크 테스트, 파일럿 재실행 도구
- README, 배포 문서, 환경 점검 및 모델 다운로드 스크립트

## 2. 전체 실행 구조와 신규 모듈의 위치

```text
app.py
 ├─ medcheck.llm.LocalTransformersClient
 │    └─ Qwen2.5-3B + 선택한 LoRA 어댑터 한 개
 ├─ medcheck.index.MedicalHybridIndex
 │    └─ Dense + BM25 검색 → EvidenceItem(E1~Ek)
 ├─ medcheck.pipeline.MedCheckPipeline
 │    ├─ assess_clarification()
 │    ├─ merge_clarification_answers()
 │    └─ _run_decision() → 선택적 검증
 ├─ medcheck.action_card
 │    └─ 응급·보류 상황의 우선 정보 카드
 └─ medcheck.ui
      └─ 구조화 정보·지표·감별진단·빈 상태 렌더링

오프라인 품질 경로
 ├─ medcheck.damage
 ├─ tests/test_damage.py
 ├─ tests/test_action_card.py
 ├─ scripts/run_pipeline_smoke_test.py
 └─ scripts/replay_pilot.py
```

추가 질문, 감별진단, 검증기를 위해 베이스 모델을 각각 따로 적재하지 않는다. `app.py:get_runtime()`이 검색 인덱스와 하나의 `LocalTransformersClient`를 만들고, 같은 클라이언트를 파이프라인의 여러 LLM 호출에서 순차적으로 사용한다.

## 3. 파일 단위 변경 요약

### 예선본에 없던 주요 파일

| 파일 | 역할 | 운영 경로 여부 |
|---|---|---|
| `medcheck/action_card.py` | 응급·보류 시 진료 이유, 준비정보, 증상 요약을 우선 표시 | 사용 |
| `medcheck/ui.py` | 결과 지표·구조화 정보·감별진단 카드 렌더링 | 사용 |
| `medcheck/damage.py` | 완성된 임상 시나리오에서 정보를 제거해 불완전 초진 사례 생성 | 오프라인 데이터·평가 |
| `tests/test_damage.py` | 환자정보 혼입, 인용, 보류 동작 회귀 검사 | 테스트 |
| `tests/test_action_card.py` | 행동 카드의 트리거와 추출 함수 검사 | 테스트 |
| `scripts/run_pipeline_smoke_test.py` | 실제 모델을 이용한 대표 사례 품질 점검 | 평가 |
| `scripts/replay_pilot.py` | JSON/JSONL 사례를 재실행하고 중단 후 재개 | 평가 |
| `scripts/check_environment.py` | 모델·어댑터·인덱스·장치 점검 | 배포 |
| `scripts/download_models.py` | Qwen2.5-3B와 임베딩 모델 다운로드 | 배포 |
| `scripts/build_sft_dataset_*.py` | Full/Trimmed SFT 데이터셋 생성 | 서버 연구용 |
| `scripts/train_sft_*.py` | 두 LoRA 어댑터 학습 | 서버 연구용 |
| `scripts/eval_3way.py` | Base·Full·Trimmed 비교 | 서버 연구용 |
| `scripts/eval_question_quality.py` | 제거 필드 회수 기준 질문 품질 비교 | 서버 연구용 |
| `.env.example` | 상대경로 기반 실행 설정 예시 | 배포 |
| `DEPLOYMENT.md` | 범용 배포와 어댑터 전달 규격 | 배포 |

서버 연구용 스크립트에는 서버 절대경로가 남아 있으므로 팀 실행용 압축본에서는 제외했다. `server_ready/pipeline.py`와 `server_ready/prompts.py`도 운영 기준 파일이 아니며 비교·배포 대상에서 제외한다.

### 예선본에서 변경된 주요 파일

| 파일 | 예선본 | 현재본 |
|---|---|---|
| `app.py` | 기본 Streamlit 결과와 단순 경고 | 앱형 디자인, 단계 표시, 행동 카드, 판단 유예 배너, 우선순위 배지 |
| `medcheck/pipeline.py` | 기본 출력 정규화와 보류 | 긴급도 보정, 잘못 배치된 ID 이동, 중복 제거, 보류 후처리 강화 |
| `medcheck/prompts.py` | 일반적인 근거·보류 규칙 | 환자 사실과 검색 지식의 경계, 필수 근거 필드, 구체적 검증 사유 강화 |
| `medcheck/llm.py` | 지정된 로컬 베이스 모델만 적재 | 기본 상대경로, LoRA, 파일 완전성, 라이브러리 버전·장치 호환 강화 |
| `medcheck/index.py` | 로컬 캐시에 있는 임베딩 모델 사용 | 번들 상대경로와 온라인·오프라인 선택 지원 |
| `README.md` | 개발 환경 중심 | 최초 사용자 설치, 모델 다운로드, 어댑터, 한계 안내 |

`medcheck/models.py`, `medcheck/constants.py`, `medcheck/safety.py`는 예선본과 현재본이 동일하다. 즉 `NextStep`, `RedFlag`, `ClarificationAssessment`, 고위험 문자열 목록과 기본 인용 필터는 예선에 이미 있었다.

## 4. 행동 카드 구현 명세

### 4.1 “행동”을 무엇으로 정의했는가

현재 코드에는 서로 다른 두 의미가 있다.

#### A. 임상 의사결정의 다음 단계

`medcheck/models.py:NextStep`이 정의한다.

```text
action       실제로 추가 수행할 평가·조치의 이름
rationale    그 조치가 필요한 이유
urgency      즉시 | 긴급 | 일반 | 선택
evidence_ids 조치의 타당성을 뒷받침하는 검색 문서 ID
```

이 값은 `ClinicalDecision.next_steps`에 저장되고 최종 결과의 요약 탭과 근거 연결 탭에서 표시된다.

#### B. 심사 피드백에 대응하는 행동 카드

`medcheck/action_card.py`가 정의한다. 새 의학적 행동을 생성하지 않고 이미 생성된 필드를 다음 세 묶음으로 재구성한다.

1. **진료가 필요한 이유**: `red_flags`; 추가 질문 단계에서는 `urgent_message`
2. **방문 시 준비할 정보**: 최종 단계의 `missing_information` 또는 추가 질문 단계의 `questions`
3. **의료진에게 보여줄 증상 요약**: `structured_case`의 비어 있지 않은 필드

판단 보류 사유인 `abstain_reason`은 카드 제목 아래 설명으로 표시된다. 의료진에게 복사할 수 있는 평문은 `format_handoff_text()`가 만든다.

중요: 현재 행동 카드는 `next_steps`의 `action` 목록을 카드 본문에 직접 표시하지 않는다. `next_steps`는 긴급도 판정에만 사용되며 실제 목록은 최종 결과의 “다음 단계” 영역에 표시된다. 따라서 현재 카드는 정확히는 **행동 결정을 돕는 우선 정보 카드**다. “구체 조치를 행동 카드 안에 통합했다”고 발표하려면 `next_steps` 렌더링을 추가 구현해야 한다.

### 4.2 행동 카드가 나타나는 조건

진입점은 `should_show_action_card(decision=None, assessment=None)`이다.

```text
추가 질문 단계
  assessment.urgent == True
    → 행동 카드 표시

최종 결과 단계
  decision.abstain == True
    → 행동 카드 표시
  또는 infer_urgency(decision)가 즉시/긴급
    → 행동 카드 표시

그 외
  → 행동 카드 미표시
```

`infer_urgency()`는 `red_flags[*].urgency`와 `next_steps[*].urgency`를 합쳐 가장 높은 수준을 선택한다.

| 입력 수준 | 내부 순위 | 행동 카드 트리거 |
|---|---:|---|
| 즉시 | 3 | 예 |
| 긴급 | 2 | 예 |
| 주의 | 1 | 아니오 |
| 일반·선택·낮음·미지정 | 0 | 아니오 |

따라서 `red_flags`가 비어도 `next_steps`에 `즉시` 또는 `긴급` 조치가 있으면 행동 카드가 표시된다.

### 4.3 카드에 들어가는 데이터 선택 규칙

#### 진료 이유: `extract_reasons()`

- 최종 결과에서는 모든 `red_flags`를 수집한다.
- `finding`, `reason`, `urgency`, `evidence_ids`를 유지한다.
- `즉시 → 긴급 → 주의 → 일반` 순으로 정렬한다.
- 추가 질문 단계에서 `assessment.urgent=True`이고 `urgent_message`가 있으면 “응급 가능성 감지” 항목을 맨 앞에 넣는다.
- 보류 사유는 이 목록에 넣지 않고 카드 부제에서 별도로 표시한다.

#### 방문 준비정보: `extract_preparations()`

- 최종 결과: `decision.missing_information`
- 추가 질문 단계: `assessment.questions`
- 질문 문장이 완전히 같은 항목을 중복 제거한다.
- `필수 → 중요 → 보조` 순으로 정렬한다.

#### 증상 요약: `build_symptom_summary()`

- 표준 필드: 나이, 성별, 주호소, 증상, 발생 시점, 지속기간, 통증 양상, 방사 여부, 활력징후, 과거력, 복용약
- 모델 변형 키를 표준 키로 바꾼다. 예: `pain_radiation → radiation`, `gender → sex`
- 빈 문자열, `미상`, `정보 없음`, `없음`, `N/A`, `불명`, `미지수`, 빈 목록·사전은 제외한다.
- 알려지지 않은 추가 키는 값이 비어 있지 않으면 마지막에 그대로 표시한다.

### 4.4 화면 연결 위치

- `app.py:render_clarification_stage()`  
  `assessment.urgent=True`이면 질문보다 행동 카드를 먼저 그린다.
- `app.py:render_pipeline_result()`  
  보류 또는 긴급 결과이면 지표·탭보다 행동 카드를 먼저 그린다.
- 일반 결과에서는 행동 카드 대신 성공 안내를 표시한다.

### 4.5 예선본과의 차이

예선 `app.py`는 추가 질문 단계에서 `st.error(urgent_message)` 한 줄만 표시했고, 결과 단계에서는 `st.error("판단 보류")` 또는 `st.success(...)`를 표시했다. 현재본은 이 단순 문구를 행동 카드 호출로 대체했다. 이것이 심사 피드백의 가장 직접적인 코드 변경이다.

### 4.6 현재 한계

- 보류됐지만 `red_flags`와 `missing_information`이 모두 비면 카드에는 보류 제목과 사유, 가능한 경우 증상 요약만 나타난다.
- `next_steps` 실제 목록은 카드 내부가 아니라 최종 요약 탭에 있다.
- 카드 내용은 기존 모델 출력에 의존하므로, 모델이 필드를 비우면 UI만으로 내용을 만들 수 없다.
- 카드의 복사용 평문은 임상 문서가 아니라 참고용 요약이다.

## 5. 추가 질문과 판단 유예 구현 명세

### 5.1 파이프라인의 실행 중단은 예선부터 존재

`MedCheckPipeline.run()`의 핵심 상태 전이는 예선본과 동일하다.

```text
최초 실행
  → assess_clarification(patient_case)
  → requires_clarification=True이면 ClarificationAssessment 반환
  → 검색·감별진단을 실행하지 않고 종료

사용자 답변 제출
  → merge_clarification_answers(...)
  → 원 사례와 질문/답변 병합
  → _run_decision(...)
```

따라서 본선 변경은 “질문 전에 멈추는 기능을 새로 만든 것”이 아니라, 이미 있던 중단 상태를 사용자가 명확히 인식하도록 화면과 프롬프트를 강화한 것이다.

### 5.2 질문 필요 상태의 코드 보정

`assess_clarification()`은 모델 출력을 받은 후 다음 규칙을 적용한다.

1. 빈 질문 제거
2. 공백을 제거하고 소문자로 바꾼 질문 문장을 기준으로 중복 제거
3. 앞에서부터 최대 3개 유지
4. `priority="필수"` 질문이 하나라도 있으면 `requires_clarification=True`로 강제
5. 질문이 하나도 없으면 `requires_clarification=False`로 강제

이 후처리도 예선본에 이미 존재했다.

### 5.3 본선에서 화면에 추가된 표현

`app.py:render_clarification_hold_notice()`는 새 표현 계층 함수다.

- 조건: `assessment.requires_clarification=True`이면서 `assessment.urgent=False`
- 표시: “현재는 판단을 유예하고 추가 정보를 확인합니다”
- 배지: 질문 목록을 세어 `필수 N`, `중요 N`, `보조 N` 표시
- 개별 질문: 각 카드 제목 앞에 우선순위 배지 표시
- 응급이면 일반 유예 배너를 생략하고 행동 카드를 우선 표시

이 함수는 모델 출력이나 파이프라인 상태를 변경하지 않는다.

### 5.4 프롬프트 강화

현재 `CLARIFICATION_SYSTEM_PROMPT`와 `build_clarification_prompt()`에는 다음 규칙이 추가됐다.

- 감별진단·치료·검사 권고를 질문 단계에서 생성하지 않음
- 이미 제공된 정보를 다시 묻지 않음
- 답변에 따라 판단이 달라질 질문만 선택
- 정보가 충분하면 억지로 질문하지 않고 빈 배열 허용
- 필수 질문이 있으면 질문 필요 상태 설정
- 한국어 중심 서술

다만 현재 프롬프트에는 베체트병과 흉통 사례의 구체 예시가 남아 있다. 이는 실행 중인 운영 프롬프트이며, 특정 사례 편향 여부는 별도 일반화 평가가 필요하다.

### 5.5 확인된 구현 문제

현재 `merge_clarification_answers()` 결과에 `[원래 환자 사례]` 표지가 두 번 연속 들어간다. 예선본에는 한 번만 있었다. 기능을 중단시키지는 않지만 모델 입력 중복이므로 정리 대상이다.

## 6. 판단 보류 장치의 실제 구현

“다섯 보류 조건”은 하나의 함수에 5개 `if`문으로 모여 있지 않다. 프롬프트, 질문 전 게이트, 규칙 기반 강제 보류, 결정 후처리, 검증기 후처리에 분산돼 있다.

### 6.1 조건별 코드 경로

| 발표자료의 조건 | 실제 입력/판정 | 실제 처리 | 예선 대비 |
|---|---|---|---|
| 필수 임상정보 부족 | `ClarificationAssessment.requires_clarification` | 최종 분석 전에 `ClarificationAssessment`를 반환하고 실행 대기 | 기존, UI·프롬프트 강화 |
| 확정 진단·구체 처방 요구 | `requires_forced_abstention(patient_case)`가 `HIGH_RISK_PATTERNS` 포함 여부 검사 | `abstain=True`; 현재본은 진단과 다음 단계도 제거 | 기존, 출력 차단 강화 |
| 구조화 결과가 비거나 불완전 | Pydantic 스키마 검증 실패 또는 최종 진단 목록 없음 | 스키마 실패는 예외; 진단이 없으면 보류 | 부분 강화 |
| 검색 지식과 연결된 감별진단 없음 | 인용 정리 후 모든 진단의 `evidence_ids`가 비어 있음 | `abstain=True`와 고정 사유 | 기존 |
| 핵심 주장 검증 실패 | `needs_abstention=True` 또는 `overall_grounded=False` | `abstain=True`, 가능한 경우 검증 사유 사용 | 기존, 검증 프롬프트 강화 |

### 6.2 강제 안전 보류 조건

`medcheck/constants.py:HIGH_RISK_PATTERNS`의 문자열 중 하나가 입력에 포함되면 활성화된다.

```text
확진해
확정 진단
검사 없이 진단
약 용량
투여량
처방해
처방전
수술 여부를 결정
```

`requires_forced_abstention()`은 입력을 소문자화하고 연속 공백을 정리한 뒤 단순 포함 여부를 검사한다. 의미 분류 모델이나 정규식 기반 의도 분석은 아니다.

현재 강제 보류 결과:

```text
decision.abstain = True
decision.differential_diagnoses = []
decision.next_steps = []
decision.abstain_reason = 고정 안전 안내
result.forced_safety_abstention = True
```

예선본도 강제 보류를 설정했지만 진단과 다음 단계를 명시적으로 비우지는 않았다.

### 6.3 비강제 판단 보류

검색 근거 부족이나 검증기 실패로 보류될 때는 기존 감별진단을 삭제하지 않는다. 이유는 결과가 왜 보류됐는지 사후 검토할 수 있도록 하기 위해서다. 이것은 “보류이면 모든 진단 삭제”와 구분된다.

### 6.4 빈 결과 처리의 변경

예선본:

```text
감별진단도 없고 next_steps도 없을 때만 보류
```

현재본:

```text
감별진단이 없으면 next_steps 존재 여부와 관계없이 보류
```

즉 “다음 단계만 있고 감별진단이 없는 결과”를 정상 감별 결과로 통과시키지 않는다.

### 6.5 구조 불완전 검사의 정확한 범위

`_has_incomplete_differential_evidence()`는 지지 근거, 반대·누락 근거, 추가 확인사항 중 하나라도 비었는지 알려주는 순수 검사 함수다. 그러나 현재 운영 파이프라인은 이 결과로 자동 보류하거나 재생성하지 않는다. 스모크·회귀 검토를 위한 관찰 도구다.

또한 Pydantic 모델은 해당 목록들의 빈 배열을 허용한다. 따라서 “모든 필수 근거 필드가 비면 운영 코드가 무조건 보류한다”고 설명하면 부정확하다.

## 7. 근거 신뢰성과 회귀 방지 구현 명세

### 7.1 환자 사실과 검색 지식의 역할 분리

현재 `DECISION_SYSTEM_PROMPT`는 다음 경계를 명시한다.

- `supporting_evidence`: 현재 환자 사례에서 직접 확인된 사실만 사용
- `contradicting_or_missing_evidence`: 현재 환자와 맞지 않는 사실 또는 아직 부족한 정보
- `recommended_checks`: 후보를 확인·배제하기 위한 추가 평가
- `evidence_ids`: 후보의 의학적 타당성을 뒷받침하는 검색 문서

후보는 환자 사례의 직접 지지 사실과 검색 문서가 모두 있을 때만 생성하도록 지시한다. 세 개를 채우기 위한 관련 없는 후보 생성도 금지한다.

이것은 프롬프트 규칙이며 완전한 런타임 보장은 아니다. 소형 모델이 규칙을 어긴 사례가 실제로 관찰됐다.

### 7.2 인용 ID 처리

#### 예선부터 있던 처리

`_sanitize_citations()`는 다음 위치의 ID를 실제 검색 결과의 `valid_ids`와 비교한다.

- `red_flags[*].evidence_ids`
- `differential_diagnoses[*].evidence_ids`
- `next_steps[*].evidence_ids`

검색 결과에 없는 ID는 제거한다.

#### 현재 추가된 처리

소형 모델이 `E1` 같은 ID를 `supporting_evidence`나 `contradicting_or_missing_evidence`에 넣는 경우가 있어, `^E\d+$`와 정확히 일치하는 항목만 서술 필드에서 빼고 `evidence_ids`로 이동한다. 이후 기존 인용 정리기를 적용한다.

### 7.3 감별진단 목록 처리

현재 `_run_decision()`은 다음 순서로 목록을 정리한다.

1. 진단명을 `strip().casefold()`로 정규화
2. 빈 이름 제거
3. 같은 이름 중 첫 항목만 유지
4. 최대 3개 유지

이 3개는 검색 문서 Top-K와 다른 값이다. 감별진단 출력 상한이며, 현재 앱에서 사용자가 조절하지 않는다.

### 7.4 검증기 강화

현재 `VERIFIER_SYSTEM_PROMPT`는 예선 규칙에 다음을 추가했다.

- 환자 입력에 없는 검사 결과·진단·증상·과거력을 환자 사실처럼 쓴 주장은 `unsupported`
- 검색 문서의 다른 환자 검사 결과를 현재 환자 검사처럼 썼으면 `needs_abstention=True`
- 보류 사유에는 문제가 된 주장과 누락된 환자 사실을 구체적으로 작성
- 모호한 일반론만으로 보류 사유를 작성하지 않음

검증 결과의 ID도 실제 검색 ID만 남긴다.

### 7.5 Damage 모듈

`medcheck/damage.py`는 런타임 보호 장치가 아니라 학습·평가 데이터를 만드는 오프라인 모듈이다.

목적:

```text
AI-Hub의 검사·진단이 끝난 완성 시나리오
  → exam_results와 diagnosis를 반드시 제거
  → 추가 임상 필드 2~4개 제거
  → 불완전한 초진 사례 + 제거 정답지 생성
```

주요 구성:

- `is_scenario_question()`: 나이·성별로 시작하는 시나리오 문항 판정
- `extract_scenario()`: 객관식 선택지와 시험 지시문 제거
- `build_damage_prompt()`: 정보 제거용 프롬프트
- `detect_fields()`: 텍스트에서 임상 필드 흔적 탐지
- `validate_damage()`: 제거·유지 라벨과 실제 텍스트 일치 검증
- `trim_removed()`: 원본에 없던 제거 라벨 정리

### 7.6 회귀 테스트가 실제로 보장하는 것

`tests/test_damage.py`는 다음 8가지를 검사한다.

1. 알려진 검색 문서 사실이 환자 지지 근거로 섞이면 탐지
2. 정상 환자 근거는 혼입으로 오인하지 않음
3. 존재하지 않는 근거 ID 탐지
4. 실제 검색 ID 허용
5. 비강제 보류 시 진단 유지
6. 강제 보류 시 진단이 없어도 허용
7. 인용 정리 때문에 보류돼도 진단은 감사 목적으로 유지
8. 다음 단계만 생성되고 진단이 없으면 보류

첫 번째 검사는 대표 사례에 미리 등록한 금지 표현의 정확 일치를 찾는다. 임의의 모든 환자정보 혼입을 자동 탐지하는 범용 런타임 가드는 아니다.

`tests/test_action_card.py`는 Streamlit 렌더링 자체가 아니라 순수 추출 로직을 검사한다.

- 카드 트리거
- 긴급도 정렬
- 준비정보 우선순위 정렬
- 빈 구조화 필드 제외
- 복사용 인계문 형식

### 7.7 스모크 테스트의 판정 의미

`scripts/run_pipeline_smoke_test.py`는 결과를 세 단계로 나눈다.

- `FAIL`: 검색 결과에 없는 ID, 검색 근거 0개, 강제 보류인데 진단이 남음
- `REVIEW`: 진단 없음, 알려진 환자정보 혼입 표현, 빈 근거 필드, 판단 보류
- `PASS`: 위 오류와 경고가 없음

`REVIEW`는 프로그램 실행 실패가 아니라 사람이 출력 품질을 검토해야 한다는 뜻이다. `PASS`도 의료적 정확성을 보증하지 않는다.

## 8. UI 구조 개선

### `medcheck/ui.py`

예선에는 없던 표현 전용 모듈이다.

- `render_structured_case()`: 원본 JSON을 한국어 라벨 표로 변환
- `render_metrics()`: 응급도·진단 수·근거 수·검증 상태 카드
- `render_differential_diagnoses()`: 접힌 expander 대신 후보별 지지·반대·추가 확인 카드
- `render_key_value_list()`: 위험 신호·누락 정보·다음 단계 공통 렌더링
- `empty_note()`: 빈 상태를 과도하게 강조하지 않는 중립 안내

표현 계층은 모델 출력과 파이프라인 상태를 변경하지 않는다.

### `app.py`

주요 본선 변경:

- K-DS·AI-Hub 로고를 로컬 자산에서 data URI로 표시
- 입력·추가 질문·분석·결과 단계를 시각화
- 모델과 어댑터 경로까지 포함한 Streamlit 캐시 키 사용
- 추가 질문의 우선순위 배지와 판단 유예 배너
- 행동 카드 우선 배치
- 새 `ui.py`의 결과 카드 사용
- 모바일 너비를 고려한 반응형 CSS

## 9. 모델·어댑터 및 범용 실행 변경

### 예선본

- `LOCAL_LLM_MODEL`에 절대 또는 로컬 경로를 반드시 지정
- 베이스 Transformers 모델만 적재
- 모델 파일은 `config.json` 존재 여부 정도만 확인

### 현재본

`medcheck/llm.py`가 다음을 추가한다.

- 기본 모델 경로: 프로젝트 기준 `models/Qwen2.5-3B-Instruct`
- 상대경로는 프로젝트 루트 기준으로 해석
- shard 인덱스와 예상 전체 크기로 불완전 다운로드 검사
- `LOCAL_LLM_ADAPTER`가 있으면 `PeftModel.from_pretrained()` 적용
- 어댑터 설정과 가중치 존재 확인
- Transformers 버전에 따라 `dtype`/`torch_dtype` 인자 선택
- CUDA → MPS → CPU 자동 선택
- 비샘플링 생성에서는 `temperature`, `top_p`, `top_k`를 비활성화
- 출력에 `</think>`가 있으면 이후 JSON만 사용
- `{"clinical_decision": {...}}` 같은 불필요한 최상위 래퍼 해제

같은 앱 프로세스에서 Full과 Trimmed를 동시에 적재하지 않는다. 환경변수로 하나를 선택하고 앱을 재시작한다.

### 검색 모델

`medcheck/index.py:resolve_embedding_model()`은 `MEDCHECK_EMBEDDING_MODEL`의 상대경로, 프로젝트의 번들 모델, Hugging Face 모델 ID 순으로 사용할 수 있게 한다. `MEDCHECK_OFFLINE=1`이면 로컬 파일만 허용한다. 검색 공식 자체는 예선과 같은 Dense + BM25 하이브리드다. Cross-Encoder는 추가되지 않았다.

## 10. 운영 경로에서 사용하지 않는 실험 코드

`medcheck/prompts.py`에는 다음 실험용 요소가 남아 있다.

- `CANDIDATE_FILTER_SYSTEM_PROMPT`
- `EVIDENCE_REPAIR_SYSTEM_PROMPT`
- `build_candidate_filter_prompt()`
- `build_evidence_repair_prompt()`

후보 필터는 동일 사례에서 모든 후보를 `False`로 만드는 불안정성이 있었고, 근거 보정은 관련성이 낮은 후보의 필드를 억지로 채우는 문제가 있었다. 현재 `pipeline.py`는 이를 import하거나 호출하지 않는다.

운영 호출 순서는 다음과 같다.

```text
CLARIFICATION_SYSTEM_PROMPT
  → DECISION_SYSTEM_PROMPT
  → VERIFIER_SYSTEM_PROMPT(verify=True일 때)
```

실험용 프롬프트가 파일에 존재한다고 해서 실행 비용이나 추가 모델 호출이 발생하지 않는다. 다만 팀원이 혼동하지 않도록 “비활성 실험용”으로 유지하거나 추후 별도 실험 파일로 옮기는 것이 좋다.

## 11. 요구사항에서 코드까지의 추적표

| 요구사항 | 데이터·조건 | 처리 함수 | 화면 위치 |
|---|---|---|---|
| 답변 전 추가 질문 | `requires_clarification` | `assess_clarification()`, `run()` | `render_clarification_stage()` |
| 질문 우선순위 | `question.priority` | 질문 후처리, `render_clarification_hold_notice()` | 질문 배지 |
| 응급 경고 우선 | `assessment.urgent` 또는 최종 긴급도 | `should_show_action_card()` | 질문·결과 최상단 |
| 진료 이유 | `red_flags`, `urgent_message` | `extract_reasons()` | 행동 카드 ① |
| 방문 준비정보 | `missing_information`, `questions` | `extract_preparations()` | 행동 카드 ② |
| 의료진용 요약 | `structured_case` | `build_symptom_summary()`, `format_handoff_text()` | 행동 카드 ③ 및 복사 상자 |
| 구체 처방 차단 | `HIGH_RISK_PATTERNS` | `requires_forced_abstention()` | 보류 행동 카드 |
| 무효 근거 ID 제거 | 검색 결과의 ID 집합 | `_sanitize_citations()` | 근거 탭에는 정리 후 ID만 표시 |
| 서술 필드의 ID 이동 | `^E\d+$` | `_run_decision()` 내부 보정 | 직접 표시 없음 |
| 중복 진단 제거 | 정규화된 진단명 | `_run_decision()` 내부 보정 | 감별진단 카드 |
| 검증 실패 보류 | `needs_abstention` 또는 `not overall_grounded` | `_run_decision()` | 보류 행동 카드·검증 탭 |
| 환자정보 혼입 회귀 | 대표 금지 표현 | `find_patient_fact_leaks()` | 테스트·스모크 출력 |

## 12. 예선 자료의 본선 계획 대비 코드 상태

| 계획 | 코드 상태 | 근거 |
|---|---|---|
| 중복·유사 문항 및 오답 유도 사례 | 부분 반영 | `damage.py`와 대표 회귀 사례는 있으나 대규모 유사 문항 탐지는 없음 |
| Cross-Encoder 재순위화 | 미반영 | `index.py` 검색 공식은 여전히 Dense + BM25 |
| 결과를 바꿀 질문 우선 | 부분 반영 | 프롬프트와 우선순위 UI는 있으나 정량 캘리브레이션 미완료 |
| 문장·문서 구간 근거 연결 | 미반영 | 감별진단·조치 단위 `E1~Ek`만 사용 |
| 과잉 보류와 위험 응답 균형 | 부분 반영 | 보류 후처리와 회귀 검사는 있으나 임상 검증셋 기반 임계값 평가 부족 |
| 감별·안전성 정량 평가 | 부분 반영 | 3-way·질문 품질·스모크 도구는 있으나 최종 임상 성능 평가 아님 |

## 13. 팀원이 수정할 때 지켜야 할 경계

- `app.py`, `action_card.py`, `ui.py`의 표현 변경이 `ClinicalDecision` 값을 바꾸지 않도록 한다.
- 행동 카드에서 새 임상 문장을 임의로 만들지 말고 기존 구조화 필드를 사용한다.
- `supporting_evidence`와 검색 문서의 역할을 다시 섞지 않는다.
- 비강제 보류의 진단 유지와 강제 보류의 진단 삭제를 구분한다.
- 실험용 후보 필터·근거 보정을 운영 경로에 다시 연결하려면 별도 회귀 평가를 먼저 수행한다.
- `server_ready/pipeline.py`, `server_ready/prompts.py`를 기준 파일로 사용하지 않는다.
- 테스트 `PASS`를 의료적 정확성이라고 표현하지 않는다.

## 14. 다음 코드 정리 우선순위

1. `merge_clarification_answers()`의 중복 `[원래 환자 사례]` 표지 제거
2. 행동 카드에 `next_steps`를 직접 포함할지 제품 정의 확정
3. 이미 답한 정보를 다시 `missing_information`으로 생성하는 사례 회귀 테스트
4. 응급인데 `next_steps`가 비는 사례의 평가 기준 추가
5. 실험용 후보 필터·근거 보정 프롬프트를 별도 파일로 이동하거나 삭제
6. 서버 연구용 학습·평가 스크립트의 절대경로를 인자로 전환
7. `tests/test_action_card.py`를 출력식 스크립트가 아닌 단언 기반 `unittest`로 정리
8. 환자정보 혼입 검사를 고정 문자열을 넘어 일반화할 평가셋 구축

---

이 문서의 핵심 구분은 다음과 같다.

```text
예선 핵심 구조는 유지
  + 본선 피드백에 따른 행동 중심 UI 추가
  + 모델 출력 경계와 후처리 강화
  + 회귀 평가와 범용 배포 추가
  - Cross-Encoder와 구간 인용은 아직 미구현
```
