<h1>🌟 DKU Data Science Contest 2025 </h1>
<h3>전력 사용량 예측 모델 (우수상 수상 프로젝트)</h3> 


2025 단국대학교 데이터 사이언스 경진대회에서
우수상을 수상한 전력 수요 예측 시스템 입니다.

대회 규정상 데이터는 비공개이며 코드의 경로는 모두 예시로 작성되어 있습니다.


<h2>📁 Dataset Overview</h2>

원본 데이터는 총 18개 Module × 전력/전류/전압 관련 12개 이상의 변수로 구성된 고해상도 시계열입니다.
| Category       | Columns                                                       |
| -------------- | ------------------------------------------------------------- |
| Identification | module(equipment)                                             |
| Timestamp      | timestamp, localtime                                          |
| Voltage        | voltageR, voltageS, voltageT, voltageRS, voltageST, voltageTR |
| Current        | currentR, currentS, currentT                                  |
| Power          | ActivePower, reactivePowerLagging                             |
| Power Factor   | PowerFactorR, PowerFactorS, PowerFactorT                      |
| Energy         | accumActiveEnergy                                             |
| Operation flag | operation                                                     |



목표:
2024-12 ~ 2025-04 데이터 기반으로 → 2025년 5월 ActivePower 1분 단위 예측

<h3>🔧 Project Pipeline Overview</h3>

본 저장소에는 다음 5개의 모듈이 포함됩니다:
| Step          | File                           | 역할                              |
| ------------- | ------------------------------ | ------------------------------- |
| 1️⃣ Module 분리 | `separate.py`                  | 전체 원본 CSV → Module별 개별 CSV 생성   |
| 2️⃣ 노이즈 제거    | `XGBoost.py`                   | 물리 기반 이상치 검출 + XGBoost 보정       |
| 3️⃣ 1분 단위 예측  | `PatchTST.py`                  | PatchTST 기반 Rolling Forecasting |
| 4️⃣ Module 합산 | `merged_activePower.py`        | 18개 Module의 ActivePower 합산      |
| 5️⃣ 제출 형식 변환  | `merged_activePower_1hour.py`  | 1시간 단위 제출포맷 생성                  |



<h2>🧩 1. Module Separation (separate.py)</h2>

✔ 공급된 원본 CSV에서 module_raw 컬럼을 기준으로 모듈별 파일로 자동 분리

✔ 이름 매핑 테이블을 기반으로 파일명도 자동 생성


<h2>🧩 2. Noise Reduction (XGBoost.py)</h2>

물리 기반 전력 모델 + ML 기반 보정을 결합한 하이브리드 노이즈 제거 모델 구현 부분

✔ 2.1 물리 기반 이상치 탐지

삼상 교류 전력 공식:P=3^(1/2)​×V×I×PF


✔ 2.2 XGBoost 보정

이상치만 XGBoost 기반으로 다시 예측해 대체하였습니다.


<h2>🧩 3. Forecasting with PatchTST (PatchTST.py)</h2>

✔ 입력 & 출력 구조

| 항목               | 값                   |
| ---------------- | ------------------- |
| Input window     | **180분 (3시간)**      |
| Forecast horizon | **720분 (12시간)**     |
| Resolution       | 1-minute            |
| Method           | Rolling Forecasting |

✔ Rolling Forecasting

5월 1일부터 29일까지의 전체 1분 단위 예측을
horizon만큼 반복적으로 예측해 이어 붙였습니다.


<h2>🧩 4. Module Aggregation (merged_activePower.py)</h2>

모듈별 분리된 예측 결과를 다시 하나로 합쳐
같은 시각의 ActivePower들을 모두 합산했습니다.


<h2>🧩 5. Submission Format (merged_activePower_1hour.py)</h2>


대회의 요구 형식대로
1분 단위를 → 1시간 단위로 리샘플링(sum) 하여 최종 제출 CSV를 생성했습니다.
