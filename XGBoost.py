import os
import pandas as pd
import numpy as np
import xgboost as xgb
# 1. 입력/출력 디렉토리 설정
input_dir  = r"path"
output_dir = os.path.join(input_dir, "dataReplace")
os.makedirs(output_dir, exist_ok=True)
# 2. 사용할 피처 컬럼 정의 (숫자형만)
feature_cols = [
    'voltager', 'voltages', 'voltaget',
    'currentr', 'currents', 'currentt',
    'powerfactorr', 'powerfactors', 'powerfactort'
]
# 3. XGBoost GPU 회귀 모델 초기화
model = xgb.XGBRegressor(
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    n_estimators=100,
    random_state=42,
    verbosity=0
)
# 4. 폴더 내 모든 CSV 순회
for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".csv"):
        continue
    input_path  = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)
    print(f"\n🔹 처리 시작: {fname}")
    # --- 데이터 로드 및 숫자형 컬럼만 남기기 ---
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=feature_cols + ['activepower'])
    print(f"  • 로드 완료, 행 수: {len(df)}")

    # --- 1단계: 물리 기반 이상치 탐지 ---
    v_avg  = df[['voltager','voltages','voltaget']].mean(axis=1)
    i_avg  = df[['currentr','currents','currentt']].mean(axis=1)
    pf_avg = df[['powerfactorr','powerfactors','powerfactort']].mean(axis=1) / 100
    p_pred    = np.sqrt(3) * v_avg * i_avg * pf_avg / 1000
    residual  = np.abs(df['activepower'] - p_pred)
    thr       = residual.mean() + 3 * residual.std()
    df['is_outlier'] = residual > thr
    n_outliers = df['is_outlier'].sum()
    print(f"  • 이상치 탐지: {n_outliers}건 (threshold={thr:.4f})")

    # --- 2단계: XGBoost 회귀 학습 및 보정 ---
    # 2-1. 학습용 데이터 준비 (이상치 제외)
    train = df[~df['is_outlier']]
    X_train = train[feature_cols]
    y_train = train['activepower']
    print(f"  • 모델 학습 데이터: {len(X_train)}개")

    model.fit(X_train, y_train)
    print("  • XGBoost 회귀 모델 학습 완료")

    # 2-2. 이상치 위치만 예측하여 보정
    df.loc[df['is_outlier'], 'activepower'] = (
        model.predict(df.loc[df['is_outlier'], feature_cols])
    )
    df.drop(columns=['is_outlier'], inplace=True)
    print("  • 이상치 보정 완료")

    df.to_csv(output_path, index=False)
    print(f"저장 완료: {output_path}")

print("\n 처리 끝 ")
