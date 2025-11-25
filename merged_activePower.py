import os
import pandas as pd

folder = r"path"
dfs = []

for root, _, files in os.walk(folder):
    for fname in files:
        if not fname.lower().endswith('.csv'): 
            continue
            path = os.path.join(root, fname)
        df = pd.read_csv(path, usecols=['localtime','activepower'])
        dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# ← 여기서 infer_datetime_format=True 로 자동 감지
all_df['localtime'] = pd.to_datetime(
    all_df['localtime'], 
    infer_datetime_format=True, 
    errors='raise'         # 파싱 불가 시 에러, 필요시 'coerce'로 NaT 처리
)

# 같은 타임스탬프끼리 activepower 합산
result = (
    all_df
    .groupby('localtime', as_index=False)['activepower']
    .sum()
    .rename(columns={'activepower':'sum_activepower'})
)

print(result.head())
result.to_csv(
    r"path",
    index=False
)
