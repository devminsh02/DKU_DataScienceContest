import pandas as pd

# 1) 1분 단위 합계 CSV 읽기 (localtime을 datetime으로 파싱)
df = pd.read_csv(
    r"path",
    parse_dates=['localtime']
)

# 2) 인덱스를 localtime으로 설정
df.set_index('localtime', inplace=True)

# 3) 1시간 단위로 sum_activepower 합산
hourly = (
    df['sum_activepower']
    .resample('1H')
    .sum()
    .rename('hourly_activepower')
)

# 4) 결과를 CSV로 저장
hourly.reset_index().to_csv(
    r"path",
    index=False
)
