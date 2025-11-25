import os
import pandas as pd

# 1) 경로 설정
input_path = r"path"
output_dir = r"path"
os.makedirs(output_dir, exist_ok=True)

# 2) CSV 로드
df = pd.read_csv(input_path)

# 3) 원래 컬럼명을 module_raw로 변경
df.rename(columns={'module(equipment)': 'module_raw'}, inplace=True)

# 4) module_raw에서 숫자 ID만 뽑아서 정수형 module_id 컬럼 생성
#    E.g.) "11(우측분전반1)" → 11
df['module_id'] = df['module_raw'].str.extract(r'^(\d+)').astype(int)

# 5) 모듈 번호 → 한글 이름 매핑
module_names = {
    1:  "PM-3",
    2:  "L-1전등",
    3:  "분쇄기(2)",
    4:  "분쇄기(1)",
    5:  "좌측분전반",
    11: "우측분전반1",
    12: "4호기",
    13: "3호기",
    14: "2호기",
    15: "예비건조기",
    16: "호이스트",
    17: "6호기",
    18: "우측분전반2"
}

#모듈별 따로 저장
for mod_id, group in df.groupby('module_id'):
    name     = module_names.get(mod_id, str(mod_id))
    filename = f"{mod_id}_{name}.csv"
    outpath  = os.path.join(output_dir, filename)
    group.to_csv(outpath, index=False, encoding='utf-8-sig')
    print(f"[저장완료] {filename}")
