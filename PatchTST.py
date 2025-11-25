import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
from neuralforecast.core import NeuralForecast
from neuralforecast.models import PatchTST
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('medium')

def process_file(train_file_path: str, val_file_path: str, output_path: str):
    module_id = os.path.splitext(os.path.basename(train_file_path))[0]

    # 1) 훈련 데이터: 1분 리샘플링
    raw_train = pd.read_csv(train_file_path, parse_dates=['localtime'])
    raw_train.columns = raw_train.columns.str.lower()
    raw_train.set_index('localtime', inplace=True)
    df_1min = (
        raw_train[['activepower']]
        .resample('1min')
        .mean()
        .reset_index()
        .rename(columns={'localtime':'ds','activepower':'y'})
    )
    df_1min['unique_id'] = module_id

    # 2) 검증 데이터: 1분 리샘플링
    raw_val = pd.read_csv(val_file_path, parse_dates=['localtime'])
    raw_val.columns = raw_val.columns.str.lower()
    raw_val.set_index('localtime', inplace=True)
    df_val = (
        raw_val[['activepower']]
        .resample('1min')
        .mean()
        .reset_index()
        .rename(columns={'localtime':'ds','activepower':'y'})
    )
    df_val['unique_id'] = module_id

    # 3) 훈련 + 검증 데이터 결합
    df = pd.concat([df_1min, df_val], ignore_index=True)
    val_size = len(df_val)  # 검증 데이터 크기

    # 4) Callback & Logger 설정
    callbacks = [
        EarlyStopping(monitor='valid_loss', patience=5, mode='min'),
        ModelCheckpoint(
            monitor='valid_loss',
            save_top_k=1,
            mode='min',
            filename=f'{module_id}' + '-{epoch}'
        )
    ]
    logger = CSVLogger(
        save_dir=os.path.join(output_path, 'logs'),
        name=module_id
    )
    device_available = torch.cuda.is_available()

    trainer_kwargs = {
        'accelerator': 'gpu',
        'devices': 1,
        'enable_checkpointing': True,
        'callbacks': callbacks,
        'logger': logger
    } if device_available else {
        'accelerator': 'cpu',
        'enable_checkpointing': True,
        'callbacks': callbacks,
        'logger': logger
    }

    # 5) PatchTST 모델 정의
    horizon = 720  # 12시간 예측
    model = PatchTST(
        h=horizon,
        input_size=180,  # 3시간 입력
        patch_len=8,
        stride=4,
        encoder_layers=1,
        n_heads=2,
        hidden_size=16,
        linear_hidden_size=32,
        dropout=0.2,
        fc_dropout=0.2,
        attn_dropout=0.2,
        head_dropout=0.2,
        learning_rate=1e-3,
        max_steps=2500,
        batch_size=2,
        dataloader_kwargs={'num_workers':0,'pin_memory':False},
        **trainer_kwargs
    )

    # 6) 학습 + 검증
    if device_available:
        torch.cuda.empty_cache()
    nf = NeuralForecast(models=[model], freq='1min')
    nf.fit(df=df, val_size=val_size)  # 결합된 데이터와 val_size 사용

    # 7) 손실 곡선 저장
    metrics_csv = os.path.join(logger.log_dir, 'metrics.csv')
    if os.path.exists(metrics_csv):
        metrics = pd.read_csv(metrics_csv)
        plt.figure(figsize=(6,4))
        plt.plot(metrics['step'], metrics['train_loss'], label='train_loss')
        plt.plot(metrics['step'], metrics['val_loss'], label='val_loss')
        plt.xlabel('step'); plt.ylabel('loss'); plt.legend()
        img_path = os.path.join(output_path, f"{module_id}_loss_curve.png")
        plt.savefig(img_path, bbox_inches='tight'); plt.close()
        print(f"[{module_id}] 손실 곡선 저장 → {img_path}")

    # 8) 반복 예측
    start_date = pd.Timestamp("2025-05-01 00:00:00")
    end_date = pd.Timestamp("2025-05-29 23:59:00")
    total_mins = int((end_date - start_date) / pd.Timedelta(minutes=1)) + 1
    n_iter = (total_mins + horizon - 1) // horizon

    current_df = df_1min[df_1min['ds'] < start_date].copy()
    all_preds = []
    for i in range(n_iter):
        iter_time = start_date + pd.Timedelta(minutes=i*horizon)
        print(f"[{module_id}] 반복 {i+1}/{n_iter} @ {iter_time}")
        pred = nf.predict(df=current_df).reset_index()
        pred['module_raw'] = module_id
        pred = pred.rename(columns={'ds':'localtime','PatchTST':'activepower'})
        all_preds.append(pred[['module_raw','localtime','activepower']])
        tail = pred[['localtime','activepower']].rename(
            columns={'localtime':'ds','activepower':'y'}
        ); tail['unique_id'] = module_id
        current_df = pd.concat([current_df, tail], ignore_index=True)
        if device_available: torch.cuda.empty_cache()

    # 9) 결과 저장
    os.makedirs(output_path, exist_ok=True)
    out_csv = os.path.join(output_path, f'{module_id}_forecast.csv')
    final = pd.concat(all_preds, ignore_index=True)
    final = final[
        (final['localtime'] >= start_date) &
        (final['localtime'] <= end_date)
    ]
    final.to_csv(out_csv, index=False)
    print(f"[{module_id}] 예측 결과 저장 → {out_csv}")

def main():
    input_dir_train = r"path"
    input_dir_val = r"path"
    output_dir = r"path"

    # 공통 파일 이름 찾기
    train_files = {os.path.splitext(f)[0] for f in os.listdir(input_dir_train) if f.lower().endswith('.csv')}
    val_files = {os.path.splitext(f)[0] for f in os.listdir(input_dir_val) if f.lower().endswith('.csv')}
    common_files = train_files.intersection(val_files)

    # 공통 파일 처리
    for fname in common_files:
        print(f"처리 중: {fname}.csv")
        train_path = os.path.join(input_dir_train, f"{fname}.csv")
        val_path = os.path.join(input_dir_val, f"{fname}.csv")
        process_file(train_path, val_path, output_dir)

if __name__ == "__main__":
    freeze_support()
    main()
