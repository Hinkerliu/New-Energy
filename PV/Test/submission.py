import os
import pandas as pd
import zipfile
from model import load_models, predict_with_lightgbm

def generate_submission_files(station_ids, prediction_dates, output_dir):
    """
    为比赛生成提交文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    all_predictions = {station_id: pd.DataFrame() for station_id in station_ids}
    models_dict = {}
    for station_id in station_ids:
        models = load_models(station_id)
        if models is not None:
            models_dict[station_id] = models
        else:
            print(f"[警告] 站点 {station_id} 没有模型，已跳过。")
    for target_date_str in prediction_dates:
        target_date = pd.to_datetime(target_date_str, format='%Y%m%d')
        prev_date = target_date - pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y%m%d')
        print(f"\n[进度] 使用{prev_date_str}气象数据预测{target_date_str}发电量")
        for station_id in station_ids:
            if station_id not in models_dict:
                continue
            pred_df = predict_with_lightgbm(station_id, prev_date_str, models_dict[station_id])
            if pred_df is None:
                print(f"[警告] 站点 {station_id} 的 {target_date_str} 预测失败，已跳过。")
                continue
            all_predictions[station_id] = pd.concat([all_predictions[station_id], pred_df])
    for station_id, pred_df in all_predictions.items():
        if not pred_df.empty:
            output_file = os.path.join(output_dir, f"output{station_id}.csv")
            pred_df.to_csv(output_file)
            print(f"[信息] 站点 {station_id} 的所有预测结果已保存至: {output_file}")
    zip_file = os.path.join(output_dir, "output.zip")
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        for station_id in station_ids:
            csv_file = os.path.join(output_dir, f"output{station_id}.csv")
            if os.path.exists(csv_file):
                zipf.write(csv_file, os.path.basename(csv_file))
    print(f"[信息] 所有提交文件已压缩至: {zip_file}")