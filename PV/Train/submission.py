import os

def generate_submission_files(station_ids, prediction_date, models_dict, output_dir, predict_func, get_meteo_for_day_func):
    """为比赛生成提交文件"""
    os.makedirs(output_dir, exist_ok=True)
    import pandas as pd
    date = pd.to_datetime(prediction_date)
    next_day = date + pd.Timedelta(days=1)
    for station_id in station_ids:
        if station_id not in models_dict:
            print(f"站点 {station_id} 没有模型，跳过")
            continue
        pred_df = predict_func(station_id, prediction_date, models_dict[station_id], get_meteo_for_day_func)
        if pred_df is None:
            print(f"站点 {station_id} 的预测失败，跳过")
            continue
        output_file = os.path.join(output_dir, f"output{station_id}.csv")
        pred_df.to_csv(output_file)
        print(f"站点 {station_id} 的预测结果已保存至: {output_file}")
    print("所有提交文件已生成")