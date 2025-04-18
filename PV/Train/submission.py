import os

# 修改生成提交文件的函数，使其适应统一模型
def generate_submission_files(station_ids, day_str, models_dict, output_dir, predict_func, get_meteo_for_day_func):
    """生成提交文件"""
    import os
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    for station_id in station_ids:
        if station_id not in models_dict:
            print(f"站点 {station_id} 没有训练好的模型，跳过")
            continue
        
        model = models_dict[station_id]
        pred_df = predict_func(station_id, day_str, model, get_meteo_for_day_func)
        
        if pred_df is None:
            print(f"站点 {station_id} 的 {day_str} 无法生成预测")
            continue
        
        # 格式化输出
        output_df = pd.DataFrame({
            'time': pred_df.index.strftime('%Y-%m-%d %H:%M:%S'),
            'power': pred_df['power'].values
        })
        
        output_file = os.path.join(output_dir, f"{station_id}.csv")
        output_df.to_csv(output_file, index=False)
        print(f"站点 {station_id} 的预测结果已保存至: {output_file}")
    print("所有提交文件已生成")