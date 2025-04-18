"""
提交文件生成模块 - 包含生成比赛提交文件的函数
"""
import os
import pandas as pd
from model import load_models, predict_with_lightgbm
from config import OUTPUT_PATH

def generate_submission_files(station_ids, prediction_dates, output_dir=OUTPUT_PATH):
    """
    为比赛生成提交文件
    
    参数:
    - station_ids: 站点ID列表
    - prediction_dates: 预测日期列表，格式YYYYMMDD（这是要预测的日期）
    - output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个站点创建一个空的DataFrame，用于存储所有日期的预测结果
    all_predictions = {station_id: pd.DataFrame() for station_id in station_ids}
    
    # 加载每个站点的模型（只需加载一次）
    models_dict = {}
    for station_id in station_ids:
        model = load_models(station_id)
        if model is not None:
            models_dict[station_id] = model
        else:
            print(f"[错误] 站点 {station_id} 没有可用模型，跳过")
    
    # 处理每一天的预测
    for target_date_str in prediction_dates:
        # 将目标日期转换为datetime对象
        target_date = pd.to_datetime(target_date_str, format='%Y%m%d')
        
        # 计算前一天的日期（用于获取气象数据）
        prev_date = target_date - pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y%m%d')
        
        print(f"\n[进度] 使用 {prev_date_str} 气象数据预测 {target_date_str} 发电量")
        
        for station_id in station_ids:
            if station_id not in models_dict:
                continue
                
            # 使用前一天的气象数据预测目标日期的发电量
            pred_df = predict_with_lightgbm(station_id, prev_date_str, models_dict[station_id])
            if pred_df is None:
                print(f"[警告] 站点 {station_id} 的 {target_date_str} 预测失败，跳过")
                continue
            
            # 将当天的预测结果添加到总预测结果中
            all_predictions[station_id] = pd.concat([all_predictions[station_id], pred_df])
            print(f"[完成] 站点 {station_id} 的 {target_date_str} 预测成功")
    
    # 保存每个站点的所有预测结果
    output_files = []
    for station_id, pred_df in all_predictions.items():
        if not pred_df.empty:
            output_file = os.path.join(output_dir, f"output{station_id}.csv")
            # 确保输出格式正确
            output_df = pd.DataFrame({
                'time': pred_df.index.strftime('%Y-%m-%d %H:%M:%S'),
                'power': pred_df['power'].values
            })
            output_df.to_csv(output_file, index=False)
            print(f"[保存] 站点 {station_id} 的所有预测结果已保存至: {output_file}")
            output_files.append(output_file)
    
    print(f"[完成] 所有提交文件已保存至目录: {output_dir}")
    return output_dir