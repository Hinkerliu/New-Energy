"""
主程序模块 - 程序入口
"""
import os
import pandas as pd
from config import WIND_STATIONS, OUTPUT_PATH
from data_loader import get_test_dates
from prediction import load_model, predict_with_lightgbm, save_predictions
from utils import print_info, print_warning, print_error

def run_inference():
    """
    运行推理过程，为所有风电站点生成预测结果
    """
    print_info("开始风电场发电功率预测...")
    
    # 使用相对路径设置输出目录，修改为指向正确的output目录
    relative_output_path = os.path.join("..", "..", "output")
    # 确保输出目录存在
    os.makedirs(relative_output_path, exist_ok=True)
    
    for station_id in WIND_STATIONS:
        print_info(f"开始处理风电场站{station_id}的推理...")
        
        # 加载模型
        models = load_model(station_id)
        if models is None:
            continue
        
        # 获取测试日期
        test_dates = get_test_dates(station_id)
        if not test_dates:
            print_warning(f"站点{station_id}没有找到测试日期")
            continue
        
        print_info(f"找到{len(test_dates)}个测试日期")
        
        # 为每个测试日期生成预测
        all_predictions = []
        
        for day_str in test_dates:
            print_info(f"预测站点{station_id}的日期{day_str}...")
            
            # 预测
            pred_df = predict_with_lightgbm(station_id, day_str, models)
            if pred_df is not None:
                # 添加到预测列表
                all_predictions.append(pred_df)
        
        # 合并所有预测结果
        if all_predictions:
            final_predictions = pd.concat(all_predictions)
            
            # 保存预测结果，使用相对路径
            output_file = os.path.join(relative_output_path, f"output{station_id}.csv")
            final_predictions.to_csv(output_file)
            print_info(f"站点{station_id}的预测结果已保存至: {os.path.abspath(output_file)}")
        else:
            print_warning(f"站点{station_id}没有生成任何预测结果")
    
    print_info("风电场发电功率预测完成")

if __name__ == "__main__":
    run_inference()