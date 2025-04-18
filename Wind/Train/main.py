# 主程序入口
import os
import pandas as pd
import joblib
from config import WIND_STATIONS, TRAIN_START_DATE, TRAIN_END_DATE, TEST_START_DATE, TEST_END_DATE, OUTPUT_PATH
from feature_engineering import create_feature_dataset
from model import train_lightgbm_model
from evaluation import evaluate_model

def main():
    """
    主函数，执行完整的训练和评估流程
    """
    print("风电场站功率预测系统 V1.3.0")
    print("=" * 50)
    
    results = []
    
    for station_id in WIND_STATIONS:
        print(f"\n开始处理风电场站 {station_id}...")
        
        # 1. 创建特征数据集
        print(f"为站点 {station_id} 创建特征数据集...")
        features_df = create_feature_dataset(station_id, TRAIN_START_DATE, TRAIN_END_DATE)
        
        if features_df is not None:
            # 2. 训练模型 - 现在训练单一模型而不是多个时间点模型
            print(f"训练站点 {station_id} 的统一模型...")
            if features_df is None or features_df.empty:
                print(f"错误: 站点 {station_id} 的特征数据集为空，无法训练模型")
                continue
            else:
                print(f"站点 {station_id} 的特征数据集形状: {features_df.shape}")
                
            model = train_lightgbm_model(features_df, station_id)
            
            if model:
                print(f"站点 {station_id} 的模型训练成功")
                # 3. 评估模型
                print(f"评估站点 {station_id} 的模型性能...")
                eval_result = evaluate_model(station_id, model, TEST_START_DATE, TEST_END_DATE)
                if eval_result:
                    print(f"站点 {station_id} 的模型评估成功: {eval_result}")
                    results.append(eval_result)
                else:
                    print(f"站点 {station_id} 的模型评估失败")
            else:
                print(f"站点 {station_id} 的模型训练失败")
    
    # 汇总结果
    if results:
        results_df = pd.DataFrame(results)
        print("\n所有风电场站评估结果:")
        print(results_df)
        
        # 计算平均准确率
        avg_accuracy = results_df['avg_daily_accuracy'].mean()
        print(f"\n所有风电场站平均准确率: {avg_accuracy:.4f}")
        
        # 保存结果
        results_path = os.path.join(OUTPUT_PATH, "wind_stations_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"结果已保存至: {results_path}")
    else:
        print("没有生成任何评估结果")

if __name__ == "__main__":
    main()