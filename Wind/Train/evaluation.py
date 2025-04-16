# 模型评估模块
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import OUTPUT_PATH
from data_loader import load_power_data
from model import predict_with_lightgbm

def evaluate_model(station_id, models, test_start_date, test_end_date):
    """
    评估模型在测试集上的性能
    
    参数:
    - station_id: 站点ID
    - models: 训练好的模型字典
    - test_start_date: 测试开始日期
    - test_end_date: 测试结束日期
    
    返回:
    - 评估结果字典
    """
    # 转换日期为datetime对象
    start_dt = pd.to_datetime(test_start_date)
    end_dt = pd.to_datetime(test_end_date)
    
    # 创建日期范围 - 注意这里需要减去一天，因为气象数据是预报次日的
    date_range = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(days=1), freq='D')
    
    # 加载功率数据（真实值）
    power_df = load_power_data(station_id)
    if power_df is None:
        print(f"无法加载站点 {station_id} 的功率数据")
        return None
    
    all_predictions = []
    all_actuals = []
    daily_scores = []
    
    # 处理每一天
    for date in date_range:
        day_str = date.strftime('%Y%m%d')
        next_day = date + pd.Timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        
        # 预测
        pred_df = predict_with_lightgbm(station_id, day_str, models)
        if pred_df is None:
            continue
        
        # 获取真实值
        actual_df = power_df.loc[next_day_str:next_day_str]
        if actual_df.empty:
            continue
        
        # 确保真实值有96个点
        expected_times = pd.date_range(start=next_day, periods=96, freq='15min')
        if len(actual_df) != 96:
            temp_df = pd.DataFrame
            temp_df = temp_df.join(actual_df)
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            actual_df = temp_df
        
        # 计算日准确率
        y_pred = pred_df['power'].values
        y_true = actual_df['power'].values
        
        # 计算1-MAE作为准确率
        daily_mae = mean_absolute_error(y_true, y_pred)
        daily_accuracy = 1 - daily_mae
        daily_scores.append(daily_accuracy)
        
        # 存储预测和真实值
        all_predictions.extend(y_pred)
        all_actuals.extend(y_true)
        
        # 每10天绘制一次对比图
        if len(daily_scores) % 10 == 0:
            plt.figure(figsize=(15, 6))
            plt.plot(expected_times, y_true, 'b-', label='实际功率')
            plt.plot(expected_times, y_pred, 'r-', label='预测功率')
            plt.title(f'站点 {station_id} - {next_day_str} 功率预测对比')
            plt.xlabel('时间')
            plt.ylabel('归一化功率')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_PATH, f"prediction_{station_id}_{next_day_str}.png"))
            plt.close()
    
    # 计算总体评估指标
    if all_predictions and all_actuals:
        mae = mean_absolute_error(all_actuals, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        r2 = r2_score(all_actuals, all_predictions)
        
        # 计算平均日准确率
        avg_daily_accuracy = np.mean(daily_scores)
        
        print(f"\n站点 {station_id} 评估结果:")
        print(f"平均日准确率: {avg_daily_accuracy:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 绘制散点图
        plt.figure(figsize=(10, 10))
        plt.scatter(all_actuals, all_predictions, alpha=0.3)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('实际功率')
        plt.ylabel('预测功率')
        plt.title(f'站点 {station_id} 预测vs实际散点图')
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_PATH, f"scatter_plot_{station_id}.png"))
        plt.close()
        
        return {
            'station_id': station_id,
            'avg_daily_accuracy': avg_daily_accuracy,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    else:
        print(f"站点 {station_id} 没有足够的数据进行评估")
        return None