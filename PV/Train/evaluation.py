import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from .config import OUTPUT_PATH
from .data_utils import load_power_data

def evaluate_model(station_id, models, test_start_date, test_end_date, predict_func, get_meteo_for_day_func):
    """评估模型在测试集上的性能"""
    start_dt = pd.to_datetime(test_start_date)
    end_dt = pd.to_datetime(test_end_date)
    date_range = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(days=1), freq='D')
    power_df = load_power_data(station_id)
    if power_df is None:
        print(f"无法加载站点 {station_id} 的功率数据")
        return None
    all_predictions = []
    all_actuals = []
    daily_scores = []
    for date in date_range:
        day_str = date.strftime('%Y%m%d')
        next_day = date + pd.Timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        pred_df = predict_func(station_id, day_str, models, get_meteo_for_day_func)
        if pred_df is None:
            continue
        actual_df = power_df.loc[next_day_str:next_day_str]
        if actual_df.empty:
            continue
        expected_times = pd.date_range(start=next_day, periods=96, freq='15min')
        if len(actual_df) != 96:
            temp_df = pd.DataFrame(index=expected_times)
            temp_df = temp_df.join(actual_df)
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            actual_df = temp_df
        y_pred = pred_df['power'].values
        y_true = actual_df['power'].values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(mask):
            print(f"警告: 站点 {station_id} 的 {next_day_str} 所有数据都是NaN，跳过")
            continue
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        if len(y_pred_clean) == 0 or len(y_true_clean) == 0:
            print(f"警告: 站点 {station_id} 的 {next_day_str} 过滤NaN后没有数据，跳过")
            continue
        daily_mae = mean_absolute_error(y_true_clean, y_pred_clean)
        daily_accuracy = 1 - daily_mae
        daily_scores.append(daily_accuracy)
        all_predictions.extend(y_pred_clean)
        all_actuals.extend(y_true_clean)
        if len(daily_scores) % 10 == 0:
            plt.figure(figsize=(15, 6))
            pd.Series(y_true, index=expected_times).plot(label='实际功率', color='blue')
            pd.Series(y_pred, index=expected_times).plot(label='预测功率', color='red')
            plt.title(f'站点 {station_id} - {next_day_str} 功率预测对比')
            plt.xlabel('时间')
            plt.ylabel('归一化功率')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_PATH, f"prediction_{station_id}_{next_day_str}.png"))
            plt.close()
    if all_predictions and all_actuals:
        mae = mean_absolute_error(all_actuals, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        r2 = r2_score(all_actuals, all_predictions)
        avg_daily_accuracy = np.mean(daily_scores)
        print(f"\n站点 {station_id} 评估结果:")
        print(f"平均日准确率: {avg_daily_accuracy:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
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