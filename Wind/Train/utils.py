# 工具函数模块
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from config import OUTPUT_PATH

def save_model(model, station_id, model_name="lightgbm"):
    """
    保存模型到指定路径
    
    参数:
    - model: 要保存的模型
    - station_id: 站点ID
    - model_name: 模型名称
    
    返回:
    - 保存路径
    """
    model_path = os.path.join(OUTPUT_PATH, f"{model_name}_station_{station_id}.pkl")
    joblib.dump(model, model_path)
    print(f"模型已保存至: {model_path}")
    return model_path

def load_model(station_id, model_name="lightgbm"):
    """
    从指定路径加载模型
    
    参数:
    - station_id: 站点ID
    - model_name: 模型名称
    
    返回:
    - 加载的模型
    """
    model_path = os.path.join(OUTPUT_PATH, f"{model_name}_station_{station_id}.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"已加载模型: {model_path}")
        return model
    else:
        print(f"模型文件不存在: {model_path}")
        return None

def plot_feature_importance(feature_importances, station_id, top_n=20):
    """
    绘制特征重要性图
    
    参数:
    - feature_importances: 特征重要性DataFrame
    - station_id: 站点ID
    - top_n: 显示前N个重要特征
    """
    if feature_importances.empty:
        print("特征重要性数据为空")
        return
    
    # 计算每个特征的平均重要性
    avg_importance = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    avg_importance.head(top_n).plot(kind='barh')
    plt.title(f'站点 {station_id} 的特征重要性 (Top {top_n})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"feature_importance_station_{station_id}.png"))
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, station_id, date_str, time_points=None):
    """
    绘制预测值与实际值对比图
    
    参数:
    - y_true: 实际值数组
    - y_pred: 预测值数组
    - station_id: 站点ID
    - date_str: 日期字符串
    - time_points: 时间点列表
    """
    plt.figure(figsize=(15, 6))
    
    if time_points is None:
        # 如果没有提供时间点，使用索引
        plt.plot(y_true, 'b-', label='实际功率')
        plt.plot(y_pred, 'r-', label='预测功率')
    else:
        plt.plot(time_points, y_true, 'b-', label='实际功率')
        plt.plot(time_points, y_pred, 'r-', label='预测功率')
    
    plt.title(f'站点 {station_id} - {date_str} 功率预测对比')
    plt.xlabel('时间')
    plt.ylabel('归一化功率')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_PATH, f"prediction_{station_id}_{date_str}.png"))
    plt.close()

def plot_scatter(y_true, y_pred, station_id):
    """
    绘制预测值与实际值散点图
    
    参数:
    - y_true: 实际值数组
    - y_pred: 预测值数组
    - station_id: 站点ID
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('实际功率')
    plt.ylabel('预测功率')
    plt.title(f'站点 {station_id} 预测vs实际散点图')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_PATH, f"scatter_plot_{station_id}.png"))
    plt.close()