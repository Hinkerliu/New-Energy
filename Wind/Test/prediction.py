"""
预测模块 - 负责模型预测功能
"""
import os
import pandas as pd
import numpy as np
import joblib
from config import MODEL_PATH, OUTPUT_PATH
from data_loader import get_meteo_for_day
from feature_engineering import add_time_features, ensure_numeric_features, prepare_features_for_model

def predict_with_lightgbm(station_id, day_str, models):
    """
    使用训练好的LightGBM模型进行预测
    
    参数:
        station_id (int): 站点ID
        day_str (str): 日期字符串，格式 YYYYMMDD
        models (dict): 模型字典，键为时间点索引，值为模型对象
        
    返回:
        pandas.DataFrame 或 None: 预测结果数据框，预测失败则返回None
    """
    # 日期转换
    date = pd.to_datetime(day_str)
    next_day = date + pd.Timedelta(days=1)
    
    # 获取气象数据
    meteo_df = get_meteo_for_day(station_id, day_str)
    if meteo_df.empty:
        print(f"警告: 站点{station_id}的{day_str}气象数据为空，无法预测")
        return None
    
    # 创建预测时间点
    prediction_times = pd.date_range(start=next_day, periods=96, freq='15min')
    predictions = []
    
    # 为每个15分钟时间点预测
    for i, time_point in enumerate(prediction_times):
        hour = time_point.hour
        
        # 获取对应小时的气象数据
        if hour in meteo_df.index:
            hour_meteo = meteo_df.loc[hour].copy()
        else:
            # 如果没有精确匹配的小时，找最近的
            closest_hour = min(meteo_df.index, key=lambda x: abs(x - hour))
            hour_meteo = meteo_df.loc[closest_hour].copy()
        
        # 添加时间特征
        hour_meteo = add_time_features(hour_meteo, time_point)
        
        # 转换为DataFrame
        features = pd.DataFrame([hour_meteo])
        
        # 确保所有特征都是数值类型
        features = ensure_numeric_features(features)
        
        # 使用对应时间点的模型预测
        if i in models:
            model = models[i]
        else:
            # 如果没有对应时间点的模型，使用最近的模型
            closest_idx = min(models.keys(), key=lambda x: abs(x - i))
            model = models[closest_idx]
        
        # 获取模型使用的特征名称
        model_features = model.feature_name()
        
        # 准备特征
        features_filtered = prepare_features_for_model(features, model_features)
        
        # 预测
        pred = model.predict(features_filtered)[0]
        
        # 确保预测值在[0,1]范围内
        pred = max(0, min(1, pred))
        
        predictions.append(pred)
    
    # 创建预测结果DataFrame
    result_df = pd.DataFrame({
        'time': prediction_times,
        'power': predictions
    })
    result_df = result_df.set_index('time')
    
    return result_df

def load_model(station_id):
    """
    加载指定站点的模型
    
    参数:
        station_id (int): 站点ID
        
    返回:
        dict 或 None: 模型字典，加载失败则返回None
    """
    model_file = os.path.join(MODEL_PATH, f"lightgbm_station_{station_id}.pkl")
    if not os.path.exists(model_file):
        print(f"错误: 站点{station_id}的模型文件不存在: {model_file}")
        return None
    
    try:
        models = joblib.load(model_file)
        print(f"信息: 成功加载站点{station_id}的模型")
        return models
    except Exception as e:
        print(f"错误: 加载站点{station_id}的模型时出错: {e}")
        return None

def save_predictions(station_id, predictions_df):
    """
    保存预测结果
    
    参数:
        station_id (int): 站点ID
        predictions_df (pandas.DataFrame): 预测结果数据框
        
    返回:
        str: 输出文件路径
    """
    # 使用配置文件中定义的OUTPUT_PATH保存输出文件
    os.makedirs(OUTPUT_PATH, exist_ok=True)  # 确保输出目录存在
    output_file = os.path.join(OUTPUT_PATH, f"output{station_id}.csv")
    predictions_df.to_csv(output_file)
    return output_file