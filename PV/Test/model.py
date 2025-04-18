"""
模型模块 - 包含模型加载和预测相关函数
"""
import os
import pandas as pd
import numpy as np
import joblib
from data_utils import get_meteo_for_day
from config import MODELS_PATH, MODEL_FILENAME_FORMATS, PREDICTION_PERIODS, DAYLIGHT_START_HOUR, DAYLIGHT_END_HOUR

def load_models(station_id):
    """
    加载指定站点的LightGBM模型
    
    参数:
    - station_id: 站点ID
    
    返回:
    - 加载的模型对象，如果加载失败则返回None
    """
    # 尝试不同的模型文件名格式
    for format_str in MODEL_FILENAME_FORMATS:
        model_path = os.path.join(MODELS_PATH, format_str.format(station_id))
        try:
            model = joblib.load(model_path)
            print(f"[成功] 加载站点 {station_id} 的模型: {model_path}")
            return model
        except Exception as e:
            print(f"[尝试] 加载模型失败: {model_path}, 尝试下一个格式")
    
    print(f"[错误] 站点 {station_id} 的所有模型格式均加载失败")
    return None

def predict_with_lightgbm(station_id, day_str, model):
    """
    使用训练好的LightGBM模型进行预测
    
    参数:
    - station_id: 站点ID
    - day_str: 日期字符串，格式 YYYYMMDD
    - model: 加载的模型对象
    
    返回:
    - 预测结果 DataFrame，如果预测失败则返回None
    """
    # 日期转换
    date = pd.to_datetime(day_str, format='%Y%m%d')
    next_day = date + pd.Timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    
    # 获取气象数据
    meteo_df = get_meteo_for_day(station_id, day_str)
    if meteo_df.empty:
        print(f"[警告] 站点 {station_id} 的 {day_str} 气象数据为空，无法预测")
        return None
    
    # 创建预测时间点
    prediction_times = pd.date_range(start=next_day, periods=PREDICTION_PERIODS, freq='15min')
    predictions = []
    
    # 为每个15分钟时间点预测
    for i, time_point in enumerate(prediction_times):
        hour = time_point.hour
        minute = time_point.minute
        
        # 获取对应小时的气象数据
        if hour in meteo_df.index:
            hour_meteo = meteo_df.loc[hour].copy()
        else:
            # 如果没有精确匹配的小时，找最近的
            closest_hour = min(meteo_df.index, key=lambda x: abs(x - hour))
            hour_meteo = meteo_df.loc[closest_hour].copy()
        
        # 添加时间特征
        hour_meteo['hour'] = float(hour)
        hour_meteo['minute'] = float(minute)
        
        # 添加日期特征
        hour_meteo['day_of_week'] = float(time_point.dayofweek)
        hour_meteo['day_of_year'] = float(time_point.dayofyear)
        hour_meteo['month'] = float(time_point.month)
        hour_meteo['is_weekend'] = float(1 if time_point.dayofweek >= 5 else 0)
        
        # 添加周期性特征
        hour_meteo['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        hour_meteo['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        hour_meteo['sin_minute'] = np.sin(2 * np.pi * (hour * 60 + minute) / (24 * 60))
        hour_meteo['cos_minute'] = np.cos(2 * np.pi * (hour * 60 + minute) / (24 * 60))
        hour_meteo['sin_day'] = np.sin(2 * np.pi * time_point.dayofyear / 366)
        hour_meteo['cos_day'] = np.cos(2 * np.pi * time_point.dayofyear / 366)
        
        # 添加太阳能特有特征 - 日照强度
        day_progress = (hour * 60 + minute) / (24 * 60)  # 一天中的进度(0-1)
        hour_meteo['daylight_intensity'] = max(0, np.sin(np.pi * day_progress * 2 - 0.5 * np.pi))
        
        # 转换为DataFrame
        features = pd.DataFrame([hour_meteo])
        
        # 确保所有特征都是数值类型
        for col in features.columns:
            if not np.issubdtype(features[col].dtype, np.number):
                try:
                    features[col] = features[col].astype(float)
                except:
                    # 如果无法转换，使用0填充
                    features[col] = 0
        
        # 获取模型使用的特征名称
        model_features = model.feature_name()
        
        # 检查特征是否匹配
        missing_features = set(model_features) - set(features.columns)
        extra_features = set(features.columns) - set(model_features)
        
        # 添加缺失的特征
        for feat in missing_features:
            features[feat] = 0
            
        # 移除多余的特征
        if extra_features:
            features = features.drop(columns=list(extra_features))
            
        # 确保列顺序一致
        features = features[model_features]
        
        # 预测
        pred = model.predict(features)[0]
        
        # 确保预测值在[0,1]范围内
        pred = max(0, min(1, pred))
        
        # 夜间时段（太阳能特有）- 如果是夜间，强制设为0
        if hour < DAYLIGHT_START_HOUR or hour > DAYLIGHT_END_HOUR:
            pred = 0
            
        predictions.append(pred)
    
    # 创建预测结果DataFrame
    result_df = pd.DataFrame({
        'time': prediction_times,
        'power': predictions
    })
    result_df = result_df.set_index('time')
    
    return result_df