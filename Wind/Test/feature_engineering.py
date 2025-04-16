"""
特征工程模块 - 负责特征提取和转换
"""
import numpy as np
import pandas as pd

def add_time_features(features_df, time_point):
    """
    为特征DataFrame添加时间相关特征
    
    参数:
        features_df (pandas.DataFrame): 特征数据框
        time_point (datetime): 时间点
        
    返回:
        pandas.DataFrame: 添加了时间特征的数据框
    """
    hour = time_point.hour
    minute = time_point.minute
    
    # 添加基本时间特征
    features_df['hour'] = float(hour)
    features_df['minute'] = float(minute)
    
    # 添加日期特征
    features_df['day_of_week'] = float(time_point.dayofweek)
    features_df['day_of_year'] = float(time_point.dayofyear)
    features_df['month'] = float(time_point.month)
    features_df['is_weekend'] = float(1 if time_point.dayofweek >= 5 else 0)
    
    # 添加周期性特征
    features_df['sin_hour'] = np.sin(2 * np.pi * hour / 24)
    features_df['cos_hour'] = np.cos(2 * np.pi * hour / 24)
    features_df['sin_minute'] = np.sin(2 * np.pi * (hour * 60 + minute) / (24 * 60))
    features_df['cos_minute'] = np.cos(2 * np.pi * (hour * 60 + minute) / (24 * 60))
    features_df['sin_day'] = np.sin(2 * np.pi * time_point.dayofyear / 366)
    features_df['cos_day'] = np.cos(2 * np.pi * time_point.dayofyear / 366)
    
    return features_df

def ensure_numeric_features(features_df):
    """
    确保所有特征都是数值类型
    
    参数:
        features_df (pandas.DataFrame): 特征数据框
        
    返回:
        pandas.DataFrame: 转换后的数据框
    """
    for col in features_df.columns:
        if not np.issubdtype(features_df[col].dtype, np.number):
            try:
                features_df[col] = features_df[col].astype(float)
            except:
                # 如果无法转换，使用0填充
                features_df[col] = 0
    
    return features_df

def prepare_features_for_model(features_df, model_features):
    """
    准备与模型兼容的特征数据框
    
    参数:
        features_df (pandas.DataFrame): 原始特征数据框
        model_features (list): 模型使用的特征名称列表
        
    返回:
        pandas.DataFrame: 准备好的特征数据框
    """
    # 检查是否所有模型特征都在数据框中
    if all(f in features_df.columns for f in model_features):
        return features_df[model_features]
    else:
        # 创建一个与模型特征匹配的DataFrame
        features_dict = {f: [0] for f in model_features}
        for f in model_features:
            if f in features_df.columns:
                features_dict[f] = [features_df[f].iloc[0]]
        return pd.DataFrame(features_dict)