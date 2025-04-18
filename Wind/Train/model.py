# 模型训练和预测模块
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from config import MODEL_PARAMS, OUTPUT_PATH
from data_loader import get_meteo_for_day
from feature_engineering import preprocess_features

def train_lightgbm_model(features_df, station_id, save_model=True):
    """
    训练统一的LightGBM模型
    
    参数:
    - features_df: 特征数据集
    - station_id: 站点ID
    - save_model: 是否保存模型
    
    返回:
    - 训练好的模型
    """
    # 预处理特征
    X, y = preprocess_features(features_df)
    if X is None or y is None:
        return None
    
    print(f"训练站点 {station_id} 的统一模型...")
    
    # 创建时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 创建数据集
    train_data = lgb.Dataset(X, label=y)
    
    # 训练模型
    model = lgb.train(MODEL_PARAMS, train_data, num_boost_round=1000)
    
    # 保存模型
    if save_model:
        model_path = os.path.join(OUTPUT_PATH, f"lightgbm_station_{station_id}.pkl")
        joblib.dump(model, model_path)
        print(f"模型已保存至: {model_path}")
    
    # 分析特征重要性
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance()
    })
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    importance.sort_values('importance', ascending=False).head(20).plot(kind='barh', x='feature', y='importance')
    plt.title(f'站点 {station_id} 的特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"feature_importance_station_{station_id}.png"))
    plt.close()
    
    return model

def predict_with_lightgbm(station_id, day_str, model):
    """
    使用训练好的LightGBM模型进行预测
    
    参数:
    - station_id: 站点ID
    - day_str: 日期字符串，格式 YYYYMMDD
    - model: 训练好的模型
    
    返回:
    - 预测结果 DataFrame
    """
    # 日期转换
    date = pd.to_datetime(day_str)
    next_day = date + pd.Timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    
    # 获取气象数据
    meteo_df = get_meteo_for_day(station_id, day_str)
    if meteo_df.empty:
        print(f"站点 {station_id} 的 {day_str} 气象数据为空，无法预测")
        return None
    
    # 创建预测时间点
    prediction_times = pd.date_range(start=next_day, periods=96, freq='15min')
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
        hour_meteo['time_index'] = float(i)  # 添加时间索引作为特征
        
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
        
        # 只保留模型使用的特征
        features_filtered = features[model_features] if all(f in features.columns for f in model_features) else None
        
        if features_filtered is not None:
            pred = model.predict(features_filtered)[0]
        else:
            # 如果特征不匹配，创建一个与模型特征匹配的DataFrame
            features_dict = {f: [0] for f in model_features}
            for f in model_features:
                if f in features.columns:
                    features_dict[f] = [features[f].iloc[0]]
            features_filtered = pd.DataFrame(features_dict)
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