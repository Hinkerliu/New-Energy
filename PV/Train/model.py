import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
import matplotlib.pyplot as plt
# 添加中文字体支持
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 将相对导入改为绝对导入
from config import OUTPUT_PATH

def train_lightgbm_model(features_df, station_id, save_model=True):
    """训练统一的LightGBM模型"""
    if features_df is None or features_df.empty:
        print("特征数据为空，无法训练模型")
        return None
    
    # 准备特征和目标
    X = features_df.drop(['target_power', 'date'], axis=1, errors='ignore')
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"移除非数值类型列: {non_numeric_cols}")
        X = X.drop(non_numeric_cols, axis=1)
    
    # 确保所有列都是数值类型
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            try:
                X[col] = X[col].astype(float)
            except:
                X = X.drop(col, axis=1)
    
    y = features_df['target_power']
    
    # 设置LightGBM参数
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 255,
        'learning_rate': 0.003,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    print(f"训练站点 {station_id} 的统一模型...")
    
    # 创建训练数据集
    train_data = lgb.Dataset(X, label=y)
    
    # 训练统一模型
    model = lgb.train(params, train_data, num_boost_round=1000)
    
    # 保存模型
    if save_model:
        model_path = os.path.join(OUTPUT_PATH, f"lightgbm_station_{station_id}.pkl")
        joblib.dump(model, model_path)
        print(f"统一模型已保存至: {model_path}")
    
    # 分析特征重要性
    importance = pd.DataFrame({
        'feature': model.feature_name(),
        'importance': model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    importance.head(20).plot(kind='barh', x='feature', y='importance')
    plt.title(f'站点 {station_id} 的特征重要性')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f"feature_importance_station_{station_id}.png"))
    plt.close()
    
    return model

def predict_with_lightgbm(station_id, day_str, model, get_meteo_for_day_func):
    """使用训练好的统一LightGBM模型进行预测"""
    import numpy as np
    import pandas as pd
    
    date = pd.to_datetime(day_str)
    next_day = date + pd.Timedelta(days=1)
    meteo_df = get_meteo_for_day_func(station_id, day_str)
    
    if meteo_df.empty:
        print(f"站点 {station_id} 的 {day_str} 气象数据为空，无法预测")
        return None
    
    prediction_times = pd.date_range(start=next_day, periods=96, freq='15min')
    predictions = []
    
    for i, time_point in enumerate(prediction_times):
        hour = time_point.hour
        minute = time_point.minute
        
        # 获取对应小时的气象数据
        if hour in meteo_df.index:
            hour_meteo = meteo_df.loc[hour].copy()
        else:
            closest_hour = min(meteo_df.index, key=lambda x: abs(x - hour))
            hour_meteo = meteo_df.loc[closest_hour].copy()
        
        # 添加时间特征
        hour_meteo['hour'] = float(hour)
        hour_meteo['minute'] = float(minute)
        hour_meteo['time_index'] = float(i)
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
        
        # 添加日光强度特征
        day_progress = (hour * 60 + minute) / (24 * 60)
        hour_meteo['daylight_intensity'] = max(0, np.sin(np.pi * day_progress * 2 - 0.5 * np.pi))
        
        # 准备特征
        features = pd.DataFrame([hour_meteo])
        
        # 确保所有特征都是数值类型
        for col in features.columns:
            if not np.issubdtype(features[col].dtype, np.number):
                try:
                    features[col] = features[col].astype(float)
                except:
                    features[col] = 0
        
        # 确保特征列与模型期望的一致
        model_features = model.feature_name()
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
        
        # 限制预测值在合理范围内
        pred = max(0, min(1, pred))
        
        # 夜间功率设为0
        if hour < 6 or hour > 19:
            pred = 0
            
        predictions.append(pred)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({'time': prediction_times, 'power': predictions})
    result_df = result_df.set_index('time')
    
    return result_df