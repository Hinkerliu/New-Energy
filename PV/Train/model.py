import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
import matplotlib.pyplot as plt
from .config import OUTPUT_PATH

def train_lightgbm_model(features_df, station_id, save_model=True):
    """训练LightGBM模型"""
    if features_df is None or features_df.empty:
        print("特征数据为空，无法训练模型")
        return None
    X = features_df.drop(['target_power', 'date'], axis=1, errors='ignore')
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"移除非数值类型列: {non_numeric_cols}")
        X = X.drop(non_numeric_cols, axis=1)
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            try:
                X[col] = X[col].astype(float)
            except:
                X = X.drop(col, axis=1)
    y = features_df['target_power']
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
    time_models = {}
    feature_importances = pd.DataFrame()
    for time_idx in range(96):
        print(f"训练站点 {station_id} 的时间点 {time_idx} 模型...")
        time_mask = features_df['time_index'] == time_idx
        X_time = X[time_mask].drop('time_index', axis=1)
        y_time = y[time_mask]
        if len(X_time) < 10:
            print(f"时间点 {time_idx} 的数据不足，跳过")
            continue
        train_data = lgb.Dataset(X_time, label=y_time)
        model = lgb.train(params, train_data, num_boost_round=1000)
        time_models[time_idx] = model
        importance = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(),
            'time_index': time_idx
        })
        feature_importances = pd.concat([feature_importances, importance])
    if save_model and time_models:
        model_path = os.path.join(OUTPUT_PATH, f"lightgbm_station_{station_id}.pkl")
        joblib.dump(time_models, model_path)
        print(f"模型已保存至: {model_path}")
    if not feature_importances.empty:
        avg_importance = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
        plt.figure(figsize=(12, 8))
        avg_importance.head(20).plot(kind='barh')
        plt.title(f'站点 {station_id} 的特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f"feature_importance_station_{station_id}.png"))
        plt.close()
    return time_models

def predict_with_lightgbm(station_id, day_str, models, get_meteo_for_day_func):
    """使用训练好的LightGBM模型进行预测"""
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
        if hour in meteo_df.index:
            hour_meteo = meteo_df.loc[hour].copy()
        else:
            closest_hour = min(meteo_df.index, key=lambda x: abs(x - hour))
            hour_meteo = meteo_df.loc[closest_hour].copy()
        hour_meteo['hour'] = float(hour)
        hour_meteo['minute'] = float(minute)
        hour_meteo['day_of_week'] = float(time_point.dayofweek)
        hour_meteo['day_of_year'] = float(time_point.dayofyear)
        hour_meteo['month'] = float(time_point.month)
        hour_meteo['is_weekend'] = float(1 if time_point.dayofweek >= 5 else 0)
        hour_meteo['sin_hour'] = np.sin(2 * np.pi * hour / 24)
        hour_meteo['cos_hour'] = np.cos(2 * np.pi * hour / 24)
        hour_meteo['sin_minute'] = np.sin(2 * np.pi * (hour * 60 + minute) / (24 * 60))
        hour_meteo['cos_minute'] = np.cos(2 * np.pi * (hour * 60 + minute) / (24 * 60))
        hour_meteo['sin_day'] = np.sin(2 * np.pi * time_point.dayofyear / 366)
        hour_meteo['cos_day'] = np.cos(2 * np.pi * time_point.dayofyear / 366)
        day_progress = (hour * 60 + minute) / (24 * 60)
        hour_meteo['daylight_intensity'] = max(0, np.sin(np.pi * day_progress * 2 - 0.5 * np.pi))
        features = pd.DataFrame([hour_meteo])
        for col in features.columns:
            if not np.issubdtype(features[col].dtype, np.number):
                try:
                    features[col] = features[col].astype(float)
                except:
                    features[col] = 0
        if i in models:
            model = models[i]
        else:
            closest_idx = min(models.keys(), key=lambda x: abs(x - i))
            model = models[closest_idx]
        model_features = model.feature_name()
        features_filtered = features[model_features] if all(f in features.columns for f in model_features) else None
        if features_filtered is not None:
            pred = model.predict(features_filtered)[0]
        else:
            features_dict = {f: [0] for f in model_features}
            for f in model_features:
                if f in features.columns:
                    features_dict[f] = [features[f].iloc[0]]
            features_filtered = pd.DataFrame(features_dict)
            pred = model.predict(features_filtered)[0]
        pred = max(0, min(1, pred))
        if hour < 6 or hour > 19:
            pred = 0
        predictions.append(pred)
    result_df = pd.DataFrame({'time': prediction_times, 'power': predictions})
    result_df = result_df.set_index('time')
    return result_df