import os
import joblib
import numpy as np
import pandas as pd
from meteo import get_meteo_for_day

models_path = r"c:\New-Energy\models"

def load_models(station_id):
    """
    加载指定站点的LightGBM模型。
    """
    model_path = os.path.join(models_path, f"lightgbm_station_{station_id}.pkl")
    try:
        models = joblib.load(model_path)
        print(f"[信息] 成功加载站点 {station_id} 的模型。")
        return models
    except Exception as e:
        print(f"[错误] 加载站点 {station_id} 的模型时出错: {e}")
        return None

def predict_with_lightgbm(station_id, day_str, models):
    """
    使用训练好的LightGBM模型进行预测。
    """
    date = pd.to_datetime(day_str, format='%Y%m%d')
    next_day = date + pd.Timedelta(days=1)
    meteo_df = get_meteo_for_day(station_id, day_str)
    if meteo_df.empty:
        print(f"[警告] 站点 {station_id} 的 {day_str} 气象数据为空，无法预测。")
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