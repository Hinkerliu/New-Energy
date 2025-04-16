import numpy as np
import pandas as pd
from .data_utils import get_meteo_for_day, load_power_data

def create_feature_dataset(station_id, start_date, end_date):
    """为指定站点和日期范围创建特征数据集"""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    power_df = load_power_data(station_id)
    if power_df is None:
        print(f"无法加载站点 {station_id} 的功率数据")
        return None
    all_features = []
    for date in date_range:
        day_str = date.strftime('%Y%m%d')
        next_day = date + pd.Timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        meteo_df = get_meteo_for_day(station_id, day_str)
        if meteo_df.empty:
            print(f"站点 {station_id} 的 {day_str} 气象数据为空，跳过")
            continue
        next_day_power = power_df.loc[next_day_str:next_day_str]
        if next_day_power.empty:
            print(f"站点 {station_id} 的 {next_day_str} 功率数据为空，跳过")
            continue
        expected_times = pd.date_range(start=next_day, periods=96, freq='15min')
        if len(next_day_power) != 96:
            temp_df = pd.DataFrame(index=expected_times)
            temp_df = temp_df.join(next_day_power)
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            next_day_power = temp_df
        for i, time_point in enumerate(expected_times):
            hour = time_point.hour
            minute = time_point.minute
            if hour in meteo_df.index:
                hour_meteo = meteo_df.loc[hour].copy()
            else:
                closest_hour = min(meteo_df.index, key=lambda x: abs(x - hour))
                hour_meteo = meteo_df.loc[closest_hour].copy()
            hour_meteo['hour'] = float(hour)
            hour_meteo['minute'] = float(minute)
            hour_meteo['time_index'] = float(i)
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
            hour_meteo['target_power'] = next_day_power['power'].iloc[i]
            hour_meteo['date'] = next_day_str
            all_features.append(hour_meteo)
    if all_features:
        features_df = pd.DataFrame(all_features)
        print(f"站点 {station_id} 的特征数据集创建完成，形状: {features_df.shape}")
        return features_df
    else:
        print(f"站点 {station_id} 没有生成任何特征")
        return None