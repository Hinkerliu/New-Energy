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
from utils import print_info, print_warning, print_error  # 确保导入了 print_error

# 修改 predict_with_lightgbm 函数，使其适应单一模型而非模型字典
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
        print_warning(f"站点 {station_id} 的 {day_str} 气象数据为空，无法预测")
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

def load_model(station_id):
    """
    加载训练好的模型
    
    参数:
    - station_id: 站点ID
    
    返回:
    - 加载的模型
    """
    # 添加新的模型路径 - 模型文件存放在 c:\New-Energy\models\ 目录
    possible_paths = [
        os.path.join("c:", "New-Energy", "models", f"lightgbm_station_{station_id}.pkl"),  # 新添加的路径
        os.path.join("..", "..", "output", f"lightgbm_station_{station_id}.pkl"),  # 相对路径
        os.path.join("output", f"lightgbm_station_{station_id}.pkl"),  # 当前目录下的output
        os.path.join("..", "Train", "output", f"lightgbm_station_{station_id}.pkl"),  # Train目录下的output
        os.path.join("..", "..", "..", "output", f"lightgbm_station_{station_id}.pkl")  # 更上层目录
    ]
    
    # 尝试绝对路径
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths.append(os.path.join(base_dir, "output", f"lightgbm_station_{station_id}.pkl"))
    
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print_info(f"成功加载站点{station_id}的模型，路径: {model_path}")
                return model
        except Exception as e:
            continue
    
    # 如果所有路径都失败，打印错误并返回None
    print_error(f"加载站点{station_id}的模型失败: 找不到模型文件。尝试的路径: {possible_paths}")
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