# 特征工程模块，包含特征创建和处理的函数
import pandas as pd
import numpy as np
from data_loader import get_meteo_for_day, load_power_data

def create_feature_dataset(station_id, start_date, end_date):
    """
    为指定站点和日期范围创建特征数据集
    
    参数:
    - station_id: 站点ID
    - start_date: 气象数据开始日期
    - end_date: 气象数据结束日期
    
    注意: 功率数据是次日的，即start_date的气象数据对应start_date+1天的功率
    
    返回:
    - 特征数据集 DataFrame
    """
    # 转换日期为datetime对象
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 创建日期范围
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    
    # 加载功率数据
    power_df = load_power_data(station_id)
    if power_df is None:
        print(f"无法加载站点 {station_id} 的功率数据")
        return None
    
    # 创建空的特征数据框
    all_features = []
    
    # 处理每一天
    for date in date_range:
        day_str = date.strftime('%Y%m%d')
        next_day = date + pd.Timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        
        # 获取气象数据
        meteo_df = get_meteo_for_day(station_id, day_str)
        if meteo_df.empty:
            print(f"站点 {station_id} 的 {day_str} 气象数据为空，跳过")
            continue
        
        # 气压特征PCA处理
        # 气压特征PCA处理 - 修改为保留原始特征
        pressure_cols = ['sp_NWP_1', 'sp_NWP_3', 'msl_NWP_2']
        if all(col in meteo_df.columns for col in pressure_cols):
            from sklearn.decomposition import PCA
            pressure_data = meteo_df[pressure_cols].values
            pca = PCA(n_components=1)
            pressure_pc1 = pca.fit_transform(pressure_data)
            meteo_df['pressure_pc1'] = pressure_pc1.flatten()
            # 不再移除原始气压特征
            # meteo_df = meteo_df.drop(pressure_cols, axis=1)
        else:
            print(f"警告: 站点 {station_id} 的气象数据中缺少气压特征列")
            if 'sp_NWP_1' in meteo_df.columns:
                meteo_df['pressure_pc1'] = meteo_df['sp_NWP_1']
            elif 'msl_NWP_2' in meteo_df.columns:
                meteo_df['pressure_pc1'] = meteo_df['msl_NWP_2']
            else:
                meteo_df['pressure_pc1'] = 0.0
        
        # 打印气象数据的列名，帮助调试
        # print(f"气象数据列名: {meteo_df.columns.tolist()}")
        
        # 检查是否存在风速分量列，考虑NWP后缀
        u_cols = [col for col in meteo_df.columns if col.startswith('u100_NWP')]
        v_cols = [col for col in meteo_df.columns if col.startswith('v100_NWP')]
        
        if u_cols and v_cols:
            # 使用第一个NWP模型的风速分量
            u_col = u_cols[0]
            v_col = v_cols[0]
            # print(f"使用风速分量列: {u_col}, {v_col}")
            
            # 添加风速特征 - 经纬度两个方向的风速进行向量计算得到总风速
            meteo_df['wind_speed'] = np.sqrt(meteo_df[u_col]**2 + meteo_df[v_col]**2)
            
            # 添加风速的立方值作为特征，因为风能与风速的三次方成正比
            # meteo_df['wind_speed_cubed'] = meteo_df['wind_speed']**3
            
            # 计算风向（气象惯例：0°为北，90°为东）
            # wind_direction = (270 - np.arctan2(meteo_df[v_col], meteo_df[u_col]) * (180 / np.pi)) % 360
            # meteo_df['sin_wind_direction'] = np.sin(np.radians(wind_direction))
            # meteo_df['cos_wind_direction'] = np.cos(np.radians(wind_direction))
        else:
            # 如果没有风速相关列，添加一个默认值
            print(f"警告: 站点 {station_id} 的气象数据中没有找到风速分量列，使用默认值")
            meteo_df['wind_speed'] = 0.0
            # # meteo_df['wind_speed_cubed'] = 0.0
            # meteo_df['sin_wind_direction'] = 0.0
            # meteo_df['cos_wind_direction'] = 0.0
        
        # 获取次日功率数据（预测目标）
        next_day_power = power_df.loc[next_day_str:next_day_str]
        if next_day_power.empty:
            print(f"站点 {station_id} 的 {next_day_str} 功率数据为空，跳过")
            continue
        
        # 确保功率数据有96个点（15分钟间隔）
        expected_times = pd.date_range(start=next_day, periods=96, freq='15min')
        if len(next_day_power) != 96:
            # 如果点数不对，重新索引
            temp_df = pd.DataFrame(index=expected_times)
            temp_df = temp_df.join(next_day_power)
            # 填充缺失值
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            next_day_power = temp_df
        
        # 为每个15分钟时间点创建特征
        for i, time_point in enumerate(expected_times):
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
            # 修改time_index计算方式，使用更精确的公式
            hour_meteo['time_index'] = float((time_point.hour * 4) + (time_point.minute // 15))  # 0-95的时间索引
            
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
            
            # 添加目标值
            hour_meteo['target_power'] = next_day_power['power'].iloc[i]
            
            # 添加日期标识（以字符串形式）
            hour_meteo['date'] = next_day_str
            
            # 添加到特征列表
            all_features.append(hour_meteo)
    
    # 合并所有特征
    if all_features:
        features_df = pd.DataFrame(all_features)
        
        # 打印数据类型信息，帮助调试
        print("特征数据类型信息:")
        print(features_df.dtypes)
        
        print(f"站点 {station_id} 的特征数据集创建完成，形状: {features_df.shape}")
        return features_df
    else:
        print(f"站点 {station_id} 没有生成任何特征")
        return None

def preprocess_features(features_df):
    """
    预处理特征数据，确保所有特征都是数值类型
    
    参数:
    - features_df: 特征数据集
    
    返回:
    - X: 处理后的特征
    - y: 目标变量
    """
    if features_df is None or features_df.empty:
        print("特征数据为空，无法处理")
        return None, None
    
    # 准备特征和目标
    # 移除非数值类型列和目标列
    X = features_df.drop(['target_power', 'date'], axis=1, errors='ignore')
    
    # 检查并移除所有非数值类型的列
    non_numeric_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"移除非数值类型列: {non_numeric_cols}")
        X = X.drop(non_numeric_cols, axis=1)
    
    # 确保所有列都是数值类型
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            print(f"将列 {col} 转换为数值类型")
            try:
                X[col] = X[col].astype(float)
            except:
                print(f"无法转换列 {col}，将其移除")
                X = X.drop(col, axis=1)
    
    y = features_df['target_power']
    
    return X, y