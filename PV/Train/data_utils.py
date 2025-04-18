import os
import pandas as pd
import xarray as xr
# 将相对导入改为绝对导入
from config import NWP_DATA_PATH, POWER_DATA_PATH

def load_meteo_data(station_id, source, day_str):
    """加载指定站点的气象数据文件"""
    filename = f"{day_str}.nc"
    full_path = os.path.join(NWP_DATA_PATH, str(station_id), source, filename)
    try:
        ds = xr.open_dataset(full_path)
        return ds
    except FileNotFoundError:
        return None
    except Exception:
        return None

def get_variable_names(source):
    """根据数据来源返回变量列表"""
    if source in ['NWP_1', 'NWP_3']:
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'sp']
    elif source == 'NWP_2':
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'msl']
    else:
        raise ValueError("无效的数据来源")

def extract_central_point(ds, source):
    """提取中心网格点数据并按通道拆分"""
    if ds is None:
        return None
    central_lat = 5
    central_lon = 5
    try:
        data_central = ds.sel(lat=ds.lat[central_lat], lon=ds.lon[central_lon])
        data_central = data_central.data.to_dataset(dim='channel')
        var_names = get_variable_names(source)
        channel_values = list(data_central.data_vars.keys())
        if len(channel_values) == len(var_names):
            data_central = data_central.rename_vars(dict(zip(channel_values, var_names)))
        else:
            print(f"通道数量与变量名称不匹配: {len(channel_values)} vs {len(var_names)}")
            return None
        return data_central
    except Exception as e:
        print(f"提取中心点数据时出错: {e}")
        return None

def get_meteo_for_day(station_id, day_str):
    """为指定站点和日期，从三个来源加载并合并气象数据"""
    sources = ['NWP_1', 'NWP_2', 'NWP_3']
    all_data = {}
    for source in sources:
        ds = load_meteo_data(station_id, source, day_str)
        if ds is None:
            continue
        data_central = extract_central_point(ds, source)
        if data_central is not None:
            meteo_df = data_central.to_dataframe()
            if isinstance(meteo_df.index, pd.MultiIndex):
                meteo_df = meteo_df.reset_index()
                if 'lead_time' in meteo_df.columns:
                    meteo_df = meteo_df.set_index('lead_time')
                elif 'hour' in meteo_df.columns:
                    meteo_df = meteo_df.set_index('hour')
            meteo_df.columns = [f"{col}_{source}" for col in meteo_df.columns]
            all_data[source] = meteo_df
        else:
            all_data[source] = pd.DataFrame()
    if all_data:
        merged_df = pd.concat(all_data.values(), axis=1)
        # 修改这一行，使用更新的方法替代弃用的方法
        merged_df = merged_df.ffill().bfill().fillna(0)
        return merged_df
    else:
        return pd.DataFrame()

def load_power_data(station_id):
    """加载指定站点的功率数据"""
    filename = f"{station_id}_normalization_train.csv"
    full_path = os.path.join(POWER_DATA_PATH, filename)
    try:
        df = pd.read_csv(full_path)
        if len(df.columns) >= 2:
            if df.columns[0] != 'time' or df.columns[1] != 'power':
                df.columns = ['time', 'power']
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
            df.loc[df['power'] < 0, 'power'] = 0
            df.loc[df['power'] > 1, 'power'] = 1
            return df
        else:
            print(f"文件 {full_path} 格式不正确")
            return None
    except Exception as e:
        print(f"加载功率数据时出错: {e}")
        return None

def prepare_unified_dataset(station_id, start_date, end_date):
    """
    准备用于统一模型训练的数据集
    
    参数:
    - station_id: 站点ID
    - start_date: 开始日期
    - end_date: 结束日期
    
    返回:
    - 包含所有时间点数据的DataFrame
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
            
            # 创建特征字典
            feature = {}
            
            # 添加时间特征
            feature['hour'] = float(hour)
            feature['minute'] = float(minute)
            feature['time_index'] = float((hour * 4) + (minute // 15))  # 0-95的时间索引
            feature['day_of_week'] = float(time_point.dayofweek)
            feature['day_of_year'] = float(time_point.dayofyear)
            feature['month'] = float(time_point.month)
            feature['is_weekend'] = float(1 if time_point.dayofweek >= 5 else 0)
            
            # 添加周期性时间特征
            feature['sin_hour'] = float(np.sin(2 * np.pi * hour / 24))
            feature['cos_hour'] = float(np.cos(2 * np.pi * hour / 24))
            feature['sin_minute'] = float(np.sin(2 * np.pi * (hour * 60 + minute) / (24 * 60)))
            feature['cos_minute'] = float(np.cos(2 * np.pi * (hour * 60 + minute) / (24 * 60)))
            feature['sin_day'] = float(np.sin(2 * np.pi * time_point.dayofyear / 366))
            feature['cos_day'] = float(np.cos(2 * np.pi * time_point.dayofyear / 366))
            
            # 添加气象特征
            for col in hour_meteo.index:
                feature[col] = float(hour_meteo[col])
            
            # 添加目标值
            feature['target_power'] = float(next_day_power['power'].iloc[i])
            
            # 添加日期标识
            feature['date'] = next_day_str
            
            # 添加到特征列表
            all_features.append(feature)
    
    # 合并所有特征
    if all_features:
        features_df = pd.DataFrame(all_features)
        print(f"站点 {station_id} 的统一数据集创建完成，形状: {features_df.shape}")
        return features_df
    else:
        print(f"站点 {station_id} 没有生成任何特征")
        return None

def preprocess_unified_dataset(features_df):
    """
    预处理统一数据集，确保所有特征都是数值类型
    
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
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"将列 {col} 转换为数值类型")
            try:
                X[col] = X[col].astype(float)
            except:
                print(f"无法转换列 {col}，将其移除")
                X = X.drop(col, axis=1)
    
    y = features_df['target_power']
    
    return X, y