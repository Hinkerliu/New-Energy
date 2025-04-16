"""
数据加载模块 - 负责加载和处理气象数据
"""
import os
import pandas as pd
import xarray as xr
from config import NWP_DATA_PATH, DATA_SOURCES

def load_meteo_data(station_id, source, day_str):
    """
    加载指定站点的气象数据文件
    
    参数：
        station_id (int): 站点ID（1-10）
        source (str): 数据来源（NWP_1, NWP_2, NWP_3）
        day_str (str): 日期字符串，格式 YYYYMMDD
        
    返回：
        xarray.Dataset 或 None: 加载的数据集，加载失败则返回None
    """
    filename = f"{day_str}.nc"
    full_path = os.path.join(NWP_DATA_PATH, str(station_id), source, filename)
    try:
        ds = xr.open_dataset(full_path)
        return ds
    except FileNotFoundError:
        print(f"信息: 文件 {full_path} 不存在，跳过")
        return None
    except Exception as e:
        print(f"错误: 加载文件 {full_path} 时出错: {e}")
        return None

def get_variable_names(source):
    """
    根据数据来源返回变量列表
    
    参数:
        source (str): 数据来源名称
        
    返回:
        list: 变量名称列表
        
    异常:
        ValueError: 当提供的数据来源无效时
    """
    if source in ['NWP_1', 'NWP_3']:
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'sp']
    elif source == 'NWP_2':
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'msl']
    else:
        raise ValueError(f"错误: 无效的数据来源 '{source}'")

def extract_central_point(ds, source):
    """
    提取11x11网格的中心点数据（索引5）并按通道拆分
    
    参数:
        ds (xarray.Dataset): 数据集
        source (str): 数据来源
        
    返回:
        xarray.Dataset 或 None: 处理后的中心点数据，处理失败则返回None
    """
    if ds is None:
        return None
    
    # 提取中心点数据
    central_lat = 5
    central_lon = 5
    
    try:
        # 选择中心点
        data_central = ds.sel(lat=ds.lat[central_lat], lon=ds.lon[central_lon])
        
        # 按channel维度转换为Dataset
        data_central = data_central.data.to_dataset(dim='channel')
        
        # 获取变量名称并重命名
        var_names = get_variable_names(source)
        channel_values = list(data_central.data_vars.keys())
        
        if len(channel_values) == len(var_names):
            data_central = data_central.rename_vars(dict(zip(channel_values, var_names)))
        else:
            print(f"警告: 通道数量与变量名称不匹配: {len(channel_values)} vs {len(var_names)}")
            return None
            
        return data_central
    except Exception as e:
        print(f"错误: 提取中心点数据时出错: {e}")
        return None

def get_meteo_for_day(station_id, day_str):
    """
    为指定站点和日期，从三个来源加载并合并气象数据
    
    参数:
        station_id (int): 站点ID
        day_str (str): 日期字符串，格式 YYYYMMDD
        
    返回:
        pandas.DataFrame: 合并后的气象数据，如果没有数据则返回空DataFrame
    """
    all_data = {}
    
    for source in DATA_SOURCES:
        ds = load_meteo_data(station_id, source, day_str)
        if ds is None:
            continue
            
        data_central = extract_central_point(ds, source)
        if data_central is not None:
            # 使用to_dataframe()转换为DataFrame
            meteo_df = data_central.to_dataframe()
            
            # 如果索引是多级索引，保留'lead_time'作为索引
            if isinstance(meteo_df.index, pd.MultiIndex):
                meteo_df = meteo_df.reset_index()
                if 'lead_time' in meteo_df.columns:
                    meteo_df = meteo_df.set_index('lead_time')
                else:
                    # 如果没有lead_time，尝试使用hour
                    if 'hour' in meteo_df.columns:
                        meteo_df = meteo_df.set_index('hour')
            
            # 重命名列，添加来源标识
            meteo_df.columns = [f"{col}_{source}" for col in meteo_df.columns]
            all_data[source] = meteo_df
        else:
            all_data[source] = pd.DataFrame()
    
    # 合并所有来源的数据
    if all_data:
        merged_df = pd.concat(all_data.values(), axis=1)
        # 填充缺失值
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return merged_df
    else:
        return pd.DataFrame()

def get_test_dates(station_id):
    """
    获取测试集中可用的日期列表
    
    参数:
        station_id (int): 站点ID
        
    返回:
        list: 可用日期字符串列表
    """
    station_path = os.path.join(NWP_DATA_PATH, str(station_id), 'NWP_1')
    if not os.path.exists(station_path):
        print(f"警告: 站点{station_id}的测试数据路径不存在: {station_path}")
        return []
    
    # 获取所有nc文件
    nc_files = [f for f in os.listdir(station_path) if f.endswith('.nc')]
    
    # 提取日期
    dates = [f.split('.')[0] for f in nc_files]
    
    # 按日期排序
    dates.sort()
    
    return dates