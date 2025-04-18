"""
数据处理模块 - 包含所有与数据加载和处理相关的函数
"""
import os
import pandas as pd
import numpy as np
import xarray as xr
import warnings
from config import NWP_DATA_PATH, NWP_SOURCES, VARIABLE_NAMES

warnings.filterwarnings('ignore')

def load_meteo_data(station_id, source, day_str):
    """
    加载指定站点的气象数据文件
    
    参数：
    - station_id: 站点 ID（6-10为光伏站）
    - source: 数据来源（NWP_1, NWP_2, NWP_3）
    - day_str: 日期字符串，格式 YYYYMMDD
    
    返回：
    - xarray Dataset 对象，如果加载失败则返回None
    """
    filename = f"{day_str}.nc"
    full_path = os.path.join(NWP_DATA_PATH, str(station_id), source, filename)
    try:
        ds = xr.open_dataset(full_path)
        return ds
    except FileNotFoundError:
        print(f"[信息] 文件不存在: {full_path}")
        return None
    except Exception as e:
        print(f"[错误] 加载文件失败: {full_path}, 原因: {e}")
        return None

def get_variable_names(source):
    """
    根据数据来源返回变量列表
    
    参数:
    - source: 数据来源（NWP_1, NWP_2, NWP_3）
    
    返回:
    - 变量名称列表
    """
    if source in VARIABLE_NAMES:
        return VARIABLE_NAMES[source]
    else:
        raise ValueError(f"[错误] 无效的数据来源: {source}")

def extract_central_point(ds, source):
    """
    提取 11x11 网格的中心点数据（索引 5）并按通道拆分
    
    参数:
    - ds: xarray Dataset 对象
    - source: 数据来源
    
    返回:
    - 中心点数据的 Dataset，如果提取失败则返回None
    """
    if ds is None:
        return None
    
    # 提取中心点数据
    central_lat = 5
    central_lon = 5
    
    try:
        # 选择中心点
        data_central = ds.sel(lat=ds.lat[central_lat], lon=ds.lon[central_lon])
        
        # 按 channel 维度转换为 Dataset
        data_central = data_central.data.to_dataset(dim='channel')
        
        # 获取变量名称并重命名
        var_names = get_variable_names(source)
        channel_values = list(data_central.data_vars.keys())
        
        if len(channel_values) == len(var_names):
            data_central = data_central.rename_vars(dict(zip(channel_values, var_names)))
        else:
            print(f"[警告] 通道数量与变量名称不匹配: {len(channel_values)} vs {len(var_names)}")
            return None
            
        return data_central
    except Exception as e:
        print(f"[错误] 提取中心点数据失败: {e}")
        return None

def get_meteo_for_day(station_id, day_str):
    """
    为指定站点和日期，从三个来源加载并合并气象数据
    
    参数:
    - station_id: 站点ID
    - day_str: 日期字符串，格式 YYYYMMDD
    
    返回:
    - 合并后的气象数据 DataFrame，如果所有来源都加载失败则返回空 DataFrame
    """
    all_data = {}
    
    for source in NWP_SOURCES:
        ds = load_meteo_data(station_id, source, day_str)
        if ds is None:
            continue
            
        data_central = extract_central_point(ds, source)
        if data_central is not None:
            # 使用 to_dataframe() 转换为 DataFrame
            meteo_df = data_central.to_dataframe()
            
            # 如果索引是多级索引，保留 'lead_time' 作为索引
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
        print(f"[警告] 站点 {station_id} 的 {day_str} 所有气象数据源加载失败")
        return pd.DataFrame()