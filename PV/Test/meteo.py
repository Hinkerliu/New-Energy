import os
import pandas as pd
import xarray as xr

nwp_data_path = r"c:\New-Energy\data\nwp_data_test"

def load_meteo_data(station_id, source, day_str):
    """
    加载指定站点的气象数据文件。
    """
    filename = f"{day_str}.nc"
    full_path = os.path.join(nwp_data_path, str(station_id), source, filename)
    try:
        ds = xr.open_dataset(full_path)
        return ds
    except FileNotFoundError:
        print(f"[警告] 文件 {full_path} 不存在，已跳过。")
        return None
    except Exception as e:
        print(f"[错误] 加载文件 {full_path} 时出错: {e}")
        return None

def get_variable_names(source):
    """
    根据数据来源返回变量列表。
    """
    if source in ['NWP_1', 'NWP_3']:
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'sp']
    elif source == 'NWP_2':
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'msl']
    else:
        raise ValueError("无效的数据来源")

def extract_central_point(ds, source):
    """
    提取11x11网格的中心点数据并按通道拆分。
    """
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
            print(f"[警告] 通道数量与变量名称不匹配: {len(channel_values)} vs {len(var_names)}")
            return None
        return data_central
    except Exception as e:
        print(f"[错误] 提取中心点数据时出错: {e}")
        return None

def get_meteo_for_day(station_id, day_str):
    """
    为指定站点和日期，从三个来源加载并合并气象数据。
    """
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
        # 填充缺失值
        merged_df = merged_df.ffill().bfill().fillna(0)
        return merged_df
    else:
        return pd.DataFrame()