# 导入必要的库
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import xarray as xr
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 定义数据路径
nwp_data_path = r"c:\New-Energy\data\nwp_data_train"  # 气象数据路径
power_data_path = r"c:\New-Energy\data\fact_data"     # 功率数据路径
output_path = r"c:\New-Energy\models"                 # 模型输出路径

# 创建输出目录
os.makedirs(output_path, exist_ok=True)

# 1. 加载气象数据函数
def load_meteo_data(station_id, source, day_str):
    """
    加载指定站点的气象数据文件
    参数：
    - station_id: 站点 ID（6-10为光伏站）
    - source: 数据来源（NWP_1, NWP_2, NWP_3）
    - day_str: 日期字符串，格式 YYYYMMDD
    返回：
    - xarray Dataset 对象
    """
    filename = f"{day_str}.nc"
    full_path = os.path.join(nwp_data_path, str(station_id), source, filename)
    try:
        ds = xr.open_dataset(full_path)
        return ds
    except FileNotFoundError:
        # print(f"文件 {full_path} 不存在，跳过")
        return None
    except Exception as e:
        # print(f"加载文件 {full_path} 时出错: {e}")
        return None

# 2. 获取变量名称函数
def get_variable_names(source):
    """
    根据数据来源返回变量列表
    """
    if source in ['NWP_1', 'NWP_3']:
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'sp']
    elif source == 'NWP_2':
        return ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'msl']
    else:
        raise ValueError("无效的数据来源")

# 3. 提取中心网格点数据并按通道拆分
def extract_central_point(ds, source):
    """
    提取 11x11 网格的中心点数据（索引 5）并按通道拆分
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
            print(f"通道数量与变量名称不匹配: {len(channel_values)} vs {len(var_names)}")
            return None
            
        return data_central
    except Exception as e:
        print(f"提取中心点数据时出错: {e}")
        return None

# 4. 为一天获取所有来源的气象数据
def get_meteo_for_day(station_id, day_str):
    """
    为指定站点和日期，从三个来源加载并合并气象数据
    """
    sources = ['NWP_1', 'NWP_2', 'NWP_3']
    all_data = {}
    
    for source in sources:
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
        return pd.DataFrame()

# 5. 加载功率数据
def load_power_data(station_id):
    """
    加载指定站点的功率数据
    """
    filename = f"{station_id}_normalization_train.csv"
    full_path = os.path.join(power_data_path, filename)
    
    try:
        df = pd.read_csv(full_path)
        
        # 检查数据格式
        if len(df.columns) >= 2:
            # 如果列名不是'time'和'power'，尝试重命名
            if df.columns[0] != 'time' or df.columns[1] != 'power':
                df.columns = ['time', 'power']
            
            # 转换时间列为datetime类型
            df['time'] = pd.to_datetime(df['time'])
            # 设置时间列为索引
            df = df.set_index('time')
            
            # 处理异常值
            # 负值设为0
            df.loc[df['power'] < 0, 'power'] = 0
            # 大于1的值设为1（归一化数据）
            df.loc[df['power'] > 1, 'power'] = 1
            
            return df
        else:
            print(f"文件 {full_path} 格式不正确")
            return None
    except Exception as e:
        print(f"加载功率数据时出错: {e}")
        return None

# 6. 创建特征数据集
def create_feature_dataset(station_id, start_date, end_date):
    """
    为指定站点和日期范围创建特征数据集
    参数:
    - start_date: 气象数据开始日期
    - end_date: 气象数据结束日期
    注意: 功率数据是次日的，即start_date的气象数据对应start_date+1天的功率
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
            
            # 添加时间特征
            hour_meteo['hour'] = float(hour)
            hour_meteo['minute'] = float(minute)
            hour_meteo['time_index'] = float(i)  # 0-95的时间索引
            
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
            
            # 添加太阳能特有特征 - 日出日落时间
            # 简化计算，使用正弦函数模拟日照强度
            day_progress = (hour * 60 + minute) / (24 * 60)  # 一天中的进度(0-1)
            hour_meteo['daylight_intensity'] = max(0, np.sin(np.pi * day_progress * 2 - 0.5 * np.pi))
            
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

# 7. 训练LightGBM模型
def train_lightgbm_model(features_df, station_id, save_model=True):
    """
    训练LightGBM模型
    """
    if features_df is None or features_df.empty:
        print("特征数据为空，无法训练模型")
        return None
    
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
    
    # 创建时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    


    # 模型参数 - 针对光伏场站优化
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 255,            # 增加叶子节点数量以捕获更复杂的模式； V1.1.1: 63； V1.1.2: 127, V1.1.3: 255
        'learning_rate': 0.003,       # 降低学习率以获得更好的泛化能力。 V1.1.1: 0.01； V1.1.2: 0.005, V1.13: 0.03      
        'feature_fraction': 0.8,     # 特征抽样比例
        'bagging_fraction': 0.7,     # 数据抽样比例
        'bagging_freq': 5,
        'verbose': -1
    }
  
    # 存储每个时间点的模型
    time_models = {}
    feature_importances = pd.DataFrame()
    
    # 为每个15分钟时间点训练一个模型
    for time_idx in range(96):
        print(f"训练站点 {station_id} 的时间点 {time_idx} 模型...")
        
        # 筛选当前时间点的数据
        time_mask = features_df['time_index'] == time_idx
        X_time = X[time_mask].drop('time_index', axis=1)
        y_time = y[time_mask]
        
        if len(X_time) < 10:  # 确保有足够的数据
            print(f"时间点 {time_idx} 的数据不足，跳过")
            continue
        
        # 创建数据集
        train_data = lgb.Dataset(X_time, label=y_time)
        
        # 训练模型
        model = lgb.train(params, train_data, num_boost_round=1000)
        
        # 保存模型
        time_models[time_idx] = model
        
        # 记录特征重要性
        importance = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(),
            'time_index': time_idx
        })
        feature_importances = pd.concat([feature_importances, importance])
    
    # 保存模型
    if save_model and time_models:
        model_path = os.path.join(output_path, f"lightgbm_station_{station_id}.pkl")
        joblib.dump(time_models, model_path)
        print(f"模型已保存至: {model_path}")
    
    # 分析特征重要性
    if not feature_importances.empty:
        # 计算每个特征的平均重要性
        avg_importance = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(12, 8))
        avg_importance.head(20).plot(kind='barh')
        plt.title(f'站点 {station_id} 的特征重要性')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"feature_importance_station_{station_id}.png"))
        plt.close()
    
    return time_models

# 8. 预测函数
def predict_with_lightgbm(station_id, day_str, models):
    """
    使用训练好的LightGBM模型进行预测
    """
    # 日期转换
    date = pd.to_datetime(day_str)
    next_day = date + pd.Timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    
    # 获取气象数据
    meteo_df = get_meteo_for_day(station_id, day_str)
    if meteo_df.empty:
        print(f"站点 {station_id} 的 {day_str} 气象数据为空，无法预测")
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
        
        # 添加太阳能特有特征 - 日出日落时间
        day_progress = (hour * 60 + minute) / (24 * 60)  # 一天中的进度(0-1)
        hour_meteo['daylight_intensity'] = max(0, np.sin(np.pi * day_progress * 2 - 0.5 * np.pi))
        
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
        
        # 使用对应时间点的模型预测
        if i in models:
            model = models[i]
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
            
            # 夜间时段（太阳能特有）- 如果是夜间，强制设为0
            if hour < 6 or hour > 19:  # 假设6点前和19点后没有太阳能发电
                pred = 0
        else:
            # 如果没有对应时间点的模型，使用最近的模型
            closest_idx = min(models.keys(), key=lambda x: abs(x - i))
            model = models[closest_idx]
            
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
            
            # 夜间时段（太阳能特有）- 如果是夜间，强制设为0
            if hour < 6 or hour > 19:  # 假设6点前和19点后没有太阳能发电
                pred = 0
        
        predictions.append(pred)
    
    # 创建预测结果DataFrame
    result_df = pd.DataFrame({
        'time': prediction_times,
        'power': predictions
    })
    result_df = result_df.set_index('time')
    
    return result_df

# 9. 评估函数
def evaluate_model(station_id, models, test_start_date, test_end_date):
    """
    评估模型在测试集上的性能
    """
    # 转换日期为datetime对象
    start_dt = pd.to_datetime(test_start_date)
    end_dt = pd.to_datetime(test_end_date)
    
    # 创建日期范围 - 注意这里需要减去一天，因为气象数据是预报次日的
    date_range = pd.date_range(start=start_dt, end=end_dt - pd.Timedelta(days=1), freq='D')
    
    # 加载功率数据（真实值）
    power_df = load_power_data(station_id)
    if power_df is None:
        print(f"无法加载站点 {station_id} 的功率数据")
        return None
    
    all_predictions = []
    all_actuals = []
    daily_scores = []
    
    # 处理每一天
    for date in date_range:
        day_str = date.strftime('%Y%m%d')
        next_day = date + pd.Timedelta(days=1)
        next_day_str = next_day.strftime('%Y-%m-%d')
        
        # 预测
        pred_df = predict_with_lightgbm(station_id, day_str, models)
        if pred_df is None:
            continue
        
        # 获取真实值
        actual_df = power_df.loc[next_day_str:next_day_str]
        if actual_df.empty:
            continue
        
        # 确保真实值有96个点
        expected_times = pd.date_range(start=next_day, periods=96, freq='15min')
        if len(actual_df) != 96:
            temp_df = pd.DataFrame(index=expected_times)
            temp_df = temp_df.join(actual_df)
            temp_df = temp_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            actual_df = temp_df
        
        # 计算日准确率
        y_pred = pred_df['power'].values
        y_true = actual_df['power'].values
        
        # 检查并处理NaN值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if not np.any(mask):
            print(f"警告: 站点 {station_id} 的 {next_day_str} 所有数据都是NaN，跳过")
            continue
            
        # 过滤掉NaN值
        y_pred_clean = y_pred[mask]
        y_true_clean = y_true[mask]
        
        if len(y_pred_clean) == 0 or len(y_true_clean) == 0:
            print(f"警告: 站点 {station_id} 的 {next_day_str} 过滤NaN后没有数据，跳过")
            continue
        
        # 计算1-MAE作为准确率
        daily_mae = mean_absolute_error(y_true_clean, y_pred_clean)
        daily_accuracy = 1 - daily_mae
        daily_scores.append(daily_accuracy)
        
        # 存储预测和真实值（过滤掉NaN）
        all_predictions.extend(y_pred_clean)
        all_actuals.extend(y_true_clean)
        
        # 每10天绘制一次对比图
        if len(daily_scores) % 10 == 0:
            plt.figure(figsize=(15, 6))
            # 使用pandas的plot方法，它会自动处理NaN值
            pd.Series(y_true, index=expected_times).plot(label='实际功率', color='blue')
            pd.Series(y_pred, index=expected_times).plot(label='预测功率', color='red')
            plt.title(f'站点 {station_id} - {next_day_str} 功率预测对比')
            plt.xlabel('时间')
            plt.ylabel('归一化功率')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_path, f"prediction_{station_id}_{next_day_str}.png"))
            plt.close()
    
    # 计算总体评估指标
    if all_predictions and all_actuals:
        # 确保没有NaN值
        mae = mean_absolute_error(all_actuals, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_actuals, all_predictions))
        r2 = r2_score(all_actuals, all_predictions)
        
        # 计算平均日准确率
        avg_daily_accuracy = np.mean(daily_scores)
        
        print(f"\n站点 {station_id} 评估结果:")
        print(f"平均日准确率: {avg_daily_accuracy:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # 绘制散点图
        plt.figure(figsize=(10, 10))
        plt.scatter(all_actuals, all_predictions, alpha=0.3)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('实际功率')
        plt.ylabel('预测功率')
        plt.title(f'站点 {station_id} 预测vs实际散点图')
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"scatter_plot_{station_id}.png"))
        plt.close()
        
        return {
            'station_id': station_id,
            'avg_daily_accuracy': avg_daily_accuracy,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    else:
        print(f"站点 {station_id} 没有足够的数据进行评估")
        return None

# 10. 生成提交文件
def generate_submission_files(station_ids, prediction_date, models_dict, output_dir):
    """
    为比赛生成提交文件
    参数:
    - station_ids: 站点ID列表
    - prediction_date: 预测日期字符串，格式YYYYMMDD
    - models_dict: 包含每个站点模型的字典
    - output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换日期
    date = pd.to_datetime(prediction_date)
    next_day = date + pd.Timedelta(days=1)
    
    for station_id in station_ids:
        if station_id not in models_dict:
            print(f"站点 {station_id} 没有模型，跳过")
            continue
        
        # 预测
        pred_df = predict_with_lightgbm(station_id, prediction_date, models_dict[station_id])
        if pred_df is None:
            print(f"站点 {station_id} 的预测失败，跳过")
            continue
        
        # 创建输出文件
        output_file = os.path.join(output_dir, f"output{station_id}.csv")
        
        # 保存为CSV，确保格式正确
        pred_df.to_csv(output_file)
        print(f"站点 {station_id} 的预测结果已保存至: {output_file}")
    
    print("所有提交文件已生成")

# 主函数
def main():
    # 设置参数
    solar_stations = [6, 7, 8, 9, 10]  # 光伏场站
    
     # 修正训练和测试日期范围
    # 训练数据：使用2024年1月1日至11月29日的气象数据预测1月2日至11月30日的功率
    train_start_date = '2024-01-01'
    train_end_date = '2024-11-29'
    
    # 测试数据：使用2024年11月30日至12月30日的气象数据预测12月1日至12月31日的功率
    test_start_date = '2024-11-30'
    test_end_date = '2024-12-30'
    
    results = []
    models_dict = {}
    
    for station_id in solar_stations:
        print(f"\n开始处理光伏场站 {station_id}...")
        
        # 1. 创建特征数据集
        features_df = create_feature_dataset(station_id, train_start_date, train_end_date)
        
        if features_df is not None:
            # 2. 训练模型
            models = train_lightgbm_model(features_df, station_id)
            
            if models:
                # 保存模型到字典
                models_dict[station_id] = models
                
                # 3. 评估模型
                eval_result = evaluate_model(station_id, models, test_start_date, test_end_date)
                if eval_result:
                    results.append(eval_result)
    
    # 汇总结果
    if results:
        results_df = pd.DataFrame(results)
        print("\n所有光伏场站评估结果:")
        print(results_df)
        
        # 计算平均准确率
        avg_accuracy = results_df['avg_daily_accuracy'].mean()
        print(f"\n所有光伏场站平均准确率: {avg_accuracy:.4f}")
        
        # 保存结果
        results_df.to_csv(os.path.join(output_path, "solar_stations_results.csv"), index=False)
    
    # 生成提交文件示例（使用最后一天的气象数据预测）
    submission_dir = os.path.join(output_path, "submission")
    if models_dict:
        generate_submission_files(solar_stations, '20241230', models_dict, submission_dir)

if __name__ == "__main__":
    main()