"""
配置文件 - 存储所有路径和常量配置
"""
import os

# 定义路径
MODELS_PATH = r"c:\New-Energy\models"                  # 模型路径
NWP_DATA_PATH = r"c:\New-Energy\data\nwp_data_test"    # 测试气象数据路径
# 修改为相对路径
OUTPUT_PATH = r"..\..\output"                          # 输出路径

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 光伏场站ID列表
SOLAR_STATIONS = [6, 7, 8, 9, 10]

# 预测日期范围
START_DATE = '20250101'
END_DATE = '20250228'

# 模型文件名格式 - 调整顺序，将实际使用的格式放在最前面
MODEL_FILENAME_FORMATS = [
    "lightgbm_station_{}.pkl",          # 标准格式（实际使用的格式）
    "lightgbm_unified_station_{}.pkl",  # 统一模型格式
    "lightgbm_{}.pkl"                   # 旧格式
]

# 气象数据源
NWP_SOURCES = ['NWP_1', 'NWP_2', 'NWP_3']

# 变量名称映射
VARIABLE_NAMES = {
    'NWP_1': ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'sp'],
    'NWP_2': ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'msl'],
    'NWP_3': ['u100', 'v100', 't2m', 'tp', 'tcc', 'poai', 'ghi', 'sp']
}

# 预测参数
PREDICTION_PERIODS = 96  # 每天96个15分钟时间点
DAYLIGHT_START_HOUR = 6  # 日出时间（小时）
DAYLIGHT_END_HOUR = 19   # 日落时间（小时）