"""
配置模块 - 存储全局配置信息
"""
import os

# 定义数据路径
NWP_DATA_PATH = r"c:\New-Energy\data\nwp_data_test"  # 测试气象数据路径
MODEL_PATH = r"c:\New-Energy\models"                 # 模型路径
OUTPUT_PATH = r"c:\New-Energy\output"                # 输出结果路径

# 风电站点ID列表
WIND_STATIONS = [1, 2, 3, 4, 5]

# 数据源列表
DATA_SOURCES = ['NWP_1', 'NWP_2', 'NWP_3']

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)