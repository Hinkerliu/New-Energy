import os

# 数据路径配置
NWP_DATA_PATH = r"c:\New-Energy\data\nwp_data_train"
POWER_DATA_PATH = r"c:\New-Energy\data\fact_data"
OUTPUT_PATH = r"c:\New-Energy\models"

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)