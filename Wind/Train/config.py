# 配置文件，包含路径设置和全局参数
import os
import warnings
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 定义数据路径
NWP_DATA_PATH = r"c:\New-Energy\data\nwp_data_train"  # 气象数据路径
POWER_DATA_PATH = r"c:\New-Energy\data\fact_data"     # 功率数据路径
OUTPUT_PATH = r"c:\New-Energy\models"                 # 模型输出路径

# 创建输出目录
os.makedirs(OUTPUT_PATH, exist_ok=True)

# 模型参数
MODEL_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 255,            # 增加叶子节点数量以捕获更复杂的模式
    'learning_rate': 0.003,       # 降低学习率以获得更好的泛化能力
    'feature_fraction': 0.8,      # 特征抽样比例
    'bagging_fraction': 0.7,      # 数据抽样比例
    'bagging_freq': 5,
    'verbose': -1
}

# 风电场站ID列表
WIND_STATIONS = [1, 2, 3, 4, 5]

# 训练和测试日期范围
TRAIN_START_DATE = '2024-01-01'
TRAIN_END_DATE = '2024-11-29'
TEST_START_DATE = '2024-11-30'
TEST_END_DATE = '2024-12-30'