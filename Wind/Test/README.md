# 风电场发电功率预测系统

## 项目概述

本项目是一个基于气象数据的风电场发电功率预测系统，使用LightGBM模型进行预测。系统可以处理多个风电场站点的数据，从多个气象数据源获取信息，并生成未来24小时（96个15分钟时间点）的发电功率预测。

## 项目结构
```
├── config.py              # 配置信息
├── data_loader.py         # 数据加载相关函数
├── feature_engineering.py # 特征工程相关函数
├── prediction.py          # 预测相关函数
├── utils.py               # 工具函数
├── main.py                # 主程序入口
└── README.md              # 项目说明文档
```

## 模块说明

### config.py

包含项目的全局配置信息，如数据路径、模型路径、输出路径等。

### data_loader.py

负责加载和处理气象数据，包括：
- 从不同来源加载气象数据文件
- 提取中心网格点数据
- 合并多个来源的气象数据
- 获取测试集日期范围

### prediction.py

负责模型预测功能，包括：
- 加载训练好的LightGBM模型
- 使用模型进行预测
- 保存预测结果

### utils.py

提供辅助功能，包括：
- 日志格式化和打印
- 警告处理

### main.py

程序入口，协调各模块完成风电场发电功率预测任务。

## 数据说明

### 输入数据

- 气象数据：存储在`NWP_DATA_PATH`指定的路径下
  - 格式：NetCDF (.nc)文件
  - 组织方式：按站点ID和数据源（NWP_1, NWP_2, NWP_3）分类存储
  - 文件命名：YYYYMMDD.nc

### 模型文件

- 存储在`MODEL_PATH`指定的路径下
- 文件命名：lightgbm_station_{station_id}.pkl
- 内容：为每个15分钟时间点训练的LightGBM模型

### 输出数据

- 存储在`OUTPUT_PATH`指定的路径下
- 文件命名：output{station_id}.csv
- 内容：包含时间和预测功率的CSV文件

## 使用方法

1. 确保已安装所有依赖库：
   ```bash
   pip install pandas numpy xarray lightgbm joblib
   ```

2. 配置数据路径：
   - 在`config.py`中设置正确的数据路径、模型路径和输出路径

3. 运行预测：
   ```bash
   python main.py
   ```

4. 查看结果：
   - 预测结果将保存在`OUTPUT_PATH`指定的路径下
   - 日志信息将在控制台输出

## 注意事项

- 确保气象数据文件格式正确，且按照指定的目录结构组织
- 确保模型文件已正确训练并保存
- 预测结果中的功率值已归一化到[0,1]范围
```