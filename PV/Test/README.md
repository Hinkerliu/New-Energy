### 5. README.md

# PV Power Forecast System

本项目用于基于气象数据和LightGBM模型预测光伏电站发电量，并生成比赛提交文件。

## 目录结构

- `PV-Test-V1.3.0.py`：主程序入口
- `meteo.py`：气象数据加载与处理
- `model.py`：模型加载与预测
- `submission.py`：生成提交文件
- `README.md`：项目说明

## 运行环境

- Python 3.7+
- pandas
- numpy
- lightgbm
- xarray
- joblib

## 使用方法

1. 确保模型文件和气象数据已放置在 `c:\New-Energy\models` 和 `c:\New-Energy\data\nwp_data_test` 目录下。
2. 运行主程序：

   ```bash
   python PV-Test-V1.3.0.py
   ```

3. 预测结果将保存在 `c:\New-Energy\output` 目录下，并自动打包为 `output.zip`。

## 功能说明

- 支持多站点、多天批量预测
- 自动处理缺失数据
- 预测结果自动保存并压缩

## 贡献

欢迎提交issue和PR改进本项目。