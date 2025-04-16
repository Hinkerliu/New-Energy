# 光伏功率预测项目

## 项目结构

- `config.py`：路径和全局参数配置
- `data_utils.py`：气象和功率数据加载、预处理
- `feature_engineering.py`：特征工程与数据集构建
- `model.py`：模型训练与预测
- `evaluation.py`：模型评估
- `submission.py`：生成比赛提交文件
- `main.py`：主程序入口

## 运行环境

- Python 3.8+
- 依赖库：pandas, numpy, lightgbm, scikit-learn, matplotlib, xarray, joblib, seaborn

## 使用方法

1. 配置 `config.py` 中的数据路径。
2. 运行主程序：
   ```bash
   python main.py
   ```
3. 训练、评估和提交文件会自动生成在 `models` 目录下。

## 说明

- 支持多站点批量训练和预测。
- 支持特征重要性分析和可视化。
- 评估结果和提交文件自动保存。

---

如需自定义特征或模型参数，请修改对应模块。
