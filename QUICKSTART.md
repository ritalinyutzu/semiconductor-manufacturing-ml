# 快速开始指南

## 📦 项目已完成的框架

✅ 完整的项目结构
✅ 5个核心Python模块（预处理、模型、评估、工具）
✅ 详细的代码注释和文档
✅ Git仓库已初始化

---

## 🚀 下一步操作

### 1️⃣ 关联到GitHub仓库

你已经在GitHub上创建了空仓库 `semiconductor-manufacturing-ml`

在你的电脑上执行：

```bash
# 进入项目文件夹
cd /home/claude/semiconductor-manufacturing-ml

# 添加远程仓库（替换YOUR_USERNAME为你的GitHub用户名）
git remote add origin https://github.com/YOUR_USERNAME/semiconductor-manufacturing-ml.git

# 改名分支（如需要）
git branch -M main

# 推送到GitHub
git push -u origin main
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 3️⃣ 下载数据

从 Kaggle 下载 [UCI SECOM Dataset](https://www.kaggle.com/datasets/paresh2047/uci-semcom)

将文件放在 `data/raw/` 文件夹

---

## 📊 项目工作流程

我们将按以下顺序完成：

```
01. EDA (探索性数据分析)
    ├─ 加载数据
    ├─ 检查缺失值
    ├─ 类别分布分析
    └─ 基本统计

02. 数据预处理
    ├─ 处理缺失值
    ├─ 移除零方差特征
    ├─ 特征缩放
    └─ PCA降维

03. 模型训练
    ├─ KNN
    ├─ Naive Bayes
    ├─ Logistic Regression
    ├─ Random Forest
    └─ XGBoost

04. 结果分析
    ├─ 混淆矩阵
    ├─ ROC曲线
    ├─ 特征重要性
    └─ 最终总结
```

---

## 🔧 核心模块说明

### `src/preprocessing.py`
- `DataPreprocessor`: 数据处理类
  - `load_data()` - 加载数据
  - `handle_missing_values()` - 处理缺失值
  - `remove_zero_variance_features()` - 移除零方差特征
  - `scale_features()` - 特征缩放
  - `apply_pca()` - PCA降维

### `src/models.py`
- `ModelTrainer`: 模型训练器
  - `train_and_evaluate_all()` - 训练所有模型
  - `get_best_model()` - 获取最佳模型
- `HyperparameterTuner`: 超参数调优

### `src/evaluate.py`
- `ModelEvaluator`: 模型评估
  - `plot_confusion_matrix()` - 混淆矩阵
  - `plot_roc_curve()` - ROC曲线
  - `compare_models()` - 模型对比

### `src/utils.py`
- 数据分析和可视化工具函数

---

## 📝 使用示例

```python
# 导入模块
from src.preprocessing import DataPreprocessor
from src.models import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import split_data

# 1. 加载和预处理数据
preprocessor = DataPreprocessor()
df = preprocessor.load_data('data/raw/secom.csv')
df = preprocessor.handle_missing_values(df)
df = preprocessor.remove_zero_variance_features(df, exclude_cols=['结果列'])

# 2. 特征缩放和PCA
X_train_scaled = preprocessor.scale_features(X_train)
X_train_pca = preprocessor.apply_pca(X_train_scaled, variance_ratio=0.95)

# 3. 训练模型
trainer = ModelTrainer()
results_df, trained_models = trainer.train_and_evaluate_all(
    X_train_pca, y_train, X_test_pca, y_test
)

# 4. 获取最佳模型
best_model_name, best_model = trainer.get_best_model()

# 5. 评估
evaluator = ModelEvaluator()
evaluator.plot_confusion_matrix(y_test, y_pred, model_name=best_model_name)
evaluator.plot_roc_curve(y_test, y_proba, model_name=best_model_name)
```

---

## ✨ 准备好开始了吗？

告诉我你已经：
1. ✅ 在GitHub创建了空仓库
2. ✅ 想从哪一步开始（EDA / 预处理 / 模型 / 评估）

我会为你创建完整的Jupyter notebook和详细的代码指导！

