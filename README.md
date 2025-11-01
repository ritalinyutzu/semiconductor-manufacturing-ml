# 半導體製造缺陷預測 - 機器學習項目

## 📋 項目概述

這個項目使用機器學習演算法預測半導體製造過程中的產品缺陷。通過分析590個生產參數，我們構建了一個分類模型來識別產品是否會通過品質檢驗。

**核心目標**：
- 🎯 預測產品 Pass/Fail（通過/失敗）
- 📊 識別對缺陷影響最大的特徵
- 🚀 提供可部署的機器學習模型

---

## 📊 數據集

- **來源**：UCI SECOM Dataset (Kaggle)
- **樣本數**：1,567 條製造記錄
- **特徵數**：592 列（590 個生產參數 + 時間戳 + 結果）
- **類別分布**：不均衡的 Pass/Fail 比例（93.36% / 6.64%）
- **訓練/測試**：80% / 20% 分割

---

## 🏗️ 項目結構
```
semiconductor-manufacturing-ml/
├── data/                      # 數據資料夾
│   ├── raw/                   # 原始數據
│   └── processed/             # 處理後的數據 (CSV + PKL)
├── scripts/                   # Python 腳本
│   ├── 01_eda.py             # 探索性數據分析
│   ├── 02_preprocessing.py    # 數據預處理
│   ├── 03_modeling.py         # 模型訓練
│   └── 04_results.py          # 結果分析
├── src/                       # Python 源代碼模塊
│   ├── preprocessing.py       # 預處理類
│   ├── models.py              # 模型訓練類
│   ├── evaluate.py            # 評估類
│   ├── utils.py               # 工具函數
│   └── __init__.py
├── results/                   # 結果輸出
│   ├── models/                # 訓練好的模型 (.pkl)
│   ├── figures/               # 可視化圖表 (.png)
│   └── *.txt                  # 分析報告
├── requirements.txt           # Python 依賴
├── run_all.py                 # 主運行腳本
├── README_ZH.md               # 繁體中文說明
├── VSCODE_GUIDE.md            # VSCode 使用指南
└── .gitignore                 # Git 忽略檔案
```

---

## 🚀 快速開始

### 前置條件
- Python 3.11+
- 從 Kaggle 下載 `secom.csv`

### 第 1 步：安裝依賴
```bash
pip install -r requirements.txt
```

### 第 2 步：準備數據

1. 訪問 [Kaggle UCI SECOM Dataset](https://www.kaggle.com/datasets/paresh2047/uci-semcom)
2. 下載 `secom.csv`
3. 放入 `data/raw/` 資料夾

### 第 3 步：一鍵運行所有分析
```bash
python run_all.py
```

等待 20-30 分鐘，所有分析自動完成！

### 第 4 步：查看結果
```
results/
├── figures/           # 8 個可視化圖表
├── models/            # 5 個訓練好的模型
└── *.txt              # 詳細報告
```

---

## 📈 項目成果

### 數據處理
- ✅ 1,567 個樣本的完整分析
- ✅ 缺失值處理（刪除 28 列）
- ✅ 特徵縮放 (StandardScaler)
- ✅ PCA 降維：440 維 → 135 維

### 模型訓練

| 模型 | 測試準確率 | 狀態 |
|------|----------|------|
| **KNN** | **91.40%** | ⭐ 最佳 |
| XGBoost | 91.40% | 並列最佳 |
| 隨機森林 | 89.25% | 良好 |
| 高斯樸素貝葉斯 | 88.17% | 可接受 |
| 邏輯迴歸 | 88.17% | 可接受 |

### 可視化輸出

- ✅ 類別分布圖
- ✅ 缺失值熱力圖
- ✅ PCA 方差解釋
- ✅ 模型性能對比
- ✅ 混淆矩陣（所有模型）
- ✅ ROC 曲線
- ✅ 特徵重要性分析
- ✅ 項目總結可視化

---

## 🔬 核心技術棧

| 工具 | 版本 | 用途 |
|------|------|------|
| Python | 3.11 | 編程語言 |
| Pandas | 2.1.0+ | 數據處理 |
| NumPy | 1.26.0+ | 數值計算 |
| Scikit-learn | 1.3.0+ | 機器學習 |
| XGBoost | 2.0.0+ | 梯度提升 |
| Matplotlib | 3.8.0+ | 可視化 |
| Seaborn | 0.13.0+ | 統計可視化 |

---

## 📚 項目阶段

### 階段 1：探索性數據分析 (EDA)
- 數據形狀和統計信息
- 缺失值詳細分析（538 列有缺失值）
- 特徵分布可視化
- 類別不均衡分析（14.07:1）

### 階段 2：數據預處理
- 處理缺失值（刪除 > 50% 缺失的列）
- 移除 122 個零方差特徵
- 特徵縮放 (StandardScaler)
- PCA 降維（保留 95% 方差）
- 訓練/測試集分割（80/20）

### 階段 3：模型訓練
- KNN：K-最近鄰
- 高斯樸素貝葉斯：概率模型
- 邏輯迴歸：線性模型
- 隨機森林：集成學習
- XGBoost：梯度提升

使用 5 折交叉驗證評估所有模型

### 階段 4：結果分析
- 模型性能對比
- 混淆矩陣和 ROC 曲線
- 分類報告和評估指標
- 最佳模型選擇和部署

---

## 💡 關鍵成就

| 指標 | 結果 |
|------|------|
| 最佳模型準確率 | 91.40% ⭐ |
| PCA 維度壓縮比 | 440 → 135 (3.26:1) |
| 訓練樣本數 | 368 |
| 測試樣本數 | 93 |
| 生成的圖表數 | 8 個 |
| 訓練的模型數 | 5 個 |
| 總運行時間 | ~20-30 分鐘 |

---

## 🎓 學習收穫

通過這個項目，你將學到：

✅ **數據科學工作流程**
- 真實數據集的處理方法
- 缺失值和異常值處理
- 特徵工程和標準化

✅ **機器學習技術**
- 5 種分類演算法的實現
- 模型評估和選擇
- 交叉驗證和超參數調優

✅ **高級技巧**
- PCA 降維（維度壓縮 3.26 倍）
- 類別不均衡處理
- 特徵重要性分析

✅ **工程實踐**
- 模塊化代碼設計
- Git 版本控制
- 專業的文檔編寫
- 結果可視化和報告

---

## 📖 使用指南

### VSCode 用戶

詳見 `VSCODE_GUIDE.md`
```bash
# 在 VSCode 終端中執行
python run_all.py
```

### 詳細文檔

- `README_ZH.md` - 繁體中文完整說明
- `VSCODE_GUIDE.md` - VSCode 使用指南
- `GETTING_STARTED.md` - 詳細操作指南

---

## 🎯 應用場景

- **生產質檢**：自動化缺陷檢測
- **成本控制**：降低報廢率
- **效率提升**：加快檢驗速度
- **數據驅動**：基於機器學習的決策支持

---

## 📊 項目成果

運行完畢後，你會得到：
```
✅ 8 個高質量的可視化圖表
✅ 5 個訓練好的機器學習模型
✅ 3 份詳細的分析報告
✅ 預處理後的數據集
✅ 完整的項目文檔
```

所有檔案都保存在 `results/` 和 `data/processed/` 資料夾中。

---

## 🤝 貢獻

這是學習項目。歡迎提交 Issue 和 Pull Request！

---

## 📜 許可

MIT License

---

## 👨‍💻 作者

Rita Lin - An AI engineer and project manager
http://ritalinyutzu.vercel.app/

2025 年 | 半導體製造缺陷預測

---

## 📚 參考資源

- [UCI SECOM Dataset](https://www.kaggle.com/datasets/paresh2047/uci-semcom)
- [Kaggle 半導體項目](https://www.kaggle.com/search?q=semiconductor)
- [Scikit-learn 文檔](https://scikit-learn.org/)
- [XGBoost 文檔](https://xgboost.readthedocs.io/)

---

**祝你使用愉快！** 🚀
