# 🎯 VSCode 使用指南

## 📌 在VSCode中運行本項目

### 第1步：打開項目

1. 打開 VSCode
2. **File → Open Folder**
3. 選擇 `semiconductor-manufacturing-ml` 資料夾
4. 點擊 **Select Folder**

---

### 第2步：安裝依賴

1. 打開終端: **Ctrl + `** (或 **View → Terminal**)
2. 運行命令:
   ```bash
   pip install -r requirements.txt
   ```

---

### 第3步：準備數據

1. 從 [Kaggle](https://www.kaggle.com/datasets/paresh2047/uci-semcom) 下載 `secom.csv`
2. 創建資料夾: `data/raw/`
3. 將 `secom.csv` 放入 `data/raw/` 資料夾
4. 目標路徑應該是: `data/raw/secom.csv`

---

### 第4步：運行項目

#### 方式1：一鍵運行所有腳本 (推薦！)

在 VSCode 終端中執行:
```bash
python run_all.py
```

這將按順序運行所有分析:
- 01_eda.py - EDA分析
- 02_preprocessing.py - 數據預處理
- 03_modeling.py - 模型訓練
- 04_results.py - 結果分析

#### 方式2：個別運行腳本

如果要逐步運行，在 VSCode 終端中執行:

```bash
# 第一步：EDA分析
python scripts/01_eda.py

# 第二步：數據預處理
python scripts/02_preprocessing.py

# 第三步：模型訓練
python scripts/03_modeling.py

# 第四步：結果分析
python scripts/04_results.py
```

#### 方式3：在VSCode中直接運行

1. 打開任何 `.py` 檔案 (例如 `scripts/01_eda.py`)
2. 按右上角的 **▶ Run** 按鈕
3. 或按 **Ctrl + F5** (需要安裝Python擴展)

---

## 📊 預期結果

運行完成後，你會看到:

```
results/
├── figures/
│   ├── 00_project_summary.png
│   ├── 01_class_distribution.png
│   ├── 02_missing_values.png
│   ├── 03_pca_variance.png
│   ├── 04_model_comparison.png
│   ├── 05_confusion_matrix_*.png
│   ├── 06_roc_curve_*.png
│   ├── 07_feature_importance_*.png
│   └── 08_all_confusion_matrices.png
│
├── models/
│   ├── best_model_XGBoost.pkl
│   ├── model_KNN.pkl
│   ├── model_Naive_Bayes.pkl
│   ├── model_Logistic_Regression.pkl
│   └── model_Random_Forest.pkl
│
├── FINAL_REPORT.txt
├── COMPLETION_CHECKLIST.txt
├── eda_report.txt
├── preprocessing_report.txt
└── modeling_summary.txt

data/processed/
├── X_train_pca.csv
├── X_test_pca.csv
├── y_train.csv
├── y_test.csv
├── scaler.pkl
└── pca.pkl
```

---

## 🎨 VSCode 推薦設置

### 安裝有用的擴展

1. **Python** - Microsoft
   - 提供Python支持
   - 代碼補全和調試

2. **Pylance** - Microsoft
   - 快速代碼分析
   - 更好的類型檢查

3. **Jupyter** - Microsoft
   - 如果你想使用Jupyter的功能

4. **GitHub Copilot** (可選)
   - 代碼建議

### 安裝方法

1. 打開 **Extensions** (Ctrl + Shift + X)
2. 搜索擴展名稱
3. 點擊 **Install**

---

## 💻 常用VSCode快捷鍵

| 快捷鍵 | 功能 |
|--------|------|
| `Ctrl + `` | 打開/關閉終端 |
| `Ctrl + F5` | 運行Python檔案 |
| `F5` | 調試模式運行 |
| `Ctrl + /` | 註釋/取消註釋 |
| `Ctrl + Shift + F` | 在項目中搜索 |
| `Ctrl + P` | 快速打開檔案 |
| `Ctrl + ,` | 打開設置 |

---

## 🐛 常見問題

### Q: 運行時提示 "找不到模塊"

**A:** 確保已安裝所有依賴:
```bash
pip install -r requirements.txt
```

### Q: 提示 "找不到 secom.csv"

**A:** 檢查檔案位置:
- 正確: `data/raw/secom.csv`
- 檢查確實下載了檔案

### Q: 終端中有編碼錯誤

**A:** 在 VSCode 設置中將編碼改為 UTF-8:
1. 右下角點擊編碼格式
2. 選擇 **UTF-8**

### Q: 圖表無法顯示

**A:** 圖表會自動保存到 `results/figures/` 目錄，直接打開 PNG 檔案查看

---

## 📂 項目文件說明

```
semiconductor-manufacturing-ml/
│
├── scripts/                     # 所有運行腳本
│   ├── 01_eda.py               # 數據探索分析
│   ├── 02_preprocessing.py      # 數據預處理
│   ├── 03_modeling.py           # 模型訓練
│   └── 04_results.py            # 結果分析
│
├── src/                         # Python模塊
│   ├── preprocessing.py         # 預處理類
│   ├── models.py                # 模型類
│   ├── evaluate.py              # 評估類
│   └── utils.py                 # 工具函數
│
├── data/                        # 數據目錄
│   ├── raw/                     # 原始數據 (你需要放入secom.csv)
│   └── processed/               # 預處理後的數據
│
├── results/                     # 輸出結果
│   ├── figures/                 # 圖表輸出
│   └── models/                  # 模型輸出
│
├── run_all.py                   # 主運行腳本 (推薦!)
├── requirements.txt             # 依賴列表
└── README.md                    # 項目說明
```

---

## 🚀 推薦的運行流程

1. **準備環境**
   ```bash
   pip install -r requirements.txt
   ```

2. **準備數據**
   - 下載 `secom.csv` 並放入 `data/raw/`

3. **一鍵運行**
   ```bash
   python run_all.py
   ```

4. **查看結果**
   - 打開 `results/` 資料夾查看所有輸出

5. **上傳到GitHub**
   ```bash
   git add .
   git commit -m "Complete ML project"
   git push origin main
   ```

---

## 📖 更多資源

- [Python官方文檔](https://docs.python.org/)
- [Scikit-learn文檔](https://scikit-learn.org/)
- [Pandas文檔](https://pandas.pydata.org/)
- [VSCode Python指南](https://code.visualstudio.com/docs/python/python-tutorial)

---

## 💬 需要幫助?

查看以下檔案:
- `README.md` - 項目說明
- `GETTING_STARTED.md` - 詳細指南
- `results/FINAL_REPORT.txt` - 完整報告

祝你使用愉快！🎉
