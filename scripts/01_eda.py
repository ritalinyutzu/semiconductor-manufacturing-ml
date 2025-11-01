#!/usr/bin/env python3
"""
第一步：探索性數據分析 (EDA)
運行: python scripts/01_eda.py
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import print_header, print_section, describe_dataset
import warnings

warnings.filterwarnings('ignore')

# 設置風格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def main():
    print_header("📊 第一步：探索性數據分析 (EDA)")
    
    # 步驟1: 載入數據
    print_section("步驟1: 載入數據")
    DATA_PATH = 'data/raw/secom.csv'
    
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"✅ 數據載入成功")
        print(f"   形狀: {df.shape}")
        print(f"   列數: {df.shape[0]:,}")
        print(f"   行數: {df.shape[1]:,}")
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到檔案 {DATA_PATH}")
        print(f"   請確保已從Kaggle下載數據: https://www.kaggle.com/datasets/paresh2047/uci-semcom")
        return
    
    # 步驟2: 基本資訊
    print_section("步驟2: 數據基本資訊")
    describe_dataset(df)
    
    # 步驟3: 檢查缺失值
    print_section("步驟3: 缺失值分析")
    missing_info = pd.DataFrame({
        '列名': df.columns,
        '缺失數': df.isnull().sum(),
        '缺失率%': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('缺失數', ascending=False)
    
    missing_cols = missing_info[missing_info['缺失數'] > 0]
    print(f"\n有缺失值的列數: {len(missing_cols)}")
    
    if len(missing_cols) > 0:
        print("\n缺失值最多的前10列:")
        print(missing_cols.head(10).to_string(index=False))
    else:
        print("✅ 沒有缺失值！")
    
    # 步驟4: 目標變數分析
    print_section("步驟4: 目標變數分析")
    target_col = df.columns[-1]
    print(f"\n目標變數: {target_col}")
    print(f"唯一值: {df[target_col].nunique()}")
    print(f"\n值的分布:")
    
    class_dist = df[target_col].value_counts().sort_index()
    for label, count in class_dist.items():
        pct = count / len(df) * 100
        print(f"  Class {label}: {count:5d} ({pct:6.2f}%)")
    
    # 檢查不均衡
    if len(class_dist) == 2:
        imbalance = class_dist.max() / class_dist.min()
        print(f"\n⚠️  類別不均衡比: {imbalance:.2f}:1")
        if imbalance > 1.5:
            print("   數據存在明顯的類別不均衡！")
    
    # 步驟5: 特徵統計
    print_section("步驟5: 特徵統計")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\n特徵數: {X.shape[1]}")
    print(f"樣本數: {X.shape[0]}")
    
    # 檢查零方差特徵
    variance = X.select_dtypes(include=[np.number]).var()
    zero_var = variance[variance == 0].index.tolist()
    print(f"\n零方差特徵數: {len(zero_var)}")
    
    # 步驟6: 產生可視化
    print_section("步驟6: 產生可視化")
    
    # 類別分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax = axes[0]
    class_dist.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
    ax.set_title('目標變數分布 - 計數')
    ax.set_xlabel('類別')
    ax.set_ylabel('數量')
    ax.set_xticklabels(['失敗 (0)', '成功 (1)'], rotation=0)
    ax.grid(alpha=0.3, axis='y')
    
    ax = axes[1]
    class_dist.plot(kind='pie', ax=ax, autopct='%1.1f%%',
                    colors=['#FF6B6B', '#4ECDC4'],
                    labels=['失敗 (0)', '成功 (1)'])
    ax.set_title('目標變數分布 - 比例')
    ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('results/figures/01_class_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 圖表已保存: results/figures/01_class_distribution.png")
    plt.close()
    
    # 缺失值可視化
    if len(missing_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        top_missing = missing_info.head(20)
        ax.barh(range(len(top_missing)), top_missing['缺失率%'], color='#FF6B6B')
        ax.set_yticks(range(len(top_missing)))
        ax.set_yticklabels([f"Col{i}" for i in range(len(top_missing))])
        ax.set_xlabel('缺失率 (%)')
        ax.set_title('缺失值最多的前20列')
        ax.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('results/figures/02_missing_values.png', dpi=300, bbox_inches='tight')
        print("✅ 圖表已保存: results/figures/02_missing_values.png")
        plt.close()
    
    # 步驟7: 數據品質報告
    print_section("步驟7: 數據品質報告")
    
    report = f"""
📊 數據品質總結
{'─'*60}
總樣本數: {len(df):,}
總特徵數: {X.shape[1]:,}

數據品質:
  - 缺失值: {df.isnull().sum().sum():,} 個 ({(df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100):.2f}%)
  - 零方差特徵: {len(zero_var)} 個
  - 數值列: {len(X.select_dtypes(include=[np.number]).columns)}
  - 物件列: {len(X.select_dtypes(include=['object']).columns)}

目標變數:
  - 類別數: {y.nunique()}
  - 類別不均衡: {(class_dist.max()/class_dist.min()):.2f}:1
  - 正類比例: {(y.sum()/len(y)*100):.2f}%

下一步:
  1. ✅ 處理缺失值
  2. ✅ 移除零方差特徵
  3. ✅ 特徵縮放
  4. ✅ PCA降維
  5. ✅ 模型訓練
"""
    
    print(report)
    
    # 保存報告
    with open('results/eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✅ 報告已保存: results/eda_report.txt")
    
    print("\n" + "="*60)
    print("✅ EDA分析完成！")
    print("="*60)
    print("\n🚀 下一步: python scripts/02_preprocessing.py")


if __name__ == '__main__':
    main()
