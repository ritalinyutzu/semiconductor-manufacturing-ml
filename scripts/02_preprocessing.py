#!/usr/bin/env python3
"""
第二步：數據預處理
運行: python scripts/02_preprocessing.py
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from src.preprocessing import DataPreprocessor
from src.utils import print_header, print_section, split_data
import warnings

warnings.filterwarnings('ignore')


def main():
    print_header("🛠️ 第二步：數據預處理")
    
    # 步驟1: 載入數據
    print_section("步驟1: 載入數據")
    df = pd.read_csv('data/raw/secom.csv')
    print(f"✅ 數據載入完成: {df.shape}")
    
    # 步驟2: 分離特徵和目標變數
    print_section("步驟2: 分離特徵和目標變數")
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"✅ 特徵形狀: {X.shape}")
    print(f"✅ 目標變數形狀: {y.shape}")
    print(f"   目標類別分布: {dict(y.value_counts().sort_index())}")
    
    # 步驟3: 處理缺失值
    print_section("步驟3: 處理缺失值")
    preprocessor = DataPreprocessor()
    preprocessor.check_missing_values(X)
    X = preprocessor.handle_missing_values(X, strategy='drop', threshold=0.5)
    
    # 同步y
    y = y[X.index]
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    print(f"✅ 處理完成: X={X.shape}, y={y.shape}")
    
    # 步驟4: 只保留數值特徵（移除時間戳等文字欄位）
    print_section("步驟4: 只保留數值特徵")
    X = X.select_dtypes(include=[np.number])
    print(f"✅ 特徵過濾完成: {X.shape}")
    
    # 步驟5: 移除零方差特徵
    print_section("步驟5: 移除零方差特徵")
    X = preprocessor.remove_zero_variance_features(X)
    
    # 步驟6: 分割訓練/測試集
    print_section("步驟6: 分割訓練/測試集")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    
    print(f"✅ 訓練集目標分布: {dict(y_train.value_counts().sort_index())}")
    print(f"✅ 測試集目標分布: {dict(y_test.value_counts().sort_index())}")
    
    # 步驟7: 特徵縮放
    print_section("步驟7: 特徵縮放")
    X_train_scaled, X_test_scaled = preprocessor.scale_features(
        X_train, X_test, fit=True
    )
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"✅ 縮放完成")
    print(f"   訓練集平均值: {X_train_scaled.mean().mean():.4f}")
    print(f"   訓練集標準差: {X_train_scaled.std().mean():.4f}")
    
    # 步驟8: PCA降維
    print_section("步驟8: PCA降維")
    X_train_pca, X_test_pca = preprocessor.apply_pca(
        X_train_scaled, X_test_scaled, variance_ratio=0.95, fit=True
    )
    
    pca_columns = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
    X_train_pca = pd.DataFrame(X_train_pca, columns=pca_columns)
    X_test_pca = pd.DataFrame(X_test_pca, columns=pca_columns)
    
    print(f"✅ PCA降維完成")
    
    # 可視化PCA解釋方差
    print_section("步驟9: 可視化PCA")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.bar(range(1, len(preprocessor.pca.explained_variance_ratio_) + 1),
           preprocessor.pca.explained_variance_ratio_, color='#4ECDC4')
    ax.set_xlabel('主成分')
    ax.set_ylabel('解釋方差比')
    ax.set_title('PCA - 個體解釋方差')
    ax.grid(alpha=0.3, axis='y')
    
    ax = axes[1]
    cumsum = np.cumsum(preprocessor.pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
    ax.axhline(y=0.95, color='green', linestyle='--', label='95% 方差')
    ax.set_xlabel('主成分')
    ax.set_ylabel('累積解釋方差')
    ax.set_title('PCA - 累積解釋方差')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/03_pca_variance.png', dpi=300, bbox_inches='tight')
    print("✅ PCA可視化已保存: results/figures/03_pca_variance.png")
    plt.close()
    
    # 步驟10: 保存數據
    print_section("步驟10: 保存預處理後的數據")
    
    X_train_pca.to_csv('data/processed/X_train_pca.csv', index=False)
    X_test_pca.to_csv('data/processed/X_test_pca.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=['target'])
    y_test.to_csv('data/processed/y_test.csv', index=False, header=['target'])
    
    print("✅ 數據已保存")
    print("   - X_train_pca.csv")
    print("   - X_test_pca.csv")
    print("   - y_train.csv")
    print("   - y_test.csv")
    
    joblib.dump(preprocessor.scaler, 'data/processed/scaler.pkl')
    joblib.dump(preprocessor.pca, 'data/processed/pca.pkl')
    
    print("\n✅ 預處理器已保存")
    print("   - scaler.pkl")
    print("   - pca.pkl")
    
    print_section("預處理摘要")
    
    summary = f"""
📊 預處理完成報告
{'─'*60}

數據處理步驟:
  1. ✅ 處理缺失值
  2. ✅ 只保留數值特徵
  3. ✅ 移除零方差特徵
  4. ✅ 分割訓練/測試集
  5. ✅ 特徵縮放
  6. ✅ PCA降維

降維成果:
  - 原始特徵數: {X_train_scaled.shape[1]}
  - 降維後特徵數: {X_train_pca.shape[1]}
  - 維度壓縮比: {X_train_scaled.shape[1]/X_train_pca.shape[1]:.2f}:1
"""
    
    print(summary)
    
    with open('results/preprocessing_report.txt', 'w', encoding='utf-8') as f:
        f.write(summary)
    print("✅ 報告已保存: results/preprocessing_report.txt")
    
    print("\n" + "="*60)
    print("✅ 數據預處理完成！")
    print("="*60)


if __name__ == '__main__':
    main()