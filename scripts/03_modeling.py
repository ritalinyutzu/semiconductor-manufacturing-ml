#!/usr/bin/env python3
"""
第三步：模型訓練和評估
運行: python scripts/03_modeling.py
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.models import ModelTrainer
from src.evaluate import ModelEvaluator
from src.utils import print_header, print_section
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')


def main():
    print_header("🤖 第三步：模型訓練和評估")
    
    # 步驟1: 載入預處理後的數據
    print_section("步驟1: 載入預處理後的數據")
    
    X_train = pd.read_csv('data/processed/X_train_pca.csv')
    X_test = pd.read_csv('data/processed/X_test_pca.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]
    y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]
    
    print(f"✅ 數據載入完成")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test: {X_test.shape}")
    print(f"   y_train: {y_train.shape}")
    print(f"   y_test: {y_test.shape}")
    
    # 步驟2: 驗證數據
    print_section("步驟2: 驗證數據")
    print(f"訓練集目標分布: {dict(y_train.value_counts().sort_index())}")
    print(f"測試集目標分布: {dict(y_test.value_counts().sort_index())}")
    print(f"特徵維度: {X_train.shape[1]}")
    
    # 步驟3: 轉換標籤（[-1, 1] 改為 [0, 1]）
    print_section("步驟3: 標籤轉換")
    y_train = (y_train + 1) // 2  # -1 -> 0, 1 -> 1
    y_test = (y_test + 1) // 2
    print(f"✅ 標籤已轉換")
    print(f"   訓練集: {dict(y_train.value_counts().sort_index())}")
    print(f"   測試集: {dict(y_test.value_counts().sort_index())}")
    
    # 轉換為numpy數組並確保是連續的
    X_train = np.ascontiguousarray(X_train.values, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test.values, dtype=np.float64)
    y_train = y_train.values
    y_test = y_test.values
    
    # 步驟4: 訓練所有模型
    print_section("步驟4: 訓練所有模型")
    
    trainer = ModelTrainer(random_state=42)
    results_df, trained_models = trainer.train_and_evaluate_all(
        X_train, y_train, X_test, y_test
    )
    
    # 步驟5: 可視化模型對比
    print_section("步驟5: 模型性能對比")
    
    evaluator = ModelEvaluator()
    fig = evaluator.compare_models(results_df, figsize=(14, 5))
    plt.savefig('results/figures/04_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 對比圖已保存: results/figures/04_model_comparison.png")
    plt.close()
    
    # 步驟6: 獲取最佳模型
    print_section("步驟6: 選擇最佳模型")
    
    best_model_name, best_model = trainer.get_best_model()
    
    # 轉換為連續數組進行預測
    X_test_cont = np.ascontiguousarray(X_test, dtype=np.float64)
    y_pred = best_model.predict(X_test_cont)
    y_proba = trainer.get_prediction_probabilities(best_model_name, X_test_cont)
    
    print(f"✅ 最佳模型: {best_model_name}")
    
    # 步驟7: 最佳模型的詳細評估
    print_section("步驟7: 最佳模型詳細評估")
    
    evaluator.print_classification_report(y_test, y_pred, best_model_name)
    
    # 步驟8: 混淆矩陣
    print_section("步驟8: 混淆矩陣")
    
    fig = evaluator.plot_confusion_matrix(y_test, y_pred, best_model_name, figsize=(8, 6))
    plt.savefig(f'results/figures/05_confusion_matrix_{best_model_name}.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩陣已保存")
    plt.close()
    
    # 步驟9: ROC曲線
    print_section("步驟9: ROC曲線")
    
    if y_proba is not None:
        fig = evaluator.plot_roc_curve(y_test, y_proba, best_model_name, figsize=(8, 6))
        plt.savefig(f'results/figures/06_roc_curve_{best_model_name}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"✅ ROC曲線已保存")
        plt.close()
    else:
        print("⚠️ 最佳模型不支持概率預測")
    
    # 步驟10: 特徵重要性
    print_section("步驟10: 特徵重要性分析")
    
    feature_names = [f'PC{i+1}' for i in range(X_train.shape[1])]
    feature_importance = evaluator.get_feature_importance_from_model(
        best_model, feature_names
    )
    
    if feature_importance:
        fig = evaluator.plot_feature_importance(feature_importance, figsize=(12, 8))
        plt.savefig(f'results/figures/07_feature_importance_{best_model_name}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"✅ 特徵重要性圖已保存")
        plt.close()
    else:
        print("⚠️ 模型不支持特徵重要性提取")
    
    # 步驟11: 所有模型的混淆矩陣
    print_section("步驟11: 所有模型的混淆矩陣對比")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    model_names = list(trained_models.keys())
    for idx, model_name in enumerate(model_names[:4]):
        model = trained_models[model_name]
        y_pred_temp = model.predict(X_test_cont)
        cm = confusion_matrix(y_test, y_pred_temp)
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['失敗', '成功'], yticklabels=['失敗', '成功'])
        ax.set_title(f'{model_name}')
        ax.set_ylabel('真實')
        ax.set_xlabel('預測')
    
    plt.tight_layout()
    plt.savefig('results/figures/08_all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ 所有混淆矩陣已保存")
    plt.close()
    
    # 步驟12: 保存最佳模型
    print_section("步驟12: 保存最佳模型")
    
    model_path = f'results/models/best_model_{best_model_name}.pkl'
    joblib.dump(best_model, model_path)
    print(f"✅ 最佳模型已保存: {model_path}")
    
    # 保存所有模型
    for model_name, model in trained_models.items():
        path = f'results/models/model_{model_name.replace(" ", "_")}.pkl'
        joblib.dump(model, path)
        print(f"✅ {model_name} 已保存")
    
    # 步驟13: 生成最終總結報告
    print_section("步驟13: 最終總結報告")
    
    summary_report = evaluator.create_summary_report(
        best_model_name, y_test, y_pred, y_proba
    )
    print(summary_report)
    
    # 保存報告
    with open('results/modeling_summary.txt', 'w', encoding='utf-8') as f:
        f.write("\n" + results_df.to_string(index=False) + "\n")
        f.write(summary_report)
    
    print("✅ 總結報告已保存: results/modeling_summary.txt")
    
    # 步驟14: 模型資訊總結
    print_section("🎉 項目完成總結")
    
    final_summary = f"""
📊 半導體製造缺陷預測 - 最終報告
{'─'*60}

✅ 項目完成！

最佳模型: {best_model_name}
  - 測試準確率: {results_df[results_df['模型']==best_model_name]['測試準確率'].values[0]:.4f}

模型性能排名:
"""
    
    for idx, row in results_df.sort_values('測試準確率', ascending=False).iterrows():
        final_summary += f"  {idx+1}. {row['模型']}: {row['測試準確率']:.4f}\n"
    
    final_summary += f"""
已生成檔案:
  ✅ models/ - 訓練好的模型 (.pkl)
  ✅ figures/ - 可視化圖表
  ✅ modeling_summary.txt - 詳細報告

🎉 模型訓練完成！
"""
    
    print(final_summary)
    
    with open('results/figures/final_summary.txt', 'w', encoding='utf-8') as f:
        f.write(final_summary)
    
    print("✅ 最終總結已保存: results/figures/final_summary.txt")
    
    print("\n" + "="*60)
    print("✅ 模型訓練完成！")
    print("="*60)
    print("\n🚀 下一步: python scripts/04_results.py")


if __name__ == '__main__':
    main()