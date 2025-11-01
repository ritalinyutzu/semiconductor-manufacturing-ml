#!/usr/bin/env python3
"""
模型評估類
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self):
        pass
    
    def print_classification_report(self, y_test, y_pred, model_name):
        """打印分類報告"""
        print(f"\n{model_name} - 詳細分類報告")
        print("="*60)
        print(classification_report(y_test, y_pred, zero_division=0))
    
    def plot_confusion_matrix(self, y_test, y_pred, model_name, figsize=(8, 6)):
        """繪製混淆矩陣"""
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['失敗', '成功'], yticklabels=['失敗', '成功'])
        ax.set_title(f'{model_name} - 混淆矩陣')
        ax.set_ylabel('真實')
        ax.set_xlabel('預測')
        return fig
    
    def plot_roc_curve(self, y_test, y_proba, model_name, figsize=(8, 6)):
        """繪製ROC曲線"""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='隨機分類器')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('偽正率 (FPR)')
        ax.set_ylabel('真正率 (TPR)')
        ax.set_title(f'{model_name} - ROC曲線')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        return fig
    
    def compare_models(self, results_df, figsize=(14, 5)):
        """對比多個模型"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 準確率對比
        ax = axes[0]
        colors = ['#FF6B6B' if x != results_df['測試準確率'].max() else '#4ECDC4' 
                  for x in results_df['測試準確率']]
        ax.bar(results_df['模型'], results_df['測試準確率'], color=colors)
        ax.set_ylabel('測試準確率')
        ax.set_title('模型測試準確率對比')
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # F1分數對比
        ax = axes[1]
        ax.bar(results_df['模型'], results_df['F1分數'], color=colors)
        ax.set_ylabel('F1分數')
        ax.set_title('模型F1分數對比')
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        return fig
    
    def get_feature_importance_from_model(self, model, feature_names):
        """從模型獲取特徵重要性"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(feature_names, importances))
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_[0])
            return dict(zip(feature_names, coef))
        return None
    
    def plot_feature_importance(self, feature_dict, top_n=20, figsize=(12, 8)):
        """繪製特徵重要性"""
        sorted_features = sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(features)), importances, color='#4ECDC4')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('重要性')
        ax.set_title(f'前{top_n}個最重要的特徵')
        ax.grid(alpha=0.3, axis='x')
        return fig
    
    def create_summary_report(self, best_model_name, y_test, y_pred, y_proba):
        """創建總結報告"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        report = f"""
📊 最佳模型評估報告
{'─'*60}

模型名稱: {best_model_name}

性能指標:
  - 測試準確率: {accuracy_score(y_test, y_pred):.4f}
  - 精確率: {precision_score(y_test, y_pred, zero_division=0):.4f}
  - 召回率: {recall_score(y_test, y_pred, zero_division=0):.4f}
  - F1分數: {f1_score(y_test, y_pred, zero_division=0):.4f}

分類報告:
{classification_report(y_test, y_pred, zero_division=0)}
"""
        return report