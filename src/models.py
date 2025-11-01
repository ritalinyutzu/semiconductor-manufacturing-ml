#!/usr/bin/env python3
"""
模型訓練和評估類
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = None
        self.trained_models = {}
        self.results = None
    
    def train_knn(self, X_train, y_train):
        """訓練KNN模型"""
        model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        X_train = np.ascontiguousarray(X_train)
        model.fit(X_train, y_train)
        return model
    
    def train_naive_bayes(self, X_train, y_train):
        """訓練高斯樸素貝葉斯"""
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """訓練邏輯迴歸"""
        model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train, y_train):
        """訓練隨機森林"""
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def train_xgboost(self, X_train, y_train):
        """訓練XGBoost"""
        model = XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """評估單個模型"""
        # 確保數據格式正確
        X_train = np.ascontiguousarray(X_train)
        X_test = np.ascontiguousarray(X_test)
        
        # 訓練集預測
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        # 測試集預測
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # 其他指標
        precision = precision_score(y_test, y_test_pred, zero_division=0)
        recall = recall_score(y_test, y_test_pred, zero_division=0)
        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        
        # 交叉驗證
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        # ROC-AUC
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        return {
            '模型': model_name,
            '訓練準確率': train_acc,
            '測試準確率': test_acc,
            '精確率': precision,
            '召回率': recall,
            'F1分數': f1,
            'ROC-AUC': roc_auc,
            '交叉驗證平均': cv_scores.mean(),
            '交叉驗證標準差': cv_scores.std()
        }
    
    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test):
        """訓練和評估所有模型"""
        print("\n" + "="*60)
        print("🤖 模型訓練和評估")
        print("="*60 + "\n")
        
        models_config = [
            ('KNN', self.train_knn),
            ('高斯樸素貝葉斯', self.train_naive_bayes),
            ('邏輯迴歸', self.train_logistic_regression),
            ('隨機森林', self.train_random_forest),
            ('XGBoost', self.train_xgboost)
        ]
        
        results = []
        
        for model_name, train_func in models_config:
            print(f"🚀 訓練 {model_name}...")
            try:
                model = train_func(X_train, y_train)
                self.trained_models[model_name] = model
                
                result = self.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
                results.append(result)
                
                print(f"✅ {model_name} 訓練完成")
                print(f"   測試準確率: {result['測試準確率']:.4f}")
                print()
            except Exception as e:
                print(f"❌ {model_name} 訓練失敗: {str(e)}")
                print()
        
        self.results = pd.DataFrame(results)
        
        # 選擇最佳模型
        best_idx = self.results['測試準確率'].idxmax()
        self.best_model_name = self.results.loc[best_idx, '模型']
        self.best_model = self.trained_models[self.best_model_name]
        
        print("="*60)
        print(f"✅ 最佳模型: {self.best_model_name}")
        print(f"   測試準確率: {self.results.loc[best_idx, '測試準確率']:.4f}")
        print("="*60 + "\n")
        
        return self.results, self.trained_models
    
    def get_best_model(self):
        """獲取最佳模型"""
        return self.best_model_name, self.best_model
    
    def get_prediction_probabilities(self, model_name, X_test):
        """獲取預測概率"""
        model = self.trained_models[model_name]
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_test)[:, 1]
        return None