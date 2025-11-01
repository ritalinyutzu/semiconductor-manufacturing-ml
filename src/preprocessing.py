"""
数据预处理模块
处理缺失值、特征缩放、特征选择等
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理类"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
        self.n_components = None
        
    def load_data(self, filepath):
        """
        加载数据集
        
        Args:
            filepath: 数据文件路径
            
        Returns:
            pandas DataFrame
        """
        print(f"📂 加载数据: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 数据加载完成! 形状: {df.shape}")
        return df
    
    def check_missing_values(self, df):
        """
        检查缺失值
        
        Args:
            df: 输入DataFrame
            
        Returns:
            缺失值统计
        """
        print("\n📊 缺失值分析:")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            '缺失数': missing,
            '缺失百分比': missing_pct
        }).sort_values('缺失数', ascending=False)
        
        missing_df = missing_df[missing_df['缺失数'] > 0]
        if len(missing_df) == 0:
            print("✅ 没有缺失值！")
        else:
            print(missing_df.head(10))
        
        return missing_df
    
    def handle_missing_values(self, df, strategy='drop', threshold=0.5):
        """
        处理缺失值
        
        Args:
            df: 输入DataFrame
            strategy: 'drop' 或 'mean'
            threshold: 缺失值百分比阈值
            
        Returns:
            处理后的DataFrame
        """
        print(f"\n🔧 处理缺失值 (策略: {strategy}, 阈值: {threshold})...")
        
        # 计算缺失百分比
        missing_pct = df.isnull().sum() / len(df)
        
        # 删除缺失超过阈值的列
        cols_to_drop = missing_pct[missing_pct > threshold].index
        df = df.drop(columns=cols_to_drop)
        print(f"  - 删除了 {len(cols_to_drop)} 列(缺失 > {threshold*100}%)")
        
        # 处理剩余缺失值
        if strategy == 'drop':
            df = df.dropna()
            print(f"  - 删除了含有缺失值的行")
        elif strategy == 'mean':
            df = df.fillna(df.mean())
            print(f"  - 用均值填充缺失值")
        
        print(f"✅ 处理后形状: {df.shape}")
        return df
    
    def remove_zero_variance_features(self, df, exclude_cols=None):
        """
        移除零方差特征（无信息的列）
        
        Args:
            df: 输入DataFrame
            exclude_cols: 需要排除的列（如目标列）
            
        Returns:
            处理后的DataFrame
        """
        print("\n🗑️ 移除零方差特征...")
        exclude_cols = exclude_cols or []
        variance = df.drop(columns=exclude_cols).select_dtypes(include=[np.number]).var()
        zero_var_features = variance[variance == 0].index.tolist()
        
        if zero_var_features:
            df = df.drop(columns=zero_var_features)
            print(f"  - 删除了 {len(zero_var_features)} 个零方差特征")
        else:
            print("  - 没有零方差特征")
        
        print(f"✅ 处理后形状: {df.shape}")
        return df
    
    def scale_features(self, X_train, X_test=None, fit=True):
        """
        特征缩放
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            fit: 是否拟合scaler
            
        Returns:
            缩放后的特征
        """
        print("\n📈 特征缩放 (StandardScaler)...")
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        print(f"✅ 训练集缩放完成")
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            print(f"✅ 测试集缩放完成")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def apply_pca(self, X_train, X_test=None, variance_ratio=0.95, fit=True):
        """
        应用PCA降维
        
        Args:
            X_train: 训练集特征
            X_test: 测试集特征
            variance_ratio: 保留的方差比例
            fit: 是否拟合PCA
            
        Returns:
            降维后的特征
        """
        if fit:
            print(f"\n📉 PCA降维 (保留方差: {variance_ratio*100}%)...")
            
            # 先用所有特征拟合以确定最优组件数
            pca_temp = PCA()
            pca_temp.fit(X_train)
            
            # 计算需要的组件数
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_ratio) + 1
            
            # 创建最终的PCA对象
            self.pca = PCA(n_components=n_components)
            X_train_pca = self.pca.fit_transform(X_train)
            
            self.n_components = n_components
            variance_explained = self.pca.explained_variance_ratio_.sum()
            
            print(f"  - 原始特征数: {X_train.shape[1]}")
            print(f"  - 降维后特征数: {n_components}")
            print(f"  - 解释方差比: {variance_explained:.2%}")
            print(f"✅ PCA拟合完成")
        else:
            X_train_pca = self.pca.transform(X_train)
        
        if X_test is not None:
            X_test_pca = self.pca.transform(X_test)
            return X_train_pca, X_test_pca
        
        return X_train_pca
    
    def get_feature_importance_pca(self):
        """
        获取PCA特征重要性
        
        Returns:
            特征重要性DataFrame
        """
        if self.pca is None:
            print("❌ PCA还未拟合!")
            return None
        
        loadings = self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        importance = np.abs(loadings).mean(axis=1)
        
        importance_df = pd.DataFrame({
            '特征': [f'PC{i+1}' for i in range(self.n_components)],
            '重要性': importance
        }).sort_values('重要性', ascending=False)
        
        return importance_df


class FeatureSelector:
    """特征选择类"""
    
    @staticmethod
    def select_by_variance_threshold(X, threshold=0.01):
        """
        通过方差阈值选择特征
        
        Args:
            X: 特征矩阵
            threshold: 方差阈值
            
        Returns:
            选中的特征列表
        """
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"✅ 通过方差阈值保留 {len(selected_features)}/{len(X.columns)} 特征")
        
        return selected_features
    
    @staticmethod
    def select_by_correlation(X, target=None, threshold=0.9):
        """
        通过相关性阈值移除多重共线性特征
        
        Args:
            X: 特征矩阵
            target: 目标列
            threshold: 相关性阈值
            
        Returns:
            处理后的特征矩阵
        """
        corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_selected = X.drop(columns=to_drop)
        
        print(f"✅ 移除 {len(to_drop)} 个高度相关的特征")
        return X_selected

