"""
工具函数模块
包含数据分析、可视化等辅助函数
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def print_section(text):
    """打印小标题"""
    print(f"\n{'─'*60}")
    print(f"  {text}")
    print(f"{'─'*60}")


def describe_dataset(df):
    """
    描述数据集
    
    Args:
        df: DataFrame
    """
    print_header("📊 数据集基本信息")
    print(f"形状: {df.shape}")
    print(f"列数: {df.shape[1]}")
    print(f"行数: {df.shape[0]}")
    print(f"\n数据类型:\n{df.dtypes.value_counts()}")


def check_class_distribution(df, target_col):
    """
    检查类别分布
    
    Args:
        df: DataFrame
        target_col: 目标列名
    """
    print_section(f"📊 目标类别分布 ({target_col})")
    
    distribution = df[target_col].value_counts().sort_index()
    distribution_pct = df[target_col].value_counts(normalize=True).sort_index() * 100
    
    for class_label in distribution.index:
        count = distribution[class_label]
        pct = distribution_pct[class_label]
        print(f"  Class {class_label}: {count:5d} ({pct:6.2f}%)")
    
    # 计算不均衡比率
    if len(distribution) == 2:
        imbalance_ratio = distribution.max() / distribution.min()
        print(f"\n  不均衡比率: {imbalance_ratio:.2f}:1")


def plot_class_distribution(df, target_col, figsize=(10, 5)):
    """
    绘制类别分布
    
    Args:
        df: DataFrame
        target_col: 目标列名
        figsize: 图表大小
        
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 计数
    ax = axes[0]
    df[target_col].value_counts().sort_index().plot(
        kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4']
    )
    ax.set_title('类别计数')
    ax.set_xlabel('类别')
    ax.set_ylabel('数量')
    ax.set_xticklabels(['Fail (0)', 'Pass (1)'], rotation=0)
    
    # 百分比
    ax = axes[1]
    df[target_col].value_counts(normalize=True).sort_index().plot(
        kind='pie', ax=ax, autopct='%1.1f%%',
        colors=['#FF6B6B', '#4ECDC4'],
        labels=['Fail (0)', 'Pass (1)']
    )
    ax.set_title('类别比例')
    ax.set_ylabel('')
    
    plt.tight_layout()
    return fig


def plot_feature_distributions(df, features, ncols=3, figsize=(15, 10)):
    """
    绘制特征分布
    
    Args:
        df: DataFrame
        features: 特征列表
        ncols: 列数
        figsize: 图表大小
        
    Returns:
        matplotlib figure
    """
    n_features = len(features)
    nrows = (n_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        if df[feature].dtype in ['int64', 'float64']:
            axes[idx].hist(df[feature], bins=30, alpha=0.7, color='skyblue')
            axes[idx].set_title(f'{feature}')
            axes[idx].set_ylabel('频率')
        else:
            axes[idx].text(0.5, 0.5, f'{feature}\n(非数值特征)',
                          ha='center', va='center')
    
    # 移除多余的axes
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(X, figsize=(12, 10)):
    """
    绘制相关性热力图
    
    Args:
        X: 特征矩阵或DataFrame
        figsize: 图表大小
        
    Returns:
        matplotlib figure
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    # 计算相关性矩阵
    corr_matrix = X.corr()
    
    # 只显示相关性最高的特征
    if corr_matrix.shape[0] > 20:
        # 计算与目标变量的相关性（如果是最后一列）
        top_features = corr_matrix.iloc[:, -1].abs().nlargest(20).index
        corr_matrix = corr_matrix.loc[top_features, top_features]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, ax=ax, cbar_kws={'label': '相关系数'})
    ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    分割训练集和测试集
    
    Args:
        X: 特征矩阵
        y: 标签
        test_size: 测试集比例
        random_state: 随机种子
        stratify: 是否分层抽样
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print_section("📊 分割数据")
    
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )
        print("  ✅ 使用分层抽样")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print("  ✅ 使用随机抽样")
    
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def save_model(model, filepath):
    """
    保存模型
    
    Args:
        model: 模型对象
        filepath: 保存路径
    """
    import joblib
    joblib.dump(model, filepath)
    print(f"✅ 模型已保存: {filepath}")


def load_model(filepath):
    """
    加载模型
    
    Args:
        filepath: 模型文件路径
        
    Returns:
        模型对象
    """
    import joblib
    model = joblib.load(filepath)
    print(f"✅ 模型已加载: {filepath}")
    return model


def get_summary_stats(df):
    """
    获取数据统计摘要
    
    Args:
        df: DataFrame
        
    Returns:
        统计信息
    """
    print_section("📊 统计摘要")
    print(df.describe())


def check_data_types(df):
    """
    检查数据类型
    
    Args:
        df: DataFrame
    """
    print_section("📊 数据类型")
    print(df.dtypes)


def identify_outliers_iqr(df, column, multiplier=1.5):
    """
    使用IQR方法识别异常值
    
    Args:
        df: DataFrame
        column: 列名
        multiplier: IQR倍数
        
    Returns:
        异常值索引
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
    
    return outliers, lower_bound, upper_bound


def plot_comparison(data_dict, title='对比图', figsize=(10, 6)):
    """
    绘制对比图
    
    Args:
        data_dict: {标签: 数据} 字典
        title: 标题
        figsize: 图表大小
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(data_dict))
    values = list(data_dict.values())
    labels = list(data_dict.keys())
    
    ax.bar(x_pos, values, color=['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181'])
    ax.set_xlabel('类别')
    ax.set_ylabel('值')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (label, value) in enumerate(data_dict.items()):
        ax.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

