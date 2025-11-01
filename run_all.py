#!/usr/bin/env python3
"""
半導體製造缺陷預測 - 主運行腳本
運行: python run_all.py
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner(text):
    """打印橫幅"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_script(script_name, description):
    """運行單個腳本"""
    print_banner(f"🚀 {description}")
    
    script_path = f"scripts/{script_name}"
    
    if not os.path.exists(script_path):
        print(f"❌ 錯誤: 找不到檔案 {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], cwd=os.getcwd())
        if result.returncode == 0:
            print(f"\n✅ {description} 完成！")
            return True
        else:
            print(f"\n❌ {description} 失敗")
            return False
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        return False


def main():
    print_banner("半導體製造缺陷預測 - 完整項目運行")
    
    print("""
    本腳本將按順序運行所有分析流程:
    
    1️⃣  EDA (探索性數據分析)
    2️⃣  預處理 (數據處理和PCA降維)
    3️⃣  建模 (模型訓練和評估)
    4️⃣  結果 (結果分析和報告生成)
    
    預計總耗時: 20-30 分鐘
    """)
    
    input("\n按 Enter 鍵開始... ")
    
    # 檢查必要的檔案
    print_banner("📋 前置檢查")
    
    required_files = [
        'data/raw/secom.csv',
        'scripts/01_eda.py',
        'scripts/02_preprocessing.py',
        'scripts/03_modeling.py',
        'scripts/04_results.py'
    ]
    
    for file in required_files:
        if file.startswith('data/raw/'):
            # 這個檔案用戶需要自己下載
            if not os.path.exists(file):
                print(f"⚠️  {file} - 請從Kaggle下載")
        else:
            if os.path.exists(file):
                print(f"✅ {file} - 已找到")
            else:
                print(f"❌ {file} - 未找到")
                return
    
    # 檢查Kaggle數據
    if not os.path.exists('data/raw/secom.csv'):
        print("\n❌ 錯誤: 找不到 data/raw/secom.csv")
        print("\n請從Kaggle下載數據:")
        print("  https://www.kaggle.com/datasets/paresh2047/uci-semcom")
        print("\n將 secom.csv 保存到 data/raw/ 目錄")
        return
    
    print("\n✅ 所有檔案已就緒，開始運行...")
    
    # 運行各個腳本
    scripts = [
        ('01_eda.py', '步驟1: 探索性數據分析 (EDA)'),
        ('02_preprocessing.py', '步驟2: 數據預處理'),
        ('03_modeling.py', '步驟3: 模型訓練和評估'),
        ('04_results.py', '步驟4: 結果分析和報告生成')
    ]
    
    success_count = 0
    for script_name, description in scripts:
        if run_script(script_name, description):
            success_count += 1
        else:
            print(f"\n❌ 在 {description} 時發生錯誤")
            print("請檢查上方的錯誤訊息")
            break
    
    # 總結
    print_banner("🎉 項目完成總結")
    
    if success_count == len(scripts):
        print("✅ 所有分析完成！\n")
        print("📊 生成的檔案:")
        print("  ✅ results/figures/ - 所有可視化圖表")
        print("  ✅ results/models/ - 訓練好的模型")
        print("  ✅ data/processed/ - 預處理後的數據")
        print("  ✅ results/*.txt - 分析報告\n")
        
        print("🚀 接下來:")
        print("  1. 查看 results/ 目錄下的所有檔案")
        print("  2. 將項目推送到GitHub")
        print("  3. 更新你的作品集")
        print("  4. 準備項目演講\n")
        
        print("📖 檢查結果:")
        print("  - results/FINAL_REPORT.txt - 完整報告")
        print("  - results/COMPLETION_CHECKLIST.txt - 完成清單")
        print("  - results/figures/00_project_summary.png - 項目總結")
    else:
        print(f"⚠️  部分分析未完成 ({success_count}/{len(scripts)})")
        print("請查看上方的錯誤訊息")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
