#!/usr/bin/env python3
"""
Improved Backtesting Analysis with Better Metrics
"""
import os
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("=== 回測結果分析 ===\n")

# Fetch all verified predictions
response = supabase.table('predictions').select('*').not_.is_('actual_price', 'null').execute()
data = response.data

if not data:
    print("沒有已驗證的預測資料！")
    exit(0)

df = pd.DataFrame(data)

# Filter out extreme outliers (likely data errors)
df = df[abs(df['percentage_error']) < 200]  # 過濾誤差 > 200% 的異常值

print(f"總共分析 {len(df)} 筆預測（已過濾極端異常值）\n")

# Calculate meaningful metrics
for model in df['model_name'].unique():
    model_df = df[df['model_name'] == model]
    
    # 1. 方向準確度（最重要！）
    correct_direction = 0
    for _, row in model_df.iterrows():
        predicted_up = row['predicted_price'] > row['current_price']
        actual_up = row['actual_price'] > row['current_price']
        if predicted_up == actual_up:
            correct_direction += 1
    
    direction_accuracy = (correct_direction / len(model_df)) * 100
    
    # 2. 平均絕對誤差百分比（MAPE）
    mape = model_df['percentage_error'].abs().mean()
    
    # 3. 預測在 ±10% 範圍內的比例
    within_10_pct = (model_df['percentage_error'].abs() <= 10).sum() / len(model_df) * 100
    
    # 4. 預測在 ±20% 範圍內的比例
    within_20_pct = (model_df['percentage_error'].abs() <= 20).sum() / len(model_df) * 100
    
    print(f"📊 {model} 模型表現：")
    print(f"  測試數量: {len(model_df)} 筆")
    print(f"  ✅ 方向準確度: {direction_accuracy:.1f}%  ← 最重要指標")
    print(f"  📉 平均絕對誤差: {mape:.1f}%")
    print(f"  🎯 誤差 ≤10%: {within_10_pct:.1f}%")
    print(f"  🎯 誤差 ≤20%: {within_20_pct:.1f}%")
    print()

# Top 10 best predictions
print("🏆 最準確的 10 個預測：")
best = df.nsmallest(10, 'percentage_error', keep='first')[['ticker', 'model_name', 'current_price', 'predicted_price', 'actual_price', 'percentage_error']]
for _, row in best.iterrows():
    print(f"  {row['ticker']} ({row['model_name']}): 預測 {row['predicted_price']:.2f}, 實際 {row['actual_price']:.2f}, 誤差 {row['percentage_error']:.2f}%")

print("\n⚠️ 最差的 10 個預測：")
worst = df.nlargest(10, lambda x: abs(x['percentage_error']), keep='first')[['ticker', 'model_name', 'current_price', 'predicted_price', 'actual_price', 'percentage_error']]
for _, row in worst.iterrows():
    print(f"  {row['ticker']} ({row['model_name']}): 預測 {row['predicted_price']:.2f}, 實際 {row['actual_price']:.2f}, 誤差 {row['percentage_error']:.2f}%")
