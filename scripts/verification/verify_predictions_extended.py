#!/usr/bin/env python3
"""
驗證 predictions 表格的新欄位，並清理 dual_strategy_predictions 表格
"""
import os
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def verify_and_cleanup():
    """驗證新欄位並清理舊表"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        print("❌ 錯誤: SUPABASE_URL 或 SUPABASE_SERVICE_KEY 未設定")
        return False
    
    try:
        supabase: Client = create_client(url, key)
        print(f"✅ 成功連接到 Supabase\n")
        
        # 1. 驗證 predictions 表格的新欄位
        print("=" * 60)
        print("1️⃣  驗證 predictions 表格新欄位...")
        print("=" * 60)
        
        test_data = {
            "index_name": "TEST_VERIFY",
            "model_name": "TEST",
            "strategy_type": "雙重符合",
            "ticker": "VERIFY.TW",
            "current_price": 100.0,
            "predicted_price": 105.0,
            "potential": 5.0,
            "ma5": 99.0,
            "ma10": 98.0,
            "ma60": 97.0,
            "ma120": 96.0,
            "ma250": 95.0,
            "pullback_type": "MA60回調",
            "pe": 15.0,
            "pb": 2.0,
            "forward_pe": 14.5,
            "period": "6mo",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            result = supabase.table("predictions").insert(test_data).execute()
            test_id = result.data[0]['id']
            print("✅ 測試數據插入成功")
            print(f"   插入 ID: {test_id}")
            
            # 查詢驗證
            query = supabase.table("predictions").select("*").eq("id", test_id).execute()
            data = query.data[0]
            
            print("\n✅ 驗證欄位內容:")
            print(f"   ├─ strategy_type: {data.get('strategy_type')}")
            print(f"   ├─ ma5: {data.get('ma5')}")
            print(f"   ├─ ma10: {data.get('ma10')}")
            print(f"   ├─ ma60: {data.get('ma60')}")
            print(f"   ├─ ma120: {data.get('ma120')}")
            print(f"   ├─ ma250: {data.get('ma250')}")
            print(f"   ├─ pullback_type: {data.get('pullback_type')}")
            print(f"   ├─ pe: {data.get('pe')}")
            print(f"   ├─ pb: {data.get('pb')}")
            print(f"   └─ forward_pe: {data.get('forward_pe')}")
            
            # 清理測試數據
            supabase.table("predictions").delete().eq("id", test_id).execute()
            print("\n✅ 測試數據已清理")
            
        except Exception as e:
            error_msg = str(e).lower()
            if "column" in error_msg and "does not exist" in error_msg:
                print(f"❌ 欄位不存在: {str(e)}")
                print("💡 請確認已在 Supabase 執行 supabase_add_columns.sql")
                return False
            else:
                print(f"❌ 插入測試失敗: {str(e)}")
                return False
        
        print()
        
        # 2. 檢查並刪除 dual_strategy_predictions 表格
        print("=" * 60)
        print("2️⃣  清理舊的 dual_strategy_predictions 表格...")
        print("=" * 60)
        
        try:
            # 先檢查表是否存在
            check = supabase.table("dual_strategy_predictions").select("*").limit(1).execute()
            data_count = len(check.data)
            
            print(f"⚠️  發現 dual_strategy_predictions 表格（有 {data_count} 筆資料）")
            print("\n請在 Supabase SQL Editor 執行以下 SQL 刪除:")
            print("-" * 60)
            print("DROP TABLE IF EXISTS dual_strategy_predictions CASCADE;")
            print("-" * 60)
            print("\n或者如果想保留數據，可以先備份:")
            print("-- 1. 匯出數據")
            print("-- 2. 執行 DROP TABLE")
            
        except Exception as e:
            if "does not exist" in str(e).lower():
                print("✅ dual_strategy_predictions 表格不存在（已清理或從未創建）")
            else:
                print(f"ℹ️  無法檢查 dual_strategy_predictions: {str(e)}")
        
        print()
        
        # 3. 統計現有數據
        print("=" * 60)
        print("3️⃣  統計 predictions 表格數據...")
        print("=" * 60)
        
        try:
            all_data = supabase.table("predictions").select("strategy_type, model_name").execute()
            
            if all_data.data:
                from collections import Counter
                
                # 統計 strategy_type
                strategy_counts = Counter([d.get('strategy_type') or 'NULL (舊資料)' for d in all_data.data])
                
                print(f"✅ predictions 表格共有 {len(all_data.data)} 筆記錄:")
                for strategy, count in strategy_counts.items():
                    print(f"   ├─ {strategy}: {count} 筆")
            else:
                print("📊 predictions 表格目前為空")
                
        except Exception as e:
            print(f"⚠️  統計失敗: {str(e)}")
        
        print()
        print("=" * 60)
        print("🎉 驗證完成！")
        print("=" * 60)
        print("✅ predictions 表格已擴展，包含所有雙軌策略欄位")
        print("✅ 可以執行 python main_dual_strategy.py")
        print("✅ 回測功能可以正常使用（數據在同一張表）")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 驗證過程發生錯誤: {str(e)}")
        return False

if __name__ == "__main__":
    verify_and_cleanup()
