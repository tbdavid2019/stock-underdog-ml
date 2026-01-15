#!/usr/bin/env python3
"""
在 Supabase 的 predictions 表格新增雙軌策略欄位
"""
import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def add_columns():
    """在 predictions 表格新增欄位"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not key:
        print("❌ 錯誤: SUPABASE_URL 或 SUPABASE_SERVICE_KEY 未設定")
        return False
    
    try:
        supabase: Client = create_client(url, key)
        print(f"✅ 成功連接到 Supabase: {url}\n")
        
        # 檢查 predictions 表是否存在
        try:
            result = supabase.table("predictions").select("*").limit(1).execute()
            print(f"✅ predictions 表格存在（目前有 {len(result.data)} 筆資料可見）\n")
        except Exception as e:
            print(f"❌ predictions 表格不存在: {str(e)}")
            print("💡 請先執行 supabase_schema.sql 建立基本表格")
            return False
        
        print("=" * 60)
        print("⚠️  重要提示")
        print("=" * 60)
        print("Supabase Python client 不支持執行 ALTER TABLE")
        print()
        print("請在 Supabase Dashboard 執行以下 SQL:")
        print()
        print("1. 登入 https://app.supabase.com")
        print("2. 選擇專案 → SQL Editor → New Query")
        print("3. 複製貼上 supabase_add_columns.sql 的內容")
        print("4. 點擊 'Run' 執行")
        print()
        print("或者直接複製以下 SQL:")
        print("=" * 60)
        print()
        
        with open('supabase_add_columns.sql', 'r', encoding='utf-8') as f:
            content = f.read()
            # 只顯示 ALTER TABLE 部分
            lines = [l for l in content.split('\n') if l.strip() and not l.strip().startswith('--')]
            for line in lines[:30]:  # 只顯示前30行
                print(line)
        
        print()
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ 連接失敗: {str(e)}")
        return False

if __name__ == "__main__":
    add_columns()
