#!/usr/bin/env python3
"""
Use psycopg2 to directly connect to Supabase PostgreSQL database
This bypasses the PostgREST API layer entirely
"""
from dotenv import load_dotenv
import os
import sys


def main():
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    project_ref = supabase_url.split("//")[1].split(".")[0] if supabase_url else None

    print(f"Project Ref: {project_ref}")
    print()
    print("⚠️  需要 DATABASE PASSWORD（不是 API Key）")
    print("請到 Supabase Dashboard -> Settings -> Database -> Connection String")
    print("找到 'Password' 欄位的密碼")
    print()

    try:
        import psycopg2
        print("✅ psycopg2 installed")
    except ImportError:
        print("❌ psycopg2 not installed")
        return

    db_password = os.getenv("SUPABASE_DB_PASSWORD")
    if not db_password:
        print("\n請在 .env 檔案中加入: SUPABASE_DB_PASSWORD=你的資料庫密碼")
        return


if __name__ == "__main__":
    main()
