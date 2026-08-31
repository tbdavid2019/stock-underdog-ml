"""
Database management for stock prediction application.
Provides persistence layer for Supabase with strict JSON validation.
"""
import datetime
import os
import sys
from typing import List, Tuple, Optional, Any, Dict
import pandas as pd

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError as e:
    print(f"Supabase Import Failed: {e}", file=sys.stderr)
    HAS_SUPABASE = False
except Exception as e:
    print(f"Supabase Import Unexpected Error: {e}", file=sys.stderr)
    HAS_SUPABASE = False

from core.config import config
from json_safety import sanitize_json_value, validate_json_payload
from logger import logger


class SupabaseManager:
    """Manager for Supabase connection and operations"""
    
    def __init__(self):
        self.enabled = False
        url = config.supabase_url
        
        # Prioritize service_role key (full admin access) over publishable key
        key = os.getenv("SUPABASE_SERVICE_KEY") or config.supabase_key
        
        if not HAS_SUPABASE:
            logger.warning("⚠️ 警告: 未安裝 supabase 套件")
            return
            
        if not url or not key:
            logger.warning("⚠️ 警告: Supabase URL 或 Key 未設定")
            return
            
        try:
            self.client: Client = create_client(url, key)
            self.enabled = True
            logger.info("✅ Supabase 連接初始化成功")
        except Exception as e:
            logger.error(f"❌ Supabase 連線失敗: {str(e)}")
            self.enabled = False

    def save_predictions(self, index_name: str, predictions: List[Tuple], model_name: str, period: str):
        """
        Save prediction results to Supabase 'predictions' table
        
        Args:
            index_name: Name of stock index (e.g., '台灣50')
            predictions: List of tuples (ticker, potential, current, predicted)
            model_name: Name of the model (e.g., 'LSTM')
            period: Training period (e.g., '6mo')
        """
        if not self.enabled:
            return

        data = []
        timestamp = datetime.datetime.now().isoformat()
        
        for p in predictions:
            tk = p[0]
            pot = float(p[1].iloc[0]) if isinstance(p[1], pd.Series) else float(p[1])
            cur = float(p[2].iloc[0]) if isinstance(p[2], pd.Series) else float(p[2])
            pred = float(p[3].iloc[0]) if isinstance(p[3], pd.Series) else float(p[3])

            record = {
                "index_name": index_name,
                "model_name": model_name,
                "ticker": tk,
                "potential": pot,
                "current_price": cur,
                "predicted_price": pred,
                "period": period,
                "timestamp": timestamp
            }
            data.append(record)
            
        try:
            data = sanitize_json_value(data)
            validate_json_payload(data)
            self.client.table("predictions").insert(data).execute()
            logger.info(f"✅ 成功寫入 {len(data)} 筆預測結果到 Supabase (Model: {model_name})")
        except ValueError as e:
            logger.error(f"❌ Supabase 寫入前 JSON 檢查失敗，已取消寫入: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Supabase 寫入失敗: {str(e)}")

    def save_dual_strategy_results(self, index_name: str, results: dict, period: str = "6mo"):
        """
        保存雙軌策略結果到 Supabase predictions 表格
        
        Args:
            index_name: 指數名稱
            results: 包含 xuantie_results, lstm_results, overlap_results
            period: 數據週期
        """
        if not self.enabled:
            return
        
        timestamp = datetime.datetime.now().isoformat()
        all_data = []
        
        # 1. 保存玄鐵策略結果
        xuantie_df = results.get('xuantie_results', pd.DataFrame())
        if isinstance(xuantie_df, pd.DataFrame) and not xuantie_df.empty:
            for idx, row in xuantie_df.iterrows():
                record = {
                    "index_name": index_name,
                    "model_name": "玄鐵重劍",
                    "strategy_type": "玄鐵重劍",
                    "ticker": row['ticker'],
                    "current_price": float(row['current_price']),
                    "predicted_price": None,
                    "potential": None,
                    "ma5": float(row.get('ma5')) if row.get('ma5') and not pd.isna(row.get('ma5')) else None,
                    "ma10": float(row.get('ma10')) if row.get('ma10') and not pd.isna(row.get('ma10')) else None,
                    "ma60": float(row.get('ma60')) if row.get('ma60') and not pd.isna(row.get('ma60')) else None,
                    "ma120": float(row.get('ma120')) if row.get('ma120') and not pd.isna(row.get('ma120')) else None,
                    "ma250": float(row.get('ma250')) if row.get('ma250') and not pd.isna(row.get('ma250')) else None,
                    "pullback_type": row.get('pullback_type'),
                    "pe": float(row.get('pe')) if row.get('pe') and not pd.isna(row.get('pe')) else None,
                    "pb": float(row.get('pb')) if row.get('pb') and not pd.isna(row.get('pb')) else None,
                    "forward_pe": float(row.get('forward_pe')) if row.get('forward_pe') and not pd.isna(row.get('forward_pe')) else None,
                    "ev_ebitda": float(row.get('ev_ebitda')) if row.get('ev_ebitda') and not pd.isna(row.get('ev_ebitda')) else None,
                    "period": period,
                    "timestamp": timestamp
                }
                all_data.append(record)
        
        # 2. 保存 LSTM 預測結果
        lstm_results = results.get('lstm_results', [])
        for result in lstm_results:
            record = {
                "index_name": index_name,
                "model_name": "LSTM",
                "strategy_type": "LSTM預測",
                "ticker": result['ticker'],
                "current_price": float(result['current_price']),
                "predicted_price": float(result['predicted_price']),
                "potential": float(result['potential']),
                "ma5": None,
                "ma10": None,
                "ma60": None,
                "ma120": None,
                "ma250": None,
                "pullback_type": None,
                "pe": float(result.get('pe')) if result.get('pe') and not pd.isna(result.get('pe')) else None,
                "pb": float(result.get('pb')) if result.get('pb') and not pd.isna(result.get('pb')) else None,
                "forward_pe": float(result.get('forward_pe')) if result.get('forward_pe') and not pd.isna(result.get('forward_pe')) else None,
                "ev_ebitda": float(result.get('ev_ebitda')) if result.get('ev_ebitda') and not pd.isna(result.get('ev_ebitda')) else None,
                "period": period,
                "timestamp": timestamp
            }
            all_data.append(record)
        
        # 3. 保存雙重符合結果
        overlap_df = results.get('overlap_results', pd.DataFrame())
        if isinstance(overlap_df, pd.DataFrame) and not overlap_df.empty:
            for idx, row in overlap_df.iterrows():
                record = {
                    "index_name": index_name,
                    "model_name": "雙重符合",
                    "strategy_type": "雙重符合",
                    "ticker": row['ticker'],
                    "current_price": float(row['current_price']),
                    "predicted_price": float(row['predicted_price']),
                    "potential": float(row['lstm_potential']),
                    "ma5": float(row.get('ma5')) if row.get('ma5') and not pd.isna(row.get('ma5')) else None,
                    "ma10": float(row.get('ma10')) if row.get('ma10') and not pd.isna(row.get('ma10')) else None,
                    "ma60": float(row.get('ma60')) if row.get('ma60') and not pd.isna(row.get('ma60')) else None,
                    "ma120": float(row.get('ma120')) if row.get('ma120') and not pd.isna(row.get('ma120')) else None,
                    "ma250": float(row.get('ma250')) if row.get('ma250') and not pd.isna(row.get('ma250')) else None,
                    "pullback_type": row.get('pullback_type'),
                    "pe": float(row.get('pe')) if row.get('pe') and not pd.isna(row.get('pe')) else None,
                    "pb": float(row.get('pb')) if row.get('pb') and not pd.isna(row.get('pb')) else None,
                    "forward_pe": float(row.get('forward_pe')) if row.get('forward_pe') and not pd.isna(row.get('forward_pe')) else None,
                    "ev_ebitda": float(row.get('ev_ebitda')) if row.get('ev_ebitda') and not pd.isna(row.get('ev_ebitda')) else None,
                    "period": period,
                    "timestamp": timestamp
                }
                all_data.append(record)
        
        if not all_data:
            logger.info("⚠️ 無數據需要保存到 Supabase")
            return
        
        try:
            all_data = sanitize_json_value(all_data)
            validate_json_payload(all_data)
            self.client.table("predictions").insert(all_data).execute()
            logger.info(f"✅ 成功寫入 {len(all_data)} 筆雙軌策略結果到 Supabase predictions 表")
        except ValueError as e:
            logger.error(f"❌ Supabase 寫入前 JSON 檢查失敗，已取消寫入: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Supabase 寫入失敗: {str(e)}")
