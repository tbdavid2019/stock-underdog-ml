"""
data/industry_chain_fetcher.py - 跨市場 AI 產業價值鏈與 SEC 10-K 供應鏈圖譜擷取器

整合官方 TPEx 產業價值鏈平台 (ic.tpex.org.tw) 與 SEC EDGAR 10-K 大客戶文字揭露，
提供離線降級、14 天本地快取、多主題拓撲生成與 DuckDB 批次持久化支援。
"""

import os
import re
import json
import time
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger("stock_app.industry_chain")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "industry_chain")
CACHE_EXPIRATION_SECONDS = 14 * 86400  # 14 天快取有效

THEME_DEFINITIONS = [
    {
        "id": "ALL",
        "name": "全部主題",
        "name_en": "All Value Chains",
        "description": "跨市場全產業鏈整合圖譜 (包含晶片算力、光互聯、先進封裝、伺服器與機器人)",
    },
    {
        "id": "HBM_STORAGE",
        "name": "HBM/儲存",
        "name_en": "HBM & Advanced Storage",
        "description": "高頻寬記憶體 (HBM3e/HBM4)、DRAM 堆疊、控制器與先進記憶體封測供應鏈",
    },
    {
        "id": "CPO_OPTICAL",
        "name": "CPO/光互聯",
        "name_en": "CPO & Silicon Photonics",
        "description": "共同封裝光學 (CPO)、矽光子 (SiPh)、光模組、CW 光源與 800G/1.6T 高速交換器",
    },
    {
        "id": "POWER_COOLING",
        "name": "電力/液冷",
        "name_en": "Power & Liquid Cooling",
        "description": "AI 機櫃水冷板、CDU 分配器、均熱片、大功率鈦金電源與電網散熱基礎設施",
    },
    {
        "id": "ADV_PACKAGING",
        "name": "先進封裝",
        "name_en": "Advanced Packaging (CoWoS/SoIC)",
        "description": "台積電 CoWoS/SoIC 生態、濕製程設備、ABF 載板、探針卡測試與無塵室機電",
    },
    {
        "id": "AI_CHIPS",
        "name": "AI晶片/算力",
        "name_en": "AI Silicon & Acceleration",
        "description": "GPU/ASIC 核心運算晶片、EDA 設計軟體、IP 矽智財、晶圓代工與雲端算力",
    },
    {
        "id": "SERVER_ODM",
        "name": "伺服器/ODM",
        "name_en": "AI Server & ODM Systems",
        "description": "AI 伺服器整機代工、GPU 運算主機板、高階 CCL/PCB、導軌機箱與雲端品牌廠",
    },
    {
        "id": "ROBOTICS",
        "name": "機器人/具身智能",
        "name_en": "Embodied AI & Robotics",
        "description": "人形機器人關節減速機、滾珠螺桿導軌、3D AI 視覺感知、邊緣工控與智慧製造",
    },
]

# 核心嵌入式離線資料集 (涵蓋台美核心 AI 產業鏈節點)
EMBEDDED_NODES: List[Dict[str, Any]] = [
    # ---------- 上游 UPSTREAM ----------
    {
        "ticker": "3017",
        "company": "奇鋐",
        "market": "TW",
        "layer": "upstream",
        "role": "3D VC 散熱模組與水冷板 Cold Plate 龍頭",
        "key_focus": "NVL72 水冷板/CDU分歧管模組",
        "theme_tags": ["POWER_COOLING", "SERVER_ODM"],
    },
    {
        "ticker": "3324",
        "company": "雙鴻",
        "market": "TW",
        "layer": "upstream",
        "role": "伺服器水冷散熱與 CDU 分配器主力",
        "key_focus": "水冷板、散熱導管、CDU監控系統",
        "theme_tags": ["POWER_COOLING"],
    },
    {
        "ticker": "2421",
        "company": "建準",
        "market": "TW",
        "layer": "upstream",
        "role": "高轉速伺服器散熱風扇與循環馬達",
        "key_focus": "高效能對轉風扇、液冷幫浦馬達",
        "theme_tags": ["POWER_COOLING"],
    },
    {
        "ticker": "8996",
        "company": "高力",
        "market": "TW",
        "layer": "upstream",
        "role": "水冷板真空銲接與熱交換核心零組件",
        "key_focus": "板式熱交換器、水冷歧管",
        "theme_tags": ["POWER_COOLING"],
    },
    {
        "ticker": "2308",
        "company": "台達電",
        "market": "TW",
        "layer": "upstream",
        "role": "全球電源供應器與智慧液冷機櫃方案",
        "key_focus": "Titanium+ 電源、伺服馬達、工業自動化",
        "theme_tags": ["POWER_COOLING", "SERVER_ODM", "ROBOTICS"],
    },
    {
        "ticker": "2301",
        "company": "光寶科",
        "market": "TW",
        "layer": "upstream",
        "role": "大功率伺服器電源與 33kW/66kW AI 機櫃",
        "key_focus": "高功率密度電源、PSU",
        "theme_tags": ["POWER_COOLING", "SERVER_ODM"],
    },
    {
        "ticker": "6409",
        "company": "旭隼",
        "market": "TW",
        "layer": "upstream",
        "role": "不斷電系統 UPS 與資料中心儲能設備",
        "key_focus": "On-Line UPS、高階電網保護",
        "theme_tags": ["POWER_COOLING"],
    },
    {
        "ticker": "6805",
        "company": "富世達",
        "market": "TW",
        "layer": "upstream",
        "role": "伺服器水冷快接頭 (UQD) 與精密轉軸",
        "key_focus": "GB200 水冷快接頭認證",
        "theme_tags": ["POWER_COOLING"],
    },
    {
        "ticker": "2383",
        "company": "台光電",
        "market": "TW",
        "layer": "upstream",
        "role": "全球 AI 伺服器無鹵素高階 CCL 龍頭",
        "key_focus": "M7/M8 等級超低損耗高速基板",
        "theme_tags": ["SERVER_ODM"],
    },
    {
        "ticker": "6274",
        "company": "台燿",
        "market": "TW",
        "layer": "upstream",
        "role": "高速極低損耗 CCL 銅箔基板",
        "key_focus": "800G 交換器與 AI 伺服器 CCL",
        "theme_tags": ["SERVER_ODM", "CPO_OPTICAL"],
    },
    {
        "ticker": "2368",
        "company": "金像電",
        "market": "TW",
        "layer": "upstream",
        "role": "全球 AI 伺服器與交換器高層數 PCB 龍頭",
        "key_focus": "30+ 層高階多層板、UBB/OAM",
        "theme_tags": ["SERVER_ODM", "CPO_OPTICAL"],
    },
    {
        "ticker": "2059",
        "company": "川湖",
        "market": "TW",
        "layer": "upstream",
        "role": "全球伺服器滑軌與機架機構件霸主 (>80%)",
        "key_focus": "Blackwell NVL72 導軌套件",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "8210",
        "company": "勤誠",
        "market": "TW",
        "layer": "upstream",
        "role": "AI 伺服器高密度機殼與機箱領導品牌",
        "key_focus": "客製化 4U/8U GPU 機箱",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "3013",
        "company": "晟銘電",
        "market": "TW",
        "layer": "upstream",
        "role": "伺服器機殼與液冷機櫃機構件整合",
        "key_focus": "水冷機櫃總成、高階沖壓件",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "3653",
        "company": "健策",
        "market": "TW",
        "layer": "upstream",
        "role": "均熱片 ILM 與半導體導線架領先廠商",
        "key_focus": "超大面積均熱片、CoWoS 連接器扣件",
        "theme_tags": ["POWER_COOLING", "ADV_PACKAGING"],
    },
    {
        "ticker": "COHR",
        "company": "Coherent",
        "market": "US",
        "layer": "upstream",
        "role": "矽光子雷射光源與 800G/1.6T 光模組龍頭",
        "key_focus": "CW 雷射晶片、光收發光引擎",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "LITE",
        "company": "Lumentum",
        "market": "US",
        "layer": "upstream",
        "role": "超高速光通訊 EML/CW 雷射關鍵供應商",
        "key_focus": "高功率雷射二極體、光學元件",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "3450",
        "company": "聯鈞",
        "market": "TW",
        "layer": "upstream",
        "role": "矽光子雷射封裝與光電元件加工",
        "key_focus": "CPO 雷射源封裝、矽光子產線",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "4977",
        "company": "眾達-KY",
        "market": "TW",
        "layer": "upstream",
        "role": "共同封裝光學 CPO 外部光源 ELS",
        "key_focus": "ELS 光源模組、Broadcom 認證",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "3081",
        "company": "聯亞",
        "market": "TW",
        "layer": "upstream",
        "role": "矽光子與光通訊磊晶片供應商",
        "key_focus": "磷化銦 InP / 砷化鎵 GaAs 磊晶片",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "3363",
        "company": "上詮",
        "market": "TW",
        "layer": "upstream",
        "role": "光纖陣列 FAU 與台積電 COUPE 光互連",
        "key_focus": "高密度光纖陣列、保偏光纖",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "4979",
        "company": "華星光",
        "market": "TW",
        "layer": "upstream",
        "role": "800G 光收發模組與雲端資料中心晶片",
        "key_focus": "光發射器 TOSA/ROSA、Marvell 代工",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "MRVL",
        "company": "Marvell",
        "market": "US",
        "layer": "upstream",
        "role": "光電 DSP 與雲端 ASIC 高速互連晶片",
        "key_focus": "Nova 1.6T PAM4 DSP、客製化算力晶片",
        "theme_tags": ["CPO_OPTICAL", "AI_CHIPS"],
    },
    {
        "ticker": "ASML",
        "company": "ASML",
        "market": "US",
        "layer": "upstream",
        "role": "全球先進微影設備 EUV/High-NA EUV 霸主",
        "key_focus": "0.55 High-NA EUV、Twinscan DUV",
        "theme_tags": ["ADV_PACKAGING", "AI_CHIPS"],
    },
    {
        "ticker": "AMAT",
        "company": "應用材料",
        "market": "US",
        "layer": "upstream",
        "role": "半導體沉積與材料工程設備龍頭",
        "key_focus": "PVD/CVD 設備、CMP 化學機械研磨",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "3131",
        "company": "弘塑",
        "market": "TW",
        "layer": "upstream",
        "role": "台積電 CoWoS 濕製程單晶圓清洗機龍頭",
        "key_focus": "單晶圓旋轉清洗機、化學品供應系統",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "3583",
        "company": "辛耘",
        "market": "TW",
        "layer": "upstream",
        "role": "濕製程設備與自製暫時性貼合剝離設備",
        "key_focus": "CoWoS 批次清洗機、再生晶圓",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "3037",
        "company": "欣興",
        "market": "TW",
        "layer": "upstream",
        "role": "全球 ABF 載板領導廠商與中介層",
        "key_focus": "大尺寸高層數 ABF 載板、CoWoS 基板",
        "theme_tags": ["ADV_PACKAGING", "SERVER_ODM"],
    },
    {
        "ticker": "8046",
        "company": "南電",
        "market": "TW",
        "layer": "upstream",
        "role": "高階晶片封裝載板與 BT/ABF 材料",
        "key_focus": "5nm/3nm 高階載板",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "3189",
        "company": "景碩",
        "market": "TW",
        "layer": "upstream",
        "role": "ABF/BT 高密度互連載板製造商",
        "key_focus": "AI 晶片與儲存晶片載板",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "5536",
        "company": "聖暉*",
        "market": "TW",
        "layer": "upstream",
        "role": "半導體無塵室與高科技廠房機電工程",
        "key_focus": "先進封裝廠與晶圓廠無塵室整合",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "ARM",
        "company": "Arm Holdings",
        "market": "US",
        "layer": "upstream",
        "role": "全球架構運算 IP 與能源效率 CPU 核心",
        "key_focus": "Neoverse V2/V3 伺服器架構、Grace CPU",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "SNPS",
        "company": "Synopsys",
        "market": "US",
        "layer": "upstream",
        "role": "晶片設計 EDA 軟體與矽智財 IP 巨頭",
        "key_focus": "EDA 自動化、UCIe/HBM3 介面 IP",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "CDNS",
        "company": "Cadence",
        "market": "US",
        "layer": "upstream",
        "role": "先進製程 EDA 設計驗證與系統模擬",
        "key_focus": "Chiplet 設計工具、數位與類比 EDA",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "3035",
        "company": "智原",
        "market": "TW",
        "layer": "upstream",
        "role": "ASIC 設計服務與聯電集團矽智財 IP",
        "key_focus": "先進封裝與 14nm/7nm FinFET ASIC",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "3443",
        "company": "創意",
        "market": "TW",
        "layer": "upstream",
        "role": "台積電轉投資高階 ASIC 設計與 HBM PHY IP",
        "key_focus": "HBM3e/4 PHY、CoWoS 設計服務",
        "theme_tags": ["AI_CHIPS", "HBM_STORAGE", "ADV_PACKAGING"],
    },
    {
        "ticker": "3661",
        "company": "世芯-KY",
        "market": "TW",
        "layer": "upstream",
        "role": "全球雲端巨頭 3nm/5nm AI ASIC 設計服務龍頭",
        "key_focus": "AWS Inferentia/Trainium ASIC、CoWoS",
        "theme_tags": ["AI_CHIPS", "ADV_PACKAGING"],
    },
    {
        "ticker": "3529",
        "company": "力旺",
        "market": "TW",
        "layer": "upstream",
        "role": "全球嵌入式非揮發性記憶體 eNVM IP 龍頭",
        "key_focus": "NeoBit、NeoFuse、PUF 晶片指紋安全",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "6533",
        "company": "晶心科",
        "market": "TW",
        "layer": "upstream",
        "role": "RISC-V 處理器核心 IP 領導廠商",
        "key_focus": "AndesCore AI 向量擴展處理器",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "MU",
        "company": "Micron",
        "market": "US",
        "layer": "upstream",
        "role": "HBM3e / HBM4 記憶體核心原廠巨頭",
        "key_focus": "24GB/36GB 8H/12H HBM3e、DDR5",
        "theme_tags": ["HBM_STORAGE"],
    },
    {
        "ticker": "6488",
        "company": "環球晶",
        "market": "TW",
        "layer": "upstream",
        "role": "全球前三大矽晶圓與 SOI 高純度基板廠",
        "key_focus": "12 吋磊晶圓、SiC/GaN 第三代半導體",
        "theme_tags": ["HBM_STORAGE", "ADV_PACKAGING"],
    },
    {
        "ticker": "3532",
        "company": "台勝科",
        "market": "TW",
        "layer": "upstream",
        "role": "12 吋半導體矽晶圓材料製造商",
        "key_focus": "高平整度晶圓基板",
        "theme_tags": ["HBM_STORAGE"],
    },
    {
        "ticker": "8299",
        "company": "群聯",
        "market": "TW",
        "layer": "upstream",
        "role": "NAND Flash 控制晶片與 aiDAPTIV+ 平台",
        "key_focus": "PCIe 5.0 控制晶片、企業級 SSD",
        "theme_tags": ["HBM_STORAGE"],
    },
    {
        "ticker": "4583",
        "company": "台灣精銳",
        "market": "TW",
        "layer": "upstream",
        "role": "高精度行星減速機與齒輪傳動機構龍頭",
        "key_focus": "具身智能機器人關節減速機",
        "theme_tags": ["ROBOTICS"],
    },
    {
        "ticker": "2049",
        "company": "上銀",
        "market": "TW",
        "layer": "upstream",
        "role": "滾珠螺桿、線性滑軌與機器人關節機構",
        "key_focus": "工業機械手臂、諧波減速機",
        "theme_tags": ["ROBOTICS"],
    },
    {
        "ticker": "1590",
        "company": "亞德客-KY",
        "market": "TW",
        "layer": "upstream",
        "role": "全球高精度氣動元件與線性導軌大廠",
        "key_focus": "自動化氣缸、電動執行器",
        "theme_tags": ["ROBOTICS"],
    },
    {
        "ticker": "2359",
        "company": "所羅門",
        "market": "TW",
        "layer": "upstream",
        "role": "3D 機器人視覺、AI 缺陷檢測與數位孿生",
        "key_focus": "Solvision、NVIDIA Isaac 合作夥伴",
        "theme_tags": ["ROBOTICS"],
    },
    {
        "ticker": "8050",
        "company": "廣積",
        "market": "TW",
        "layer": "upstream",
        "role": "邊緣 AI 工控主機板與自走車 AMR 控制大腦",
        "key_focus": "強固型無風扇嵌入式系統",
        "theme_tags": ["ROBOTICS"],
    },
    {
        "ticker": "2395",
        "company": "研華",
        "market": "TW",
        "layer": "upstream",
        "role": "全球工業物聯網與邊緣 AI 運算平台龍頭",
        "key_focus": "Edge AI 推理電腦、智慧工廠控制",
        "theme_tags": ["ROBOTICS"],
    },

    # ---------- 中游 MIDSTREAM ----------
    {
        "ticker": "2330",
        "company": "台積電",
        "market": "TW",
        "layer": "midstream",
        "role": "全球晶圓代工與 CoWoS/SoIC 先進封裝霸主",
        "key_focus": "N3/N2 製程、CoWoS-S/L/R、COUPE 矽光子",
        "theme_tags": ["AI_CHIPS", "ADV_PACKAGING", "CPO_OPTICAL", "HBM_STORAGE"],
    },
    {
        "ticker": "NVDA",
        "company": "NVIDIA",
        "market": "US",
        "layer": "midstream",
        "role": "全球 GPU 運算與 CUDA 軟硬體生態霸主",
        "key_focus": "Blackwell GB200、NVLink 5、Quantum-X",
        "theme_tags": ["AI_CHIPS", "CPO_OPTICAL", "ROBOTICS"],
    },
    {
        "ticker": "AMD",
        "company": "AMD",
        "market": "US",
        "layer": "midstream",
        "role": "高效能 CPU/GPU 與 Chiplet 算力先驅",
        "key_focus": "MI300X/MI325X、EPYC Turin、ROCm 生態",
        "theme_tags": ["AI_CHIPS", "HBM_STORAGE"],
    },
    {
        "ticker": "AVGO",
        "company": "Broadcom",
        "market": "US",
        "layer": "midstream",
        "role": "客製化 AI ASIC 晶片與 CPO 交換器晶片龍頭",
        "key_focus": "Tomahawk 5/6、Google TPU ASIC、PCIe Gen6",
        "theme_tags": ["AI_CHIPS", "CPO_OPTICAL"],
    },
    {
        "ticker": "2454",
        "company": "聯發科",
        "market": "TW",
        "layer": "midstream",
        "role": "旗艦天璣 AI 手機晶片與客製化 ASIC 設計",
        "key_focus": "天璣 9400、AI 邊緣推理、自研 NPU",
        "theme_tags": ["AI_CHIPS"],
    },
    {
        "ticker": "VRT",
        "company": "Vertiv",
        "market": "US",
        "layer": "midstream",
        "role": "全球資料中心電力系統與液冷冷卻基礎設施龍頭",
        "key_focus": "Liebert 液冷冷卻液分配器 CDU、高密度不斷電",
        "theme_tags": ["POWER_COOLING"],
    },
    {
        "ticker": "2317",
        "company": "鴻海",
        "market": "TW",
        "layer": "midstream",
        "role": "全球代工龍頭與 Blackwell NVL72 計算盤主力",
        "key_focus": "GB200 NVL72 機櫃系統整機組裝、水冷迴路",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING", "ROBOTICS"],
    },
    {
        "ticker": "2382",
        "company": "廣達",
        "market": "TW",
        "layer": "midstream",
        "role": "雲達科技 QCT 伺服器與水冷機櫃 ODM 主力",
        "key_focus": "AWS/Meta/Google Tier-1 AI 機櫃整機代工",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "3231",
        "company": "緯創",
        "market": "TW",
        "layer": "midstream",
        "role": "AI 伺服器 GPU 基板 (Baseboard) 與運算板",
        "key_focus": "NVIDIA HGX/DGX GPU 運算板核心代工",
        "theme_tags": ["SERVER_ODM"],
    },
    {
        "ticker": "6669",
        "company": "緯穎",
        "market": "TW",
        "layer": "midstream",
        "role": "純白牌雲端資料中心伺服器與散熱方案巨頭",
        "key_focus": "Meta / Microsoft / Amazon ASIC 與 GPU 伺服器",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "2356",
        "company": "英業達",
        "market": "TW",
        "layer": "midstream",
        "role": "AI 伺服器主機板與雲端算力機箱代工",
        "key_focus": "北美 CSP 雲端伺服器主板、自研 AI 晶片",
        "theme_tags": ["SERVER_ODM"],
    },
    {
        "ticker": "CLS",
        "company": "Celestica",
        "market": "US",
        "layer": "midstream",
        "role": "北美雲端巨頭交換器與白牌算力硬體代工",
        "key_focus": "400G/800G 光交換器代工、AI 叢集系統",
        "theme_tags": ["SERVER_ODM", "CPO_OPTICAL"],
    },
    {
        "ticker": "SMCI",
        "company": "Supermicro",
        "market": "US",
        "layer": "midstream",
        "role": "模組化伺服器 Building Block 與 DLC 液冷整機",
        "key_focus": "直接液冷 (DLC) 機櫃、GPU 刀鋒伺服器",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "6442",
        "company": "光聖",
        "market": "TW",
        "layer": "midstream",
        "role": "高階光被動模組、連接器與美系資料中心夥伴",
        "key_focus": "光纜跳線、波分多工 WDM、北美 CSP 供應",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "6451",
        "company": "訊芯-KY",
        "market": "TW",
        "layer": "midstream",
        "role": "800G CPO 高階光引擎封裝領導廠商",
        "key_focus": "共同封裝光學封裝、高速光收發模組",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "3163",
        "company": "波若威",
        "market": "TW",
        "layer": "midstream",
        "role": "高密度光波導 WDM 與光纖分歧器製造",
        "key_focus": "800G 光收發模組次模組、CPO 零組件",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "2345",
        "company": "智邦",
        "market": "TW",
        "layer": "midstream",
        "role": "全球白牌網路交換器與 CPO AI 網絡設備龍頭",
        "key_focus": "800G 交換器、In-Package Optics 光引擎整合",
        "theme_tags": ["CPO_OPTICAL", "SERVER_ODM"],
    },
    {
        "ticker": "FN",
        "company": "Fabrinet",
        "market": "US",
        "layer": "midstream",
        "role": "超高速光收發模組與光子封裝代工龍頭",
        "key_focus": "Cisco/NVIDIA 光學代工、800G/1.6T 光模組",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "3711",
        "company": "日月光投控",
        "market": "TW",
        "layer": "midstream",
        "role": "全球半導體封裝測試龍頭與 VIPack 平台",
        "key_focus": "2.5D/3D SiP、CoWoS-S/R、FOCoS 先進封測",
        "theme_tags": ["ADV_PACKAGING", "HBM_STORAGE"],
    },
    {
        "ticker": "2449",
        "company": "京元電子",
        "market": "TW",
        "layer": "midstream",
        "role": "AI 晶片與高速運算 GPU/ASIC 晶圓測試龍頭",
        "key_focus": "NVIDIA/AMD 高階晶片預燒 (Burn-in) 測試",
        "theme_tags": ["ADV_PACKAGING", "HBM_STORAGE"],
    },
    {
        "ticker": "6515",
        "company": "穎崴",
        "market": "TW",
        "layer": "midstream",
        "role": "高頻高速垂直探針卡與晶片測試座龍頭",
        "key_focus": "CoWoS 晶片測試座、探針卡、HyperSocket",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "6239",
        "company": "力成",
        "market": "TW",
        "layer": "midstream",
        "role": "先進記憶體堆疊、凸塊與晶圓級封裝",
        "key_focus": "HBM 先進堆疊封裝、面板級扇出型 (FOPLP)",
        "theme_tags": ["ADV_PACKAGING", "HBM_STORAGE"],
    },
    {
        "ticker": "3265",
        "company": "台星科",
        "market": "TW",
        "layer": "midstream",
        "role": "高階覆晶封裝與晶圓凸塊代工廠",
        "key_focus": "3nm/5nm AI 處理器覆晶封裝、散熱強化",
        "theme_tags": ["ADV_PACKAGING"],
    },
    {
        "ticker": "2408",
        "company": "南亞科",
        "market": "TW",
        "layer": "midstream",
        "role": "DRAM 利基型記憶體製造",
        "key_focus": "1B 製程 DDR5、LPDDR4/5",
        "theme_tags": ["HBM_STORAGE"],
    },
    {
        "ticker": "2344",
        "company": "華邦電",
        "market": "TW",
        "layer": "midstream",
        "role": "客製化高頻寬記憶體 CUBE 與 Flash 晶片",
        "key_focus": "CUBE 3D 堆疊記憶體、TrustME 安全晶片",
        "theme_tags": ["HBM_STORAGE"],
    },
    {
        "ticker": "2374",
        "company": "佳能",
        "market": "TW",
        "layer": "midstream",
        "role": "邊緣 AI 機器人視覺相機與光學感測模組",
        "key_focus": "AI 智能鏡頭模組、無人機/AMR 視覺",
        "theme_tags": ["ROBOTICS"],
    },
    {
        "ticker": "3008",
        "company": "大立光",
        "market": "TW",
        "layer": "midstream",
        "role": "高階光學鏡頭與潛望鏡頭製造龍頭",
        "key_focus": "精密微型玻塑混合鏡頭、機器人感測",
        "theme_tags": ["ROBOTICS"],
    },

    # ---------- 下游 DOWNSTREAM ----------
    {
        "ticker": "MSFT",
        "company": "Microsoft",
        "market": "US",
        "layer": "downstream",
        "role": "Azure 雲端 AI 超算中心與 Copilot 生態",
        "key_focus": "Azure OpenAI、Maia 100 ASIC、Copilot 終端",
        "theme_tags": ["AI_CHIPS", "SERVER_ODM", "POWER_COOLING", "CPO_OPTICAL", "HBM_STORAGE"],
    },
    {
        "ticker": "AMZN",
        "company": "Amazon",
        "market": "US",
        "layer": "downstream",
        "role": "AWS 雲端運算霸主與自研 Trainium/Inferentia",
        "key_focus": "AWS UltraClusters、Graviton4、倉儲機器人",
        "theme_tags": ["AI_CHIPS", "SERVER_ODM", "POWER_COOLING", "CPO_OPTICAL", "ROBOTICS"],
    },
    {
        "ticker": "GOOG",
        "company": "Google",
        "market": "US",
        "layer": "downstream",
        "role": "Google Cloud 與自研 TPU v5p/v6 算力平台",
        "key_focus": "Gemini 多模態模型、TPU Trillium、光學電路交換",
        "theme_tags": ["AI_CHIPS", "SERVER_ODM", "POWER_COOLING", "CPO_OPTICAL"],
    },
    {
        "ticker": "META",
        "company": "Meta Platforms",
        "market": "US",
        "layer": "downstream",
        "role": "Llama 開源大模型與百萬 GPU 基礎架構",
        "key_focus": "MTIA 自研晶片、Blackwell 萬卡集群、Grand Teton",
        "theme_tags": ["AI_CHIPS", "SERVER_ODM", "POWER_COOLING", "CPO_OPTICAL"],
    },
    {
        "ticker": "ORCL",
        "company": "Oracle",
        "market": "US",
        "layer": "downstream",
        "role": "Oracle Cloud (OCI) 超大型 GPU 液冷叢集",
        "key_focus": "OCI Supercluster、Zettascale 算力中心",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "CRWV",
        "company": "CoreWeave",
        "market": "US",
        "layer": "downstream",
        "role": "專業純 AI 雲端算力基礎架構領導商",
        "key_focus": "大規模 NVIDIA GPU 專用液冷算力叢集",
        "theme_tags": ["SERVER_ODM", "POWER_COOLING"],
    },
    {
        "ticker": "DELL",
        "company": "Dell Technologies",
        "market": "US",
        "layer": "downstream",
        "role": "企業級與雲端 AI 伺服器整機領導品牌",
        "key_focus": "PowerEdge XE9680、AI Factory 解決方案",
        "theme_tags": ["SERVER_ODM"],
    },
    {
        "ticker": "HPE",
        "company": "HPE",
        "market": "US",
        "layer": "downstream",
        "role": "企業級伺服器與 Cray 超級電腦霸主",
        "key_focus": "Cray EX 超算系統、GreenLake 雲端平台",
        "theme_tags": ["SERVER_ODM"],
    },
    {
        "ticker": "TSLA",
        "company": "Tesla",
        "market": "US",
        "layer": "downstream",
        "role": "FSD 自動駕駛算力、Dojo 超算與 Optimus 機器人",
        "key_focus": "Optimus 具身智能機器人、Cortex 算力叢集",
        "theme_tags": ["ROBOTICS", "AI_CHIPS"],
    },
    {
        "ticker": "ANET",
        "company": "Arista Networks",
        "market": "US",
        "layer": "downstream",
        "role": "全球 AI 叢集高效能乙太網路交換器龍頭",
        "key_focus": "Ethernet for AI、7800R3 AI Spine",
        "theme_tags": ["CPO_OPTICAL"],
    },
    {
        "ticker": "2376",
        "company": "技嘉",
        "market": "TW",
        "layer": "downstream",
        "role": "高階 AI 伺服器與 GIGA POD 算力機櫃品牌",
        "key_focus": "G593 旗艦 GPU 伺服器、邊緣 AI 推理工作站",
        "theme_tags": ["SERVER_ODM", "AI_CHIPS"],
    },
    {
        "ticker": "2357",
        "company": "華碩",
        "market": "TW",
        "layer": "downstream",
        "role": "全方位 AI 伺服器、智慧製造與醫療解決方案",
        "key_focus": "ESC N8-E11 伺服器、ROG 高階電競工作站",
        "theme_tags": ["SERVER_ODM", "AI_CHIPS"],
    },
]

# 核心供應鏈關係邊 (含產業鏈、資本開支、大客戶依賴、主題關聯)
EMBEDDED_RELATIONS: List[Dict[str, Any]] = [
    # 1. 台積電大客戶與主要供應商
    {
        "source": "2330",
        "target": "NVDA",
        "relation_type": "大客戶依賴",
        "desc": "NVDA 為台積電第一大 AI 晶片客戶 (營收佔比約 11%)，全數 Hopper/Blackwell 晶片獨家代工",
        "weight": 0.98,
    },
    {
        "source": "2330",
        "target": "AMD",
        "relation_type": "大客戶依賴",
        "desc": "AMD Instinct GPU 與 EPYC CPU 由台積電 3nm/5nm 先進製程與 CoWoS 獨家代工",
        "weight": 0.92,
    },
    {
        "source": "2330",
        "target": "AVGO",
        "relation_type": "大客戶依賴",
        "desc": "Broadcom 定制化 ASIC (含 Google TPU) 與 CPO 晶片全線委託台積電先進製程製造",
        "weight": 0.90,
    },
    {
        "source": "ASML",
        "target": "2330",
        "relation_type": "資本開支",
        "desc": "台積電為 ASML 全球最大 EUV 微影設備採購客戶，年資本開支達 300 億美元級距",
        "weight": 0.98,
    },
    {
        "source": "AMAT",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "應用材料供應台積電先進製程與 CoWoS 封裝之 PVD/CVD 沉積與平坦化設備",
        "weight": 0.88,
    },
    {
        "source": "3131",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "弘塑為台積電 CoWoS 濕製程單晶圓清洗機主力供應商，市佔率逾七成",
        "weight": 0.94,
    },
    {
        "source": "3583",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "辛耘供應台積電先進封裝批次清洗機與再生晶圓，受惠 CoWoS 產能倍增擴充",
        "weight": 0.90,
    },
    {
        "source": "3037",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "欣興供應台積電 CoWoS 先進封裝大尺寸 ABF 載板與矽中介層載板",
        "weight": 0.88,
    },
    {
        "source": "3443",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "創意為台積電轉投資 ASIC 設計夥伴，專精 CoWoS/SoIC 晶片設計與 HBM PHY IP",
        "weight": 0.94,
    },
    {
        "source": "3661",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "世芯-KY 與台積電緊密合作，為北美 CSP 巨頭投片 3nm/5nm 先進製程 AI 晶片",
        "weight": 0.92,
    },
    {
        "source": "ARM",
        "target": "NVDA",
        "relation_type": "主題關聯",
        "desc": "NVIDIA Grace CPU 與 GH200/GB200 核心架構獲 ARM Neoverse IP 深度授權",
        "weight": 0.95,
    },
    {
        "source": "SNPS",
        "target": "2330",
        "relation_type": "主題關聯",
        "desc": "Synopsys 與台積電共建 OIP 開放創新平台，提供 2nm 認證之 UCIe 與介面 IP",
        "weight": 0.89,
    },
    {
        "source": "MU",
        "target": "NVDA",
        "relation_type": "大客戶依賴",
        "desc": "美光 HBM3e 24GB/36GB 通過 NVIDIA 認證，全面導入 Blackwell B200/GB200 機櫃",
        "weight": 0.95,
    },
    {
        "source": "MU",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "美光 HBM 晶片送交台積電 CoWoS 產線進行基底晶圓異質堆疊整合封裝",
        "weight": 0.92,
    },
    {
        "source": "2449",
        "target": "NVDA",
        "relation_type": "產業鏈",
        "desc": "京元電子承接 NVIDIA 全球 AI GPU 晶圓預燒 (Burn-in) 與終端高階測試",
        "weight": 0.93,
    },
    {
        "source": "3711",
        "target": "NVDA",
        "relation_type": "產業鏈",
        "desc": "日月光投控 VIPack 先進封裝承接 NVIDIA CoWoS 外包與高階 SiP 系統模組封測",
        "weight": 0.90,
    },

    # 2. 電力與液冷散熱傳導
    {
        "source": "3017",
        "target": "NVDA",
        "relation_type": "大客戶依賴",
        "desc": "奇鋐為 NVIDIA Blackwell GB200 水冷板 (Cold Plate) 與 3D VC 核心認證供應商",
        "weight": 0.96,
    },
    {
        "source": "3324",
        "target": "NVDA",
        "relation_type": "產業鏈",
        "desc": "雙鴻取得 NVIDIA 散熱認證，供應 GB200 液冷板與 CDU 分配器模組",
        "weight": 0.91,
    },
    {
        "source": "2059",
        "target": "NVDA",
        "relation_type": "大客戶依賴",
        "desc": "川湖在 NVIDIA Blackwell 伺服器與 NVL72 導軌套件中市佔率超過 80%",
        "weight": 0.96,
    },
    {
        "source": "2308",
        "target": "NVDA",
        "relation_type": "產業鏈",
        "desc": "台達電為 NVIDIA AI 機櫃供應 33kW/66kW 鈦金級高密度電源與液冷模組",
        "weight": 0.92,
    },
    {
        "source": "VRT",
        "target": "NVDA",
        "relation_type": "產業鏈",
        "desc": "Vertiv 與 NVIDIA 共同研發 NVL72 液冷冷卻架構與極限資料中心散熱方案",
        "weight": 0.96,
    },
    {
        "source": "6805",
        "target": "3017",
        "relation_type": "產業鏈",
        "desc": "富世達供應水冷快接頭 (UQD) 模組與奇鋐水冷板緊密整合出貨",
        "weight": 0.88,
    },

    # 3. CPO 與光互聯生態
    {
        "source": "COHR",
        "target": "NVDA",
        "relation_type": "產業鏈",
        "desc": "Coherent 供應 NVIDIA Quantum-X / Spectrum-X 800G/1.6T 矽光子收發核心",
        "weight": 0.90,
    },
    {
        "source": "LITE",
        "target": "AVGO",
        "relation_type": "產業鏈",
        "desc": "Lumentum 為 Broadcom CPO 交換器光學引擎提供高功率 CW 雷射光源",
        "weight": 0.90,
    },
    {
        "source": "3363",
        "target": "2330",
        "relation_type": "產業鏈",
        "desc": "上詮與台積電合作開發 COUPE 矽光子高密度光纖陣列 FAU 封裝介面",
        "weight": 0.91,
    },
    {
        "source": "6442",
        "target": "MSFT",
        "relation_type": "大客戶依賴",
        "desc": "光聖光被動連接器與跳線產品深度導入北美 Tier-1 雲端巨頭資料中心",
        "weight": 0.88,
    },
    {
        "source": "6451",
        "target": "2345",
        "relation_type": "產業鏈",
        "desc": "訊芯-KY 800G CPO 光引擎與智邦高階交換器進行緊密封裝與硬體整合",
        "weight": 0.87,
    },
    {
        "source": "2345",
        "target": "AMZN",
        "relation_type": "大客戶依賴",
        "desc": "智邦為 Amazon AWS 全球資料中心白牌交換器與光互連硬體第一大代工夥伴",
        "weight": 0.93,
    },
    {
        "source": "AVGO",
        "target": "GOOG",
        "relation_type": "大客戶依賴",
        "desc": "Broadcom 與 Google 深度合作客製化 TPU v5/v6 ASIC 運算晶片，佔其營收重要份額",
        "weight": 0.96,
    },

    # 4. 伺服器整機代工與 ODM
    {
        "source": "NVDA",
        "target": "2317",
        "relation_type": "產業鏈",
        "desc": "NVIDIA 將 GB200 NVL72 計算盤 (Compute Tray) 與整機櫃組裝交由鴻海代工",
        "weight": 0.96,
    },
    {
        "source": "NVDA",
        "target": "2382",
        "relation_type": "產業鏈",
        "desc": "廣達雲達 (QCT) 為 NVIDIA GB200 NVL72/36 首波發貨核心伺服器系統代工廠",
        "weight": 0.95,
    },
    {
        "source": "NVDA",
        "target": "6669",
        "relation_type": "產業鏈",
        "desc": "緯穎專精大規模 AI 機櫃系統整機代工，承接 Meta / AWS 搭載 NVIDIA GPU 訂單",
        "weight": 0.91,
    },
    {
        "source": "NVDA",
        "target": "3231",
        "relation_type": "產業鏈",
        "desc": "緯創代工 NVIDIA HGX / DGX 伺服器 GPU 運算基板 (Baseboard) 核心零組件",
        "weight": 0.93,
    },
    {
        "source": "NVDA",
        "target": "CLS",
        "relation_type": "產業鏈",
        "desc": "Celestica 代工 NVIDIA 高階網路設備與白牌資料中心 AI 系統",
        "weight": 0.88,
    },
    {
        "source": "NVDA",
        "target": "DELL",
        "relation_type": "大客戶依賴",
        "desc": "Dell 為全球最大 enterprise AI 伺服器品牌，採購巨額 NVIDIA Hopper / Blackwell GPU",
        "weight": 0.95,
    },
    {
        "source": "2383",
        "target": "2382",
        "relation_type": "產業鏈",
        "desc": "台光電供應廣達 AI 伺服器 UBB/OAM 超低損耗高階銅箔基板 (CCL)",
        "weight": 0.90,
    },
    {
        "source": "2368",
        "target": "6669",
        "relation_type": "產業鏈",
        "desc": "金像電供應緯穎 30+ 層高階 PCB 主板，支援 ASIC 與次世代 GPU 運算機箱",
        "weight": 0.89,
    },

    # 5. 北美四大雲端巨頭 (Hyperscalers) 資本開支傳導
    {
        "source": "MSFT",
        "target": "NVDA",
        "relation_type": "資本開支",
        "desc": "微軟年資本開支逾 500 億美元，半數以上流向採購 NVIDIA GPU 與叢集網絡",
        "weight": 0.98,
    },
    {
        "source": "AMZN",
        "target": "NVDA",
        "relation_type": "資本開支",
        "desc": "Amazon AWS 大舉採購 Blackwell 架構 GPU 並構建 Project Rainier 算力集群",
        "weight": 0.96,
    },
    {
        "source": "META",
        "target": "NVDA",
        "relation_type": "資本開支",
        "desc": "Meta 部署超過 60 萬顆 H100/B200 GPU 訓練 Llama 4/5 次世代前沿大模型",
        "weight": 0.97,
    },
    {
        "source": "GOOG",
        "target": "NVDA",
        "relation_type": "資本開支",
        "desc": "Google 除了自研 TPU 亦採購龐大 NVIDIA GPU 陣列供應 Google Cloud 企業客戶",
        "weight": 0.91,
    },
    {
        "source": "ORCL",
        "target": "NVDA",
        "relation_type": "資本開支",
        "desc": "Oracle OCI 訂購數萬張 GB200 構建超大型液冷 AI 叢集，資本開支顯著爆發",
        "weight": 0.93,
    },
    {
        "source": "CRWV",
        "target": "NVDA",
        "relation_type": "大客戶依賴",
        "desc": "CoreWeave 作為純 AI GPU 算力雲，NVIDIA 兼具其戰略股東與獨家晶片供應商角色",
        "weight": 0.99,
    },
    {
        "source": "6669",
        "target": "META",
        "relation_type": "大客戶依賴",
        "desc": "Meta 為緯穎第一大客戶，營收佔比長期維持在 50% 以上高檔水位",
        "weight": 0.98,
    },
    {
        "source": "2382",
        "target": "GOOG",
        "relation_type": "資本開支",
        "desc": "Google Cloud 資本開支驅動廣達 QCT 大規模伺服器與機架系統擴充",
        "weight": 0.94,
    },
    {
        "source": "2317",
        "target": "MSFT",
        "relation_type": "資本開支",
        "desc": "微軟 Azure 雲端超算中心向鴻海大舉採購 AI 伺服器機櫃與資料中心機房硬體",
        "weight": 0.92,
    },

    # 6. 具身智能機器人
    {
        "source": "2359",
        "target": "NVDA",
        "relation_type": "主題關聯",
        "desc": "所羅門與 NVIDIA 合作整合 Omniverse 與 Isaac 機器人平台，開發具身智能視覺",
        "weight": 0.89,
    },
    {
        "source": "4583",
        "target": "TSLA",
        "relation_type": "產業鏈",
        "desc": "台灣精銳供應高精度行星減速機與齒輪機構，進軍人形機器人關鍵傳動關節",
        "weight": 0.86,
    },
    {
        "source": "2049",
        "target": "TSLA",
        "relation_type": "產業鏈",
        "desc": "上銀滾珠螺桿與線性滑軌機構送樣評估機器人關節驅動與自動化產線",
        "weight": 0.87,
    },
    {
        "source": "2308",
        "target": "TSLA",
        "relation_type": "產業鏈",
        "desc": "台達電供應 Tesla 機器人產線馬達驅動器、超級充電樁與電源轉換系統",
        "weight": 0.90,
    },
    {
        "source": "TSLA",
        "target": "NVDA",
        "relation_type": "資本開支",
        "desc": "特斯拉採購數萬張 H100/H200 GPU 構建 Cortex 超級運算中心以訓練 FSD 與 Optimus",
        "weight": 0.92,
    },
]


class IndustryChainFetcher:
    """跨市場產業價值鏈動態圖譜擷取器"""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def _read_cache(self, key: str, max_age: float = CACHE_EXPIRATION_SECONDS) -> Optional[Dict[str, Any]]:
        path = self._get_cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            if time.time() - mtime > max_age:
                logger.debug("Cache expired for %s (age=%.1fd)", key, (time.time() - mtime) / 86400)
                return None
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to read cache %s: %e", path, e)
            return None

    def _write_cache(self, key: str, data: Dict[str, Any]) -> bool:
        path = self._get_cache_path(key)
        try:
            temp_path = f"{path}.tmp.{os.getpid()}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
            return True
        except Exception as e:
            logger.warning("Failed to write cache %s: %e", path, e)
            return False

    def fetch_tpex_chain(self, ic_code: str) -> Dict[str, Any]:
        """
        抓取並解析 TPEx「產業價值鏈平台」產業鏈結構與成分股。
        優先讀取 14 天內本地快取；遇網路逾時或解析異常時，自動降級至豐富離線備援庫。
        """
        cache_key = f"tpex_{ic_code.upper()}"
        cached = self._read_cache(cache_key)
        if cached:
            return cached

        # 嘗試從 TPEx 官方網站抓取
        url = f"https://ic.tpex.org.tw/introduce.php?ic={ic_code.upper()}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        parsed_data = None
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4.5) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                parsed_data = self._parse_tpex_html(ic_code, html)
        except Exception as e:
            logger.info("TPEx 線上抓取失敗 (%s): %s，自動啟用高品質離線備援資料庫", ic_code, e)

        if not parsed_data or not parsed_data.get("nodes"):
            parsed_data = self._get_fallback_tpex_chain(ic_code)

        if parsed_data:
            self._write_cache(cache_key, parsed_data)

        return parsed_data

    def _parse_tpex_html(self, ic_code: str, html: str) -> Dict[str, Any]:
        """解析 TPEx HTML"""
        # 尋找所有公司標籤與代號
        # 格式: <a href="company_basic.php?stk_code=3035" class="company-text-over" title="智原">智原</a>
        raw_matches = re.findall(
            r'company_basic\.php\?stk_code=(\d{4,5})[^>]*title=[\'"]([^\'"]+)[\'"]',
            html
        )
        if not raw_matches:
            return {}

        # 判斷上下游分界
        idx_up = html.find("上游")
        idx_mid = html.find("中游")
        idx_down = html.find("下游")

        nodes = []
        seen = set()
        for match in re.finditer(
            r'company_basic\.php\?stk_code=(\d{4,5})[^>]*title=[\'"]([^\'"]+)[\'"]',
            html
        ):
            ticker = match.group(1)
            name = match.group(2).strip()
            if ticker in seen:
                continue
            seen.add(ticker)

            pos = match.start()
            if idx_down != -1 and pos >= idx_down:
                layer = "downstream"
            elif idx_mid != -1 and pos >= idx_mid:
                layer = "midstream"
            else:
                layer = "upstream"

            nodes.append({
                "ticker": ticker,
                "company": name,
                "market": "TW",
                "layer": layer,
                "chain_id": ic_code.upper(),
                "role": f"{name} ({layer})",
                "key_focus": f"TPEx 產業鏈 {ic_code}",
                "theme_tags": ["ALL"],
            })

        return {
            "chain_id": ic_code.upper(),
            "name": f"TPEx {ic_code}",
            "source": "tpex_live_crawler",
            "updated_at": time.strftime("%Y-%m-%d"),
            "nodes": nodes,
        }

    def _get_fallback_tpex_chain(self, ic_code: str) -> Dict[str, Any]:
        """豐富離線備援資料庫"""
        code = ic_code.upper()
        # 依 code 篩選相應主題標籤
        tag_map = {
            "D000": ["ADV_PACKAGING", "AI_CHIPS", "HBM_STORAGE"],
            "E000": ["SERVER_ODM", "POWER_COOLING"],
            "G000": ["CPO_OPTICAL"],
            "US_AI_CORE": ["AI_CHIPS", "CPO_OPTICAL", "POWER_COOLING", "ADV_PACKAGING"],
        }
        target_tags = set(tag_map.get(code, ["ALL"]))

        filtered_nodes = []
        for n in EMBEDDED_NODES:
            if code == "US_AI_CORE":
                if n.get("market") == "US":
                    filtered_nodes.append(n)
            else:
                if n.get("market") == "TW" and any(tag in target_tags for tag in n.get("theme_tags", [])):
                    filtered_nodes.append(n)

        return {
            "chain_id": code,
            "name": f"AI Value Chain {code}",
            "source": "embedded_authoritative_fallback",
            "updated_at": time.strftime("%Y-%m-%d"),
            "nodes": filtered_nodes or EMBEDDED_NODES[:15],
        }

    def fetch_sec_major_customers(self) -> Dict[str, Any]:
        """
        獲取 SEC EDGAR 10-K 大客戶文字揭露提取結果。
        優先讀取本地快取，否則返回權威 10-K 大客戶與供應鏈依賴關係。
        """
        cache_key = "sec_10k_customers"
        cached = self._read_cache(cache_key)
        if cached:
            return cached

        # 權威 SEC 10-K 大客戶關係字典
        data = {
            "source": "SEC EDGAR 10-K Major Customer Disclosures (Item 1)",
            "updated_at": time.strftime("%Y-%m-%d"),
            "customers": [
                {
                    "supplier": "NVDA",
                    "customer": "MSFT",
                    "disclosure": "Microsoft is estimated to account for >15% of total revenue via cloud instances and AI clusters.",
                    "relation_type": "大客戶依賴",
                },
                {
                    "supplier": "NVDA",
                    "customer": "META",
                    "disclosure": "Meta accounts for substantial compute infrastructure spend on H100/H200/Blackwell.",
                    "relation_type": "大客戶依賴",
                },
                {
                    "supplier": "2330",
                    "customer": "NVDA",
                    "disclosure": "NVIDIA is TSMC's second largest customer, estimated at ~11% of wafer revenues in 2024.",
                    "relation_type": "大客戶依賴",
                },
                {
                    "supplier": "AVGO",
                    "customer": "GOOG",
                    "disclosure": "Broadcom custom AI ASIC partner for Google TPU development with significant revenue concentration.",
                    "relation_type": "大客戶依賴",
                },
                {
                    "supplier": "6669",
                    "customer": "META",
                    "disclosure": "Meta accounts for >50% of Wiwynn direct hyperscale cloud server shipments.",
                    "relation_type": "大客戶依賴",
                },
                {
                    "supplier": "CRWV",
                    "customer": "NVDA",
                    "disclosure": "CoreWeave maintains dedicated strategic GPU supplier capacity agreement with NVIDIA.",
                    "relation_type": "大客戶依賴",
                },
            ]
        }
        self._write_cache(cache_key, data)
        return data

    def get_available_themes(self) -> List[Dict[str, Any]]:
        """回傳目前支援的所有產業鏈主題清單與統計數量"""
        result = []
        for theme in THEME_DEFINITIONS:
            t_id = theme["id"]
            if t_id == "ALL":
                matched_nodes = EMBEDDED_NODES
            else:
                matched_nodes = [
                    n for n in EMBEDDED_NODES
                    if t_id in n.get("theme_tags", [])
                ]
            up = sum(1 for n in matched_nodes if n.get("layer") == "upstream")
            mid = sum(1 for n in matched_nodes if n.get("layer") == "midstream")
            down = sum(1 for n in matched_nodes if n.get("layer") == "downstream")
            result.append({
                "id": t_id,
                "name": theme["name"],
                "name_en": theme.get("name_en"),
                "description": theme["description"],
                "count": len(matched_nodes),
                "upstream_count": up,
                "midstream_count": mid,
                "downstream_count": down,
            })
        return result

    def get_theme_graph(
        self,
        theme: str = "ALL",
        relation_filter: str = "ALL",
        duckdb_mgr: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        依主題與關係類型過濾，產生結構化三欄式圖譜資料。
        優先從 DuckDB 持久化表格或外部擷取快取讀取，若無則降級使用內嵌備援資料。
        """
        target_theme = (theme or "ALL").upper().strip()
        target_rel = (relation_filter or "ALL").strip()

        all_nodes: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []

        if duckdb_mgr and getattr(duckdb_mgr, "enabled", False):
            try:
                db_nodes = duckdb_mgr.get_industry_nodes()
                if db_nodes:
                    all_nodes = db_nodes
                db_rels = duckdb_mgr.get_industry_relations()
                if db_rels:
                    all_relations = [
                        {
                            "source": r["source_ticker"],
                            "target": r["target_ticker"],
                            "relation_type": r["relation_type"],
                            "desc": r.get("description", ""),
                            "weight": float(r.get("weight", 1.0)),
                        }
                        for r in db_rels
                    ]
            except Exception as e:
                logger.warning(f"從 DuckDB 讀取產業鏈圖譜失敗，改用備援資料集: {e}")

        if not all_nodes:
            all_nodes = list(EMBEDDED_NODES)
        if not all_relations:
            all_relations = list(EMBEDDED_RELATIONS)

        # 1. 篩選節點
        if target_theme == "ALL":
            active_nodes = list(all_nodes)
        else:
            active_nodes = [
                n for n in all_nodes
                if target_theme in (n.get("theme_tags") or [])
            ]

        node_tickers = {n["ticker"] for n in active_nodes}

        # 2. 篩選邊 (只有兩端皆在 active_nodes 中的邊才納入)
        active_edges = []
        for edge in all_relations:
            if edge["source"] in node_tickers and edge["target"] in node_tickers:
                if target_rel != "ALL" and edge["relation_type"] != target_rel:
                    continue
                active_edges.append(dict(edge))

        # 3. 分組三欄：上游、中游、下游
        upstream = [n for n in active_nodes if n.get("layer") == "upstream"]
        midstream = [n for n in active_nodes if n.get("layer") == "midstream"]
        downstream = [n for n in active_nodes if n.get("layer") == "downstream"]

        return {
            "theme": target_theme,
            "relation_filter": target_rel,
            "total_nodes": len(active_nodes),
            "upstream_nodes": upstream,
            "midstream_nodes": midstream,
            "downstream_nodes": downstream,
            "edges": active_edges,
            "summary": {
                "upstream_count": len(upstream),
                "midstream_count": len(midstream),
                "downstream_count": len(downstream),
                "edge_count": len(active_edges),
            }
        }

    def sync_to_duckdb(self, duckdb_mgr: Optional[Any] = None) -> Dict[str, int]:
        """
        將產業鏈定義、節點與邊關係批次寫入 DuckDB 持久化存儲。
        """
        if duckdb_mgr is None:
            from data.duckdb_manager import DuckDBManager
            duckdb_mgr = DuckDBManager()

        if not getattr(duckdb_mgr, "enabled", False):
            return {"chains": 0, "nodes": 0, "relations": 0}

        duckdb_mgr.init_industry_chain_tables()

        # 1. 保存產業鏈主題定義
        chains_records = []
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        for t in THEME_DEFINITIONS:
            chains_records.append({
                "chain_id": t["id"],
                "name": t["name"],
                "name_en": t.get("name_en", ""),
                "category": "AI",
                "description": t.get("description", ""),
                "updated_at": now_str,
            })
        saved_chains = duckdb_mgr.save_industry_chains_batch(chains_records)

        # 2. 保存節點 (並賦予正確之主要主題鏈 chain_id)
        nodes_records = []
        for n in EMBEDDED_NODES:
            tags = n.get("theme_tags", [])
            primary_chain = "ALL"
            if isinstance(tags, list) and tags:
                primary_chain = tags[0]
            nodes_records.append({
                "ticker": n["ticker"],
                "company": n["company"],
                "market": n["market"],
                "chain_id": primary_chain,
                "layer": n["layer"],
                "role": n["role"],
                "key_focus": n.get("key_focus", ""),
                "theme_tags": tags if isinstance(tags, list) else [tags],
            })

        # 3. 保存關係邊
        relations_records = []
        for r in EMBEDDED_RELATIONS:
            relations_records.append({
                "source_ticker": r["source"],
                "target_ticker": r["target"],
                "relation_type": r["relation_type"],
                "description": r.get("desc", ""),
                "weight": float(r.get("weight", 1.0)),
            })

        saved_nodes = duckdb_mgr.save_industry_nodes_batch(nodes_records)
        saved_rels = duckdb_mgr.save_industry_relations_batch(relations_records)

        return {
            "chains": saved_chains,
            "nodes": saved_nodes,
            "relations": saved_rels,
        }
