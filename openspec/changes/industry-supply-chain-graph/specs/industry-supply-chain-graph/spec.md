# Spec Delta

## Purpose

提供全自動跨市場產業價值鏈動態圖譜模組，負責從 TPEx 官方平台與 SEC EDGAR 擷取上中下游公司角色與供應鏈關係邊，並在 DuckDB 中建立圖譜資料結構。

## ADDED Requirements

### Requirement: Industry Chain Data Ingestion
系統 MUST 支援從 TPEx 官方公開 API 與 SEC EDGAR 自動獲取產業鏈分類、上中下游層級標籤與主要客戶關係資料。

#### Scenario: Fetching TPEx industry chain mapping
- **WHEN** 系統執行產業鏈資料同步
- **THEN** 系統成功解析並儲存上中下游層級、角色描述與所屬上市櫃公司代號至本地快取與 DuckDB

#### Scenario: SEC major customer relationship extraction
- **WHEN** 系統解析美股 10-K 年報中主要客戶揭露段落
- **THEN** 系統抽取營收佔比達標之客戶與供應鏈傳導關係並寫入圖譜關係表

### Requirement: DuckDB Knowledge Graph Storage
系統 MUST 在 DuckDB 中持久化產業鏈定義 (`tw_industry_chains`)、節點歸屬 (`tw_industry_nodes`) 與實體關聯邊 (`tw_industry_relations`)。

#### Scenario: Querying graph by theme
- **WHEN** 呼叫圖譜查詢引擎並指定主題（如 HBM/儲存、CPO/光互聯、電力/液冷）
- **THEN** 系統自 DuckDB 快速回傳符合主題之節點清單及其上中下游拓撲關係
