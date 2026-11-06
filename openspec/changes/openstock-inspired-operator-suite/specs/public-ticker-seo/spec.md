# Spec Delta

## Purpose

提供免登入、強制自動跳脫防禦 XSS 之伺服器端渲染公開個股量化摘要網頁、動態社群預覽卡片、安全轉義之 JSON-LD 結構化標記、同業供應鏈導覽與全站動態 Sitemap。

## ADDED Requirements

### Requirement: Escape-Safe Server-Rendered Public Stock Dossier Route
系統 SHALL 使用獨立啟用 HTML 自動跳脫（autoescape）之模板環境提供公開免登入 `GET /s/{ticker}` 路由，驗證代號白名單格式後輸出公司名稱、交易日基準收盤價、模型預測區間、均線與籌碼摘要及互動看板導流連結。

#### Scenario: Valid ticker renders escaped server HTML
- **WHEN** 爬蟲或使用者請求有效代號之 `/s/2330.TW`、`/s/2330` 或 `/s/NVDA`（即使上游公司描述含特殊 HTML 字元）
- **THEN** 系統回傳 HTTP 200 伺服器渲染頁面，所有動態文字均安全跳脫為 HTML 實體，無需客戶端腳本即可閱讀量化摘要

#### Scenario: Malformed or unknown ticker returns non-indexable 404
- **WHEN** 客戶端請求含非法字元（如腳本標籤）或不存在於清冊與預測庫之代號
- **THEN** 系統立即回傳 HTTP 404 與 `noindex, nofollow` 標頭/標籤，拒絕渲染未驗證輸入

### Requirement: Script-Safe Dynamic Social Metadata and JSON-LD
系統 SHALL 於每個公開個股頁面動態輸出專屬標題、描述、Canonical URL、OpenGraph 與 Twitter Card 標籤，並將內嵌 JSON-LD 內之 `<`、`>`、`&` 字元安全轉義以防止 `</script>` 標籤閉合注入。

#### Scenario: Social preview and XSS-safe JSON-LD serialization
- **WHEN** 社群平台爬蟲抓取 `/s/{ticker}` 頁面
- **THEN** `<head>` 包含該標的專屬量化標題與摘要，且 `<script type="application/ld+json">` 為合法且不含未轉義 `</script>` 之安全 JSON

### Requirement: Peer Cross-Linking and Actual-Date Sitemap Discovery
系統 SHALL 於公開個股頁面提供同產業鏈或同板塊關聯標的連結，並透過 `GET /sitemap.xml` 與 `GET /robots.txt` 輸出活躍標的清單，其中 `<lastmod>` 必須使用實際資料更新交易日而非請求當下時間。

#### Scenario: Fetching cached XML sitemap with real lastmod date
- **WHEN** 爬蟲請求 `GET /sitemap.xml`
- **THEN** 系統回傳附帶快取標頭之合法 XML，各標的網址之 `<lastmod>` 反映最後有效預測或行情日期
