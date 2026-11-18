# Public Ticker SEO Specification (Delta)

## ADDED Requirements

### Requirement: Public Dossier Accessibility, Focus Visibility, and Layout Safeguards
系統 SHALL 維持 `/s/{ticker}` 公開個股量化頁面之固定深色無 JS SSR 契約，並補齊 `:focus-visible` 高對比焦點外框、`html`/`body` 層級 `overflow-x: clip`、精準 `prefers-reduced-motion` 支援與標題 `text-wrap: balance`，同時改以共用預編譯靜態 CSS 取代執行期 `cdn.tailwindcss.com`。

#### Scenario: Keyboard focus, zero horizontal overflow, and no runtime Tailwind CDN on public dossier
- **WHEN** 訪客於手機窄螢幕、無 JavaScript 環境或純鍵盤環境瀏覽 `/s/2330.TW`
- **THEN** 頁面無需客戶端腳本即可完整閱讀且不載入 `cdn.tailwindcss.com`，無水平捲動溢出，所有連結與 CTA 呈現清楚 `:focus-visible` 外框，且標題換行均衡
