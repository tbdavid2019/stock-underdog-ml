/**
 * app.js - Main Application Controller for 888 Stock Quant Platform
 * Powered by Vue 3, Tailwind CSS, Chart.js & FastAPI
 */

import { ApiClient } from '/static/js/api-client.js';
import { ChartManager } from '/static/js/chart-manager.js';

const { createApp } = Vue;

const chartManager = new ChartManager();

export const app = createApp({
    data() {
        // Ensure default theme is Obsidian Dark Terminal per Figma 20261013
        const savedTheme = localStorage.getItem('theme_v20261013');
        const isDark = savedTheme ? savedTheme === 'dark' : true;
        return {
            isDark: isDark,
            activeTab: 'dashboard',
            selectedStrategy: 'resonance',
            selectedIndex: '',
            viewMode: 'table', // 'table' | 'card'
            expandedDrawerTicker: null,
            expandedDrawerData: null,
            expandedDrawerLoading: false,

            strategyOptions: [
                { id: 'resonance', label: '三重共振焦點', icon: 'military_tech' },
                { id: 'xuantie', label: '玄鐵均線波段', icon: 'show_chart' },
                { id: 'timesfm_bullish', label: 'TimesFM 預測 TOP', icon: 'auto_awesome' },
                { id: 'bullish', label: 'LSTM 看漲 TOP', icon: 'trending_up' },
                { id: 'bearish', label: 'LSTM 看跌避險', icon: 'trending_down' },
                { id: 'timesfm_bearish', label: 'TimesFM 避險', icon: 'shield' },
                { id: 'latest', label: '最新全量批次', icon: 'dataset' }
            ],

            macro: {},
            showMacroDetails: true,
            macroRefreshing: false,
            stats: {},
            marketStats: {},
            records: [],
            batchDate: '',
            batchTimestamp: '',
            batchStale: false,
            loading: false,
            loadError: '',

            // Institutional Radar Data
            instSubTab: 'three_inst',
            institutionalFlows: [],
            instLoading: false,
            instOrderBy: 'total_net',
            instSortDir: 'desc',
            instMarket: 'ALL',
            availableDates: [],
            instSelectedDate: '',

            // Broker Radar Data
            brokerSearchTicker: '2330',
            brokerDays: 20,
            brokerLoading: false,
            brokerDates: [],
            brokerSeries: [],
            brokerSummary: {},

            // Ticker Explorer
            searchTicker: '2330.TW',
            historyRecords: [],
            historyLoading: false,
            tickerInstHistory: [],
            tickerInstLoading: false,
            hasSearched: false,
            resolvedTicker: '',
            resolvedName: '',
            historyError: '',
            historyRequestId: 0,

            // Earnings Calls State (Task B)
            upcomingEarningsCalls: [],
            earningsLoading: false,
            earningsMarket: 'ALL', // 'ALL' | 'TW' | 'US'
            earningsWeightOnly: false,
            drawerActiveSubTab: 'profile', // 'profile' | 'earnings'
            drawerEarningsMap: {},
            drawerEarningsLoading: false,
            showEarningsModal: false,
            calendarYear: new Date().getFullYear(),
            calendarMonth: new Date().getMonth() + 1,
            calendarData: [],
            calendarLoading: false,
            calendarTimezone: 'CST', // 'CST' | 'EST'
            calendarSearch: '',

            // Podcast Radar State (Task C)
            podcastEpisodes: [],
            podcastChannels: [],
            podcastTopics: [],
            podcastLoading: false,
            podcastSelectedTopic: '',
            podcastSelectedChannel: 'ALL',
            podcastSelectedLang: 'ALL',
            podcastSearchQuery: '',
            podcastExpandedEpisodeId: null,
            podcastExpandedTicker: null,
            podcastDrawerData: null,
            podcastDrawerLoading: false,
            podcastActiveAudioId: null,
            podcastActiveVideoId: null,
            podcastSubView: 'episodes', // 'episodes' | 'channels'

            // Toast
            toastMessage: '',

            // Company Profile Hover Card (2md Powered)
            hoverVisible: false,
            hoverData: null,
            hoverLoading: false,
            hoverPos: { top: 0, left: 0 },
            hoverTimeout: null,
            companyProfileCache: {},
            prefetchSet: new Set(),

            // Snippets
            claudeMcpConfig: JSON.stringify({
                "mcpServers": {
                    "stock-quant": {
                        "command": "python",
                        "args": [
                            "/home/david/stock-underdog-ml/mcp_server.py"
                        ]
                    }
                }
            }, null, 2),

            pythonSnippet: `import requests\n\n# 取得三重共振焦點推薦標的\nres = requests.get(f"${window.location.origin}/api/v1/predictions/resonance")\npicks = res.json().get("data", [])\nfor p in picks:\n    print(f"{p['ticker']} | 潛力:{p['potential']}% | 標籤:{p['tags']}")`
        };
    },
    methods: {
        toggleTheme() {
            this.isDark = !this.isDark;
            const themeStr = this.isDark ? 'dark' : 'light';
            localStorage.setItem('theme_v20261013', themeStr);
            localStorage.setItem('theme', themeStr);
            this.applyTheme();
            chartManager.updateTheme(this.isDark);
        },
        applyTheme() {
            if (this.isDark) {
                document.documentElement.classList.add('dark');
            } else {
                document.documentElement.classList.remove('dark');
            }
        },
        async loadMacro() {
            try {
                this.macro = await ApiClient.getMacroLatest();
            } catch (e) {
                console.error("Failed to load macro", e);
            }
        },
        async refreshMacro() {
            this.macroRefreshing = true;
            try {
                const { summary, polymarket } = await ApiClient.refreshMacro();
                if (summary) {
                    this.macro = {
                        ...this.macro,
                        fed_rate: summary.fed_rate,
                        earnings_calendar: summary.earnings_calendar,
                        commodities: summary.commodities,
                        economic_calendar: summary.economic_calendar,
                        catalyst_alerts: summary.catalyst_alerts,
                        polymarket: polymarket || this.macro.polymarket
                    };
                    this.showToast('總經與行事曆已即時同步！');
                } else {
                    this.showToast('更新失敗，請稍後再試');
                }
            } catch (e) {
                console.error("Failed to refresh macro", e);
                this.showToast('更新發生錯誤');
            } finally {
                this.macroRefreshing = false;
            }
        },
        async loadStats() {
            try {
                const data = await ApiClient.getStatsSummary();
                if (data.success) {
                    this.stats = data;
                }
            } catch (e) {
                console.error("Failed to load stats", e);
            }
        },
        async loadMarketStats() {
            try {
                this.marketStats = await ApiClient.getMarketStats();
            } catch (e) {
                console.error("Failed to load market stats", e);
            }
        },
        async loadAvailableDates() {
            try {
                const data = await ApiClient.getInstitutionalDates();
                if (Array.isArray(data)) {
                    this.availableDates = data;
                }
            } catch (e) {
                console.error("Failed to load available dates", e);
            }
        },
        async loadInstitutionalFlows() {
            this.instLoading = true;
            try {
                this.institutionalFlows = await ApiClient.getInstitutionalFlows({
                    orderBy: this.instOrderBy,
                    sortDir: this.instSortDir,
                    market: this.instMarket,
                    date: this.instSelectedDate,
                    limit: 30
                });
                this.prefetchVisibleProfiles(this.institutionalFlows.map(i => i.ticker));
            } catch (e) {
                console.error("Failed to load institutional flows", e);
            } finally {
                this.instLoading = false;
            }
        },
        async loadData() {
            this.loading = true;
            this.loadError = '';
            try {
                const data = await ApiClient.getPredictions({
                    strategy: this.selectedStrategy,
                    indexName: this.selectedIndex,
                    limit: 60
                });
                this.records = data.data || [];
                this.prefetchVisibleProfiles(this.records.map(r => r.ticker));
                const firstRecord = this.records[0] || {};
                this.batchDate = data.batch_date || firstRecord.analysis_date || '';
                const timestamp = data.latest_batch_timestamp || firstRecord.data_as_of || firstRecord.timestamp || '';
                this.batchTimestamp = timestamp ? timestamp.replace('T', ' ').substring(0, 19) : '';
                this.batchStale = Boolean(data.is_stale || firstRecord.is_stale);
            } catch (e) {
                this.records = [];
                this.batchDate = '';
                this.batchTimestamp = '';
                this.batchStale = false;
                this.loadError = '暫時無法取得日報，請重新整理。';
                console.error("Failed to load strategy data", e);
            } finally {
                this.loading = false;
            }
        },
        async loadHistory() {
            const query = this.searchTicker.trim();
            if (!query) return;
            const requestId = ++this.historyRequestId;
            this.historyLoading = true;
            this.hasSearched = true;
            this.historyError = '';
            this.historyRecords = [];
            this.tickerInstHistory = [];
            this.resolvedTicker = '';
            this.resolvedName = '';

            try {
                const resolution = await ApiClient.resolveTicker(query);
                if (requestId !== this.historyRequestId) return;
                this.resolvedTicker = resolution.ticker;
                this.resolvedName = resolution.name || '';
                this.searchTicker = resolution.ticker;

                const [historyData, instData] = await Promise.all([
                    ApiClient.getPredictionHistory(resolution.ticker, 40).catch(() => []),
                    ApiClient.getInstitutionalHistory(resolution.ticker, 60).catch(() => [])
                ]);

                if (requestId !== this.historyRequestId) return;
                this.historyRecords = historyData;
                this.tickerInstHistory = instData;

                if (this.tickerInstHistory.length > 0) {
                    this.$nextTick(() => {
                        this.renderInstChart();
                    });
                }
            } catch (e) {
                if (requestId !== this.historyRequestId) return;
                this.historyError = e.message || '查詢服務暫時無法使用，請稍後再試。';
                console.error("Failed to load ticker history", e);
            } finally {
                if (requestId === this.historyRequestId) {
                    this.historyLoading = false;
                }
            }
        },
        renderInstChart() {
            this.$nextTick(() => {
                chartManager.renderInstChart('instChartCanvas', this.tickerInstHistory, this.isDark);
            });
        },
        async loadBrokerTrend() {
            if (!this.brokerSearchTicker.trim()) return;
            this.brokerLoading = true;
            try {
                const ticker = this.brokerSearchTicker.trim().toUpperCase();
                const [dataTrend, dataSummary] = await Promise.all([
                    ApiClient.getBrokerTrend(ticker, this.brokerDays, 8).catch(() => ({ dates: [], series: [] })),
                    ApiClient.getBrokerSummary(ticker, this.brokerDays).catch(() => ({}))
                ]);

                this.brokerDates = dataTrend.dates || [];
                this.brokerSeries = dataTrend.series || [];
                this.brokerSummary = dataSummary || {};

                if (this.brokerSeries.length > 0) {
                    this.renderBrokerChart();
                }
            } catch (e) {
                console.error("Failed to load broker trend", e);
            } finally {
                this.brokerLoading = false;
            }
        },
        renderBrokerChart() {
            this.$nextTick(() => {
                chartManager.renderBrokerChart('brokerChartCanvas', this.brokerDates, this.brokerSeries, this.isDark);
            });
        },
        async toggleRowDrawer(ticker) {
            if (this.expandedDrawerTicker === ticker) {
                this.expandedDrawerTicker = null;
                this.expandedDrawerData = null;
                return;
            }

            this.expandedDrawerTicker = ticker;
            this.drawerActiveSubTab = 'profile';

            // Also load earnings call history for this ticker
            this.loadTickerEarnings(ticker);

            if (this.companyProfileCache[ticker]) {
                this.expandedDrawerData = this.companyProfileCache[ticker];
                this.expandedDrawerLoading = false;
                return;
            }

            this.expandedDrawerLoading = true;
            this.expandedDrawerData = {
                ticker: ticker,
                name: ticker,
                industry: '載入中...',
                business_summary: '正在連線 2md 即時爬取公司基本資料與新聞...',
                recent_news: []
            };

            try {
                const data = await ApiClient.getCompanyProfile(ticker);
                if (data && data.ticker) {
                    this.companyProfileCache[ticker] = data;
                    if (this.expandedDrawerTicker === ticker) {
                        this.expandedDrawerData = data;
                    }
                }
            } catch (e) {
                console.debug("Failed to load company profile drawer:", e);
            } finally {
                if (this.expandedDrawerTicker === ticker) {
                    this.expandedDrawerLoading = false;
                }
            }
        },
        async loadUpcomingEarnings() {
            this.earningsLoading = true;
            try {
                const res = await ApiClient.getUpcomingEarningsCalls({
                    market: this.earningsMarket,
                    days: 30,
                    isWeightStock: this.earningsWeightOnly ? true : null,
                    limit: 30
                });
                if (res && res.data) {
                    this.upcomingEarningsCalls = res.data;
                }
            } catch (e) {
                console.warn("Failed to load upcoming earnings calls:", e);
            } finally {
                this.earningsLoading = false;
            }
        },
        setEarningsMarket(m) {
            this.earningsMarket = m;
            this.loadUpcomingEarnings();
            if (this.showEarningsModal) {
                this.loadCalendarData();
            }
        },
        toggleEarningsWeightOnly() {
            this.earningsWeightOnly = !this.earningsWeightOnly;
            this.loadUpcomingEarnings();
        },
        async loadTickerEarnings(ticker) {
            if (!ticker) return;
            if (this.drawerEarningsMap[ticker]) return;
            this.drawerEarningsLoading = true;
            try {
                const res = await ApiClient.getTickerEarningsCalls(ticker);
                if (res && res.data) {
                    this.drawerEarningsMap[ticker] = res.data;
                } else {
                    this.drawerEarningsMap[ticker] = [];
                }
            } catch (e) {
                console.warn(`Failed to load earnings for ${ticker}:`, e);
                this.drawerEarningsMap[ticker] = [];
            } finally {
                this.drawerEarningsLoading = false;
            }
        },
        async openEarningsCalendar() {
            this.showEarningsModal = true;
            document.body.classList.add('overflow-hidden');
            await this.loadCalendarData();
        },
        closeEarningsCalendar() {
            this.showEarningsModal = false;
            document.body.classList.remove('overflow-hidden');
        },
        toggleCalendarTimezone() {
            this.calendarTimezone = this.calendarTimezone === 'CST' ? 'EST' : 'CST';
        },
        async changeCalendarMonth(delta) {
            let m = this.calendarMonth + delta;
            let y = this.calendarYear;
            if (m > 12) {
                m = 1;
                y++;
            } else if (m < 1) {
                m = 12;
                y--;
            }
            this.calendarMonth = m;
            this.calendarYear = y;
            await this.loadCalendarData();
        },
        async loadCalendarData() {
            this.calendarLoading = true;
            try {
                const res = await ApiClient.getEarningsCalendar(
                    this.calendarYear,
                    this.calendarMonth,
                    this.earningsMarket
                );
                if (res && res.data) {
                    this.calendarData = res.data;
                }
            } catch (e) {
                console.warn("Failed to load earnings calendar:", e);
            } finally {
                this.calendarLoading = false;
            }
        },
        async syncEarningsNow() {
            try {
                this.showToast("正在背景觸發台美股法說會資料全量同步...");
                const res = await fetch('/api/v1/earnings-call/sync', { method: 'POST' });
                const json = await res.json();
                if (json.success) {
                    this.showToast("法說會同步任務已派發！");
                    setTimeout(() => {
                        this.loadUpcomingEarnings();
                        if (this.showEarningsModal) this.loadCalendarData();
                    }, 2500);
                }
            } catch (e) {
                this.showToast("同步失敗：" + e.message);
            }
        },
        searchTickerHistory(ticker) {
            this.searchTicker = ticker;
            this.activeTab = 'history';
            this.updateUrl();
            this.loadHistory();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        },
        setTab(tab) {
            if (this.activeTab !== tab) {
                chartManager.destroyAll();
            }
            this.activeTab = tab;
            this.updateUrl();
            if (tab === 'institutional') {
                if (this.instSubTab === 'three_inst') {
                    this.loadInstitutionalFlows();
                } else {
                    this.loadBrokerTrend();
                }
            } else if (tab === 'dashboard') {
                this.loadData();
            } else if (tab === 'history') {
                if (this.searchTicker && !this.historyRecords.length) {
                    this.loadHistory();
                }
            } else if (tab === 'podcast') {
                this.loadPodcastEpisodes();
                this.loadPodcastChannels();
                this.loadPodcastTopics();
            }
        },
        async loadPodcastEpisodes() {
            this.podcastLoading = true;
            try {
                const res = await ApiClient.getPodcastEpisodes({
                    topic: this.podcastSelectedTopic || null,
                    channelId: this.podcastSelectedChannel,
                    lang: this.podcastSelectedLang,
                    limit: 30
                });
                if (res && res.data) {
                    this.podcastEpisodes = res.data;
                }
            } catch (e) {
                console.warn("Failed to load podcast episodes:", e);
            } finally {
                this.podcastLoading = false;
            }
        },
        async loadPodcastChannels() {
            try {
                const res = await ApiClient.getPodcastChannels();
                if (res && res.data) {
                    this.podcastChannels = res.data;
                }
            } catch (e) {
                console.warn("Failed to load podcast channels:", e);
            }
        },
        async loadPodcastTopics() {
            try {
                const res = await ApiClient.getPodcastTopics(25);
                if (res && res.data) {
                    this.podcastTopics = res.data;
                }
            } catch (e) {
                console.warn("Failed to load podcast topics:", e);
            }
        },
        togglePodcastSubView(view) {
            this.podcastSubView = view;
        },
        formatFollowers(count) {
            if (!count) return '0';
            if (count >= 1000000) {
                return (count / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
            }
            if (count >= 1000) {
                return (count / 1000).toFixed(0) + 'K';
            }
            return count.toString();
        },
        selectChannelAndFilter(channelId) {
            this.podcastSelectedChannel = channelId;
            this.podcastSubView = 'episodes';
            this.updateUrl();
            this.loadPodcastEpisodes();
        },
        setPodcastTopic(topic) {
            if (this.podcastSelectedTopic === topic) {
                this.podcastSelectedTopic = '';
            } else {
                this.podcastSelectedTopic = topic;
            }
            this.updateUrl();
            this.loadPodcastEpisodes();
        },
        setPodcastChannel(channelId) {
            this.podcastSelectedChannel = channelId;
            this.updateUrl();
            this.loadPodcastEpisodes();
        },
        setPodcastLang(lang) {
            this.podcastSelectedLang = lang;
            this.updateUrl();
            this.loadPodcastEpisodes();
        },
        async syncPodcastNow() {
            try {
                this.showToast("正在背景觸發 Podcast 情報雷達全量同步...");
                const res = await ApiClient.syncPodcast();
                if (res && res.success) {
                    this.showToast("Podcast 同步任務已派發！");
                    setTimeout(() => {
                        this.loadPodcastEpisodes();
                        this.loadPodcastTopics();
                        this.loadPodcastChannels();
                    }, 2500);
                }
            } catch (e) {
                this.showToast("同步失敗：" + e.message);
            }
        },
        togglePodcastTicker(episodeId, ticker) {
            if (this.podcastExpandedEpisodeId === episodeId && this.podcastExpandedTicker === ticker) {
                this.podcastExpandedEpisodeId = null;
                this.podcastExpandedTicker = null;
                this.podcastDrawerData = null;
                return;
            }

            this.podcastExpandedEpisodeId = episodeId;
            this.podcastExpandedTicker = ticker;

            if (this.companyProfileCache[ticker]) {
                this.podcastDrawerData = this.companyProfileCache[ticker];
                this.podcastDrawerLoading = false;
                return;
            }

            this.podcastDrawerLoading = true;
            this.podcastDrawerData = {
                ticker: ticker,
                name: ticker,
                industry: '查詢中...',
                business_summary: '正在查詢公司基本面資料與最新產業資訊...',
                recent_news: []
            };

            ApiClient.getCompanyProfile(ticker)
                .then(data => {
                    if (data && data.ticker) {
                        this.companyProfileCache[ticker] = data;
                        if (this.podcastExpandedTicker === ticker && this.podcastExpandedEpisodeId === episodeId) {
                            this.podcastDrawerData = data;
                        }
                    }
                })
                .catch(e => console.debug("Podcast drawer profile fetch failed:", e))
                .finally(() => {
                    if (this.podcastExpandedTicker === ticker && this.podcastExpandedEpisodeId === episodeId) {
                        this.podcastDrawerLoading = false;
                    }
                });
        },
        closePodcastDrawer() {
            this.podcastExpandedEpisodeId = null;
            this.podcastExpandedTicker = null;
            this.podcastDrawerData = null;
        },
        togglePodcastAudio(episodeId) {
            if (this.podcastActiveAudioId === episodeId) {
                this.podcastActiveAudioId = null;
            } else {
                this.podcastActiveAudioId = episodeId;
                this.podcastActiveVideoId = null; // 關閉影片避免雙重音訊衝突
            }
        },
        togglePodcastVideo(episodeId) {
            if (this.podcastActiveVideoId === episodeId) {
                this.podcastActiveVideoId = null;
            } else {
                this.podcastActiveVideoId = episodeId;
                this.podcastActiveAudioId = null; // 關閉音訊避免雙重音訊衝突
            }
        },
        getYouTubeEmbedUrl(url) {
            if (!url) return '';
            const match = url.match(/(?:youtu\.be\/|youtube\.com\/(?:watch\?v=|embed\/|shorts\/))([\w-]{11})/);
            return match ? `https://www.youtube-nocookie.com/embed/${match[1]}?autoplay=1` : '';
        },
        setStrategy(stratId) {
            this.selectedStrategy = stratId;
            this.updateUrl();
            this.loadData();
        },
        setIndex() {
            this.updateUrl();
            this.loadData();
        },
        updateUrl(replace = false) {
            try {
                const params = new URLSearchParams();
                if (this.activeTab && this.activeTab !== 'dashboard') {
                    params.set('tab', this.activeTab);
                }
                if (this.activeTab === 'dashboard') {
                    if (this.selectedStrategy && this.selectedStrategy !== 'resonance') {
                        params.set('strategy', this.selectedStrategy);
                    }
                    if (this.selectedIndex) {
                        params.set('index', this.selectedIndex);
                    }
                } else if (this.activeTab === 'institutional') {
                    if (this.instSubTab && this.instSubTab !== 'three_inst') {
                        params.set('sub', this.instSubTab);
                    }
                    if (this.instMarket && this.instMarket !== 'ALL') {
                        params.set('market', this.instMarket);
                    }
                    if (this.instOrderBy && this.instOrderBy !== 'total_net') {
                        params.set('order', this.instOrderBy);
                    }
                    if (this.instSelectedDate) {
                        params.set('date', this.instSelectedDate);
                    }
                    if (this.instSubTab === 'broker' && this.brokerSearchTicker) {
                        params.set('broker_ticker', this.brokerSearchTicker);
                    }
                } else if (this.activeTab === 'history') {
                    if (this.searchTicker) {
                        params.set('ticker', this.searchTicker);
                    }
                } else if (this.activeTab === 'podcast') {
                    if (this.podcastSelectedTopic) {
                        params.set('topic', this.podcastSelectedTopic);
                    }
                    if (this.podcastSelectedChannel && this.podcastSelectedChannel !== 'ALL') {
                        params.set('channel', this.podcastSelectedChannel);
                    }
                    if (this.podcastSelectedLang && this.podcastSelectedLang !== 'ALL') {
                        params.set('lang', this.podcastSelectedLang);
                    }
                }
                const queryStr = params.toString();
                const newUrl = queryStr ? `${window.location.pathname}?${queryStr}` : window.location.pathname;
                if (window.location.search !== (queryStr ? `?${queryStr}` : '')) {
                    if (replace) {
                        window.history.replaceState({ tab: this.activeTab }, '', newUrl);
                    } else {
                        window.history.pushState({ tab: this.activeTab }, '', newUrl);
                    }
                }
            } catch (e) {
                console.debug("History pushState error:", e);
            }
        },
        parseUrlParams() {
            try {
                const params = new URLSearchParams(window.location.search);
                const tab = params.get('tab');
                this.activeTab = (tab && ['dashboard', 'institutional', 'history', 'agent_hub', 'podcast'].includes(tab)) ? tab : 'dashboard';

                const strat = params.get('strategy');
                this.selectedStrategy = (strat && ['resonance', 'xuantie', 'bullish', 'bearish', 'timesfm_bullish', 'timesfm_bearish', 'latest'].includes(strat)) ? strat : 'resonance';

                const idx = params.get('index');
                this.selectedIndex = (idx && idx !== 'ALL') ? idx : '';

                const sub = params.get('sub');
                this.instSubTab = (sub && ['three_inst', 'broker'].includes(sub)) ? sub : 'three_inst';

                this.instMarket = params.get('market') || 'ALL';
                this.instOrderBy = params.get('order') || 'total_net';
                this.instSelectedDate = params.get('date') || '';
                this.brokerSearchTicker = params.get('broker_ticker') || '';
                this.searchTicker = params.get('ticker') || '';

                this.podcastSelectedTopic = params.get('topic') || '';
                this.podcastSelectedChannel = params.get('channel') || 'ALL';
                this.podcastSelectedLang = params.get('lang') || 'ALL';
            } catch (e) {
                console.debug("Parse URL params error:", e);
            }
        },
        syncStateFromUrl() {
            this.parseUrlParams();
            if (this.activeTab === 'dashboard') {
                this.loadData();
            } else if (this.activeTab === 'institutional') {
                if (this.instSubTab === 'three_inst') {
                    this.loadInstitutionalFlows();
                } else {
                    this.loadBrokerTrend();
                }
            } else if (this.activeTab === 'history') {
                if (this.searchTicker) {
                    this.loadHistory();
                }
            } else if (this.activeTab === 'podcast') {
                this.loadPodcastEpisodes();
                this.loadPodcastChannels();
                this.loadPodcastTopics();
            }
        },
        copyShareLink() {
            this.updateUrl(true);
            const fullUrl = window.location.href;
            this.copyText(fullUrl, '已複製目前頁面專屬分享連結！');
        },

        // --- Company Profile Hover Popover (2md Powered) ---
        showCompanyHover(event, ticker) {
            if (!ticker) return;
            if (this.hoverTimeout) {
                clearTimeout(this.hoverTimeout);
                this.hoverTimeout = null;
            }

            const rect = event.currentTarget.getBoundingClientRect();
            const popoverWidth = 384;
            const popoverHeight = 280;

            let left = rect.left;
            if (left + popoverWidth > window.innerWidth - 20) {
                left = window.innerWidth - popoverWidth - 20;
            }
            if (left < 10) left = 10;

            let top = rect.bottom + 8;
            if (top + popoverHeight > window.innerHeight - 20) {
                top = rect.top - popoverHeight - 8;
            }
            if (top < 10) top = 10;

            this.hoverPos = { top: Math.round(top), left: Math.round(left) };
            this.hoverVisible = true;

            if (this.companyProfileCache[ticker]) {
                this.hoverData = this.companyProfileCache[ticker];
                this.hoverLoading = false;
                return;
            }

            this.hoverLoading = true;
            this.hoverData = { ticker: ticker, name: ticker, industry: '載入中...', business_summary: '正在連線 2md 即時爬取公司基本資料與新聞...', recent_news: [] };

            this.fetchCompanyProfile(ticker);
        },
        hideCompanyHover(immediate = false) {
            if (immediate) {
                this.hoverVisible = false;
                if (this.hoverTimeout) {
                    clearTimeout(this.hoverTimeout);
                    this.hoverTimeout = null;
                }
                return;
            }
            this.hoverTimeout = setTimeout(() => {
                this.hoverVisible = false;
            }, 250);
        },
        clearHoverTimeout() {
            if (this.hoverTimeout) {
                clearTimeout(this.hoverTimeout);
                this.hoverTimeout = null;
            }
        },
        async fetchCompanyProfile(ticker) {
            const abortController = new AbortController();
            const timeoutId = setTimeout(() => {
                abortController.abort();
                if (this.hoverLoading && this.hoverData?.ticker === ticker) {
                    this.hoverLoading = false;
                }
            }, 6500);

            try {
                const data = await ApiClient.getCompanyProfile(ticker, abortController.signal);
                clearTimeout(timeoutId);
                if (data && data.ticker) {
                    this.companyProfileCache[ticker] = data;
                    if (this.hoverVisible && (this.hoverData?.ticker === ticker || !this.hoverData)) {
                        this.hoverData = data;
                    }
                }
            } catch (e) {
                // Silent abort/timeout fallback
            } finally {
                clearTimeout(timeoutId);
                if (this.hoverData?.ticker === ticker) {
                    this.hoverLoading = false;
                }
            }
        },
        prefetchVisibleProfiles(tickers) {
            if (!tickers || !Array.isArray(tickers) || tickers.length === 0) return;
            const pending = tickers
                .map(t => String(t || '').trim())
                .filter(t => t && !this.companyProfileCache[t] && !this.prefetchSet.has(t));

            if (pending.length === 0) return;

            pending.forEach(t => this.prefetchSet.add(t));

            const chunkSize = 20;
            for (let i = 0; i < pending.length; i += chunkSize) {
                const chunk = pending.slice(i, i + chunkSize);
                setTimeout(() => {
                    ApiClient.getCompanyProfilesBatch(chunk)
                        .then(res => {
                            if (res && res.success && res.data) {
                                Object.entries(res.data).forEach(([ticker, prof]) => {
                                    this.companyProfileCache[ticker] = prof;
                                    if (this.hoverVisible && this.hoverData?.ticker === ticker) {
                                        this.hoverData = prof;
                                        this.hoverLoading = false;
                                    }
                                });
                            }
                        })
                        .catch(e => console.debug("Prefetch batch skipped:", e));
                }, (i / chunkSize) * 300);
            }
        },
        parseTags(tags) {
            if (!tags) return [];
            if (Array.isArray(tags)) return tags;
            return String(tags).split('|').map(t => t.trim()).filter(t => t.length > 0);
        },
        showToast(msg) {
            this.toastMessage = msg || '';
            setTimeout(() => {
                this.toastMessage = '';
            }, 2500);
        },
        copyText(text, msg) {
            navigator.clipboard.writeText(text).then(() => {
                this.showToast(msg || '已複製到剪貼簿！');
            });
        },
        async registerChromeWebMCP() {
            const modelCtx = window.document?.modelContext || window.navigator?.modelContext;
            if (!modelCtx || typeof modelCtx.registerTool !== 'function') return;

            try {
                await modelCtx.registerTool({
                    name: 'get_market_macro_regime',
                    description: '查詢最新美股宏觀風控狀態（SPY季線、VIX恐慌指數、SOX費半趨勢）與建議投資曝險比例。',
                    inputSchema: { type: 'object', properties: {} },
                    execute: async () => JSON.stringify(await ApiClient.getMacroLatest())
                });

                await modelCtx.registerTool({
                    name: 'get_triple_resonance_stocks',
                    description: '篩選三重共振強勢股（技術面均線 ∩ LSTM看漲 ∩ 法人籌碼鎖碼 ∩ 估值合理）。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            index_name: { type: 'string', description: '指數名稱（台灣50、台灣中型100、SP500）' },
                            limit: { type: 'integer', description: '回傳最大筆數，預設 20' }
                        }
                    },
                    execute: async ({ index_name, limit }) => JSON.stringify(await ApiClient.getPredictions({ strategy: 'resonance', indexName: index_name, limit: limit || 20 }))
                });

                await modelCtx.registerTool({
                    name: 'get_xuantie_pullback_stocks',
                    description: '查詢玄鐵重劍策略回調買點標的（回踩 MA60 季線或 MA120 半年線不破之波段買點）。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            index_name: { type: 'string', description: '指數名稱' },
                            limit: { type: 'integer', description: '回傳筆數' }
                        }
                    },
                    execute: async ({ index_name, limit }) => JSON.stringify(await ApiClient.getPredictions({ strategy: 'xuantie', indexName: index_name, limit: limit || 20 }))
                });

                await modelCtx.registerTool({
                    name: 'get_lstm_top_predictions',
                    description: '取得 LSTM 深度學習下一交易日預測漲跌幅排行榜（看漲或看跌）。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            direction: { type: 'string', enum: ['bullish', 'bearish'], description: 'bullish 看漲 / bearish 看跌' },
                            index_name: { type: 'string', description: '指數名稱' },
                            limit: { type: 'integer', description: '回傳筆數' }
                        },
                        required: ['direction']
                    },
                    execute: async ({ direction, index_name, limit }) => JSON.stringify(await ApiClient.getPredictions({ strategy: direction === 'bearish' ? 'bearish' : 'bullish', indexName: index_name, limit: limit || 20 }))
                });

                await modelCtx.registerTool({
                    name: 'get_timesfm_top_predictions',
                    description: '取得 Google Research TimesFM 時序大模型 5 日預測漲跌幅排行榜與盈虧比。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            direction: { type: 'string', enum: ['bullish', 'bearish'], description: 'bullish 看漲 / bearish 看跌' },
                            index_name: { type: 'string', description: '指數名稱' },
                            limit: { type: 'integer', description: '回傳筆數' }
                        },
                        required: ['direction']
                    },
                    execute: async ({ direction, index_name, limit }) => JSON.stringify(await ApiClient.getPredictions({ strategy: direction === 'bearish' ? 'timesfm_bearish' : 'timesfm_bullish', indexName: index_name, limit: limit || 20 }))
                });

                await modelCtx.registerTool({
                    name: 'get_stock_history',
                    description: '查詢特定股票（如 2330.TW, NVDA）在 DuckDB 中的歷史時序預測軌跡與法人籌碼。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            ticker: { type: 'string', description: '股票代號（例如 2330.TW, NVDA）' },
                            limit: { type: 'integer', description: '回傳筆數，預設 30' }
                        },
                        required: ['ticker']
                    },
                    execute: async ({ ticker, limit }) => JSON.stringify(await ApiClient.getPredictionHistory(ticker, limit || 30))
                });

                await modelCtx.registerTool({
                    name: 'get_top_institutional_flows',
                    description: '查詢最新台股三大法人（外資、投信、自營商）淨買超或淨賣超排行焦點股。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            order_by: { type: 'string', enum: ['total_net', 'foreign_net', 'trust_net', 'dealer_net'], description: '排序欄位' },
                            sort_dir: { type: 'string', enum: ['desc', 'asc'], description: 'desc 買超榜 / asc 賣超榜' },
                            market: { type: 'string', enum: ['ALL', 'TWSE', 'TPEX'], description: '市場篩選' },
                            limit: { type: 'integer', description: '回傳筆數' }
                        }
                    },
                    execute: async ({ order_by, sort_dir, market, limit }) => JSON.stringify(await ApiClient.getInstitutionalFlows({ orderBy: order_by, sortDir: sort_dir, market: market, limit: limit || 20 }))
                });

                await modelCtx.registerTool({
                    name: 'get_broker_trades_for_stock',
                    description: '查詢特定股票（如 2330.TW, 2454.TW）各關鍵券商分點（高盛、摩根士丹利、元大台北等）主力買賣超明細與累計趨勢。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            ticker: { type: 'string', description: '股票代號（例如 2330.TW）' },
                            days: { type: 'integer', description: '統計天數，預設 20' }
                        },
                        required: ['ticker']
                    },
                    execute: async ({ ticker, days }) => JSON.stringify(await ApiClient.getBrokerSummary(ticker, days || 20))
                });

                await modelCtx.registerTool({
                    name: 'get_company_profile',
                    description: '透過 2md 查詢特定股票之繁體中文公司簡介、核心營運業務與最新新聞。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            ticker: { type: 'string', description: '股票代號（例如 9945.TW, 2330.TW, AAPL）' }
                        },
                        required: ['ticker']
                    },
                    execute: async ({ ticker }) => JSON.stringify(await ApiClient.getCompanyProfile(ticker))
                });

                console.log('[Chrome WebMCP] Registered native tools to document.modelContext');
            } catch (e) {
                console.warn('[Chrome WebMCP] Registration warning:', e);
            }
        }
    },
    mounted() {
        this.applyTheme();
        this.parseUrlParams();
        this.loadMacro();
        this.loadStats();
        this.loadMarketStats();
        this.loadAvailableDates();
        this.loadUpcomingEarnings();

        if (this.activeTab === 'institutional') {
            if (this.instSubTab === 'three_inst') {
                this.loadInstitutionalFlows();
            } else {
                this.loadBrokerTrend();
            }
        } else if (this.activeTab === 'history') {
            if (this.searchTicker) {
                this.loadHistory();
            }
        } else if (this.activeTab === 'podcast') {
            this.loadPodcastEpisodes();
            this.loadPodcastChannels();
            this.loadPodcastTopics();
        } else {
            this.loadData();
        }

        window.addEventListener('popstate', () => {
            this.syncStateFromUrl();
        });

        this.registerChromeWebMCP();
    }
});

// Auto-mount on load
app.mount('#app');
