/**
 * app.js - Main Application Controller for 888 Stock Quant Platform
 * Powered by Vue 3, Tailwind CSS, Chart.js & FastAPI
 */

import { ApiClient } from '/static/js/api-client.js?v=2.4.37';
import { ChartManager } from '/static/js/chart-manager.js?v=2.4.39';

const { createApp } = Vue;

const chartManager = new ChartManager();

export const app = createApp({
    data() {
        // Default theme is Light Mode unless explicitly set to 'dark' by user
        const savedTheme = localStorage.getItem('theme_v20261013') || localStorage.getItem('theme');
        const isDark = savedTheme === 'dark';
        const savedLocale = localStorage.getItem('stock_quant_locale') || 'zh';
        const savedViewMode = localStorage.getItem('stock_quant_view_mode') || (typeof window !== 'undefined' && window.innerWidth < 768 ? 'card' : 'table');
        return {
            isDark: isDark,
            currentLocale: savedLocale,
            activeTab: 'dashboard',
            selectedStrategy: 'resonance',
            selectedIndex: '',
            viewMode: savedViewMode, // 'table' | 'card'
            expandedDrawerTicker: null,
            expandedDrawerData: null,
            expandedDrawerLoading: false,

            strategyOptions: [
                {
                    id: 'resonance',
                    label: '綜合優選',
                    icon: 'military_tech',
                    description: '多模型同步看漲，且具備季線支撐或法人籌碼。'
                },
                {
                    id: 'timestral_bullish',
                    label: 'Timestral 看漲排行',
                    icon: 'smart_toy',
                    description: 'Timestral 預測隔日漲幅排行。'
                },
                {
                    id: 'timesfm_bullish',
                    label: 'TimesFM 看漲排行',
                    icon: 'auto_awesome',
                    description: 'TimesFM 預測隔日漲幅排行。'
                },
                {
                    id: 'moirai_bullish',
                    label: 'MOIRAI-2 看漲排行',
                    icon: 'psychology',
                    description: 'MOIRAI-2 預測隔日漲幅排行。'
                },
                {
                    id: 'xuantie',
                    label: '季線支撐買點',
                    icon: 'show_chart',
                    description: '股價回測季線或半年線支撐。'
                },
                {
                    id: 'timestral_bearish',
                    label: '看跌避險排行',
                    icon: 'shield',
                    description: '模型預測隔日跌幅排行，注意避險防守。'
                },
                {
                    id: 'latest',
                    label: '全市場最新總表',
                    icon: 'dataset',
                    description: '【全市場快照】最新批次所有追蹤標的之現價、隔日目標價、估值與籌碼總覽。'
                }
            ],

            macro: {},
            showMacroDetails: localStorage.getItem('stock_show_macro_details') !== 'false',
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
            instSyncing: false,

            // Broker Radar Data
            brokerSearchTicker: '2330',
            brokerDays: 20,
            brokerLoading: false,
            brokerDates: [],
            brokerSeries: [],
            brokerSummary: {},

            // Industry Value Chain Knowledge Graph
            industryThemes: [],
            selectedTheme: 'ALL',
            selectedRelationType: 'ALL',
            industryGraphData: {
                summary: { upstream_count: 0, midstream_count: 0, downstream_count: 0, edge_count: 0 },
                upstream_nodes: [],
                midstream_nodes: [],
                downstream_nodes: [],
                edges: []
            },
            selectedNodeTicker: null,
            highlightedNodes: [],
            highlightedEdges: [],
            industryLoading: false,
            graphZoom: 1.0,
            hideUnconnectedNodes: false,
            focusConnectedOnly: true, // 聚焦模式時只顯示直連鏈路節點（精簡隱藏無關節點）
            bezierPaths: [],

            // Ticker Explorer
            performanceMarket: 'TW',
            performanceWindow: 60,
            performanceHorizon: 1,
            performanceRevision: '',
            performanceView: 'stocks',
            performanceLoading: false,
            performanceError: '',
            performanceRequestId: 0,
            performanceStocks: null,
            performanceModels: null,
            historyPerformance: null,
            historyPerformanceError: '',
            showPerformanceLeaderboard: false,
            searchTicker: '2330',
            historyRecords: [],
            historyLoading: false,
            priceChartVisibility: {
                price: true,
                timestral: true,
                timestralBands: true,
                timesfm: true,
                moirai: true,
                resonance: false,
                ma60: false
            },
            priceChartRange: 'ALL', // 'ALL' | '130' | '66' | '45' | '22'
            tickerInstHistory: [],
            tickerInstLoading: false,
            hasSearched: false,
            resolvedTicker: '',
            resolvedName: '',
            historyError: '',
            historyRequestId: 0,
            historyCompanyData: null,
            historyCompanyLoading: false,
            historyDetailSubTab: 'profile', // 'profile' | 'network' | 'earnings' | 'tender'
            historyEarningsList: [],
            historyDetailsCollapsed: false,
            historyDecisionVerdict: null,
            historyDecisionLoading: false,

            // Earnings Calls State (Task B)
            upcomingEarningsCalls: [],
            earningsLoading: false,
            earningsMarket: 'ALL', // 'ALL' | 'TW' | 'US' | 'JP' | 'KR'
            earningsTimeFilter: 'ALL', // 'ALL' | 'TODAY' | 'UPCOMING' | 'PAST'
            earningsTodayCount: 0,
            earningsUpcomingCount: 0,
            earningsPastCount: 0,
            earningsWeightOnly: false,
            drawerActiveSubTab: 'profile', // 'profile' | 'network' | 'earnings' | 'tender'
            companyTenderLoading: false,
            drawerEarningsMap: {},
            drawerEarningsLoading: false,
            showEarningsModal: false,
            calendarTodayStr: (() => {
                const d = new Date();
                return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
            })(),
            calendarYear: new Date().getFullYear(),
            calendarMonth: new Date().getMonth() + 1,
            calendarData: [],
            calendarLoading: false,
            calendarTimezone: 'CST', // 'CST' | 'EST'
            calendarSearch: '',
            calendarTimeFilter: 'ALL', // 'ALL' | 'TODAY' | 'UPCOMING' | 'PAST'

            // Podcast Radar State (Task C) - Persistent Cache Hydration (SWR)
            podcastEpisodes: (() => {
                try {
                    const cached = localStorage.getItem('stock_quant_podcast_episodes_cache');
                    return cached ? JSON.parse(cached) : [];
                } catch (e) {
                    return [];
                }
            })(),
            podcastChannels: (() => {
                try {
                    const cached = localStorage.getItem('stock_quant_podcast_channels_cache');
                    return cached ? JSON.parse(cached) : [];
                } catch (e) {
                    return [];
                }
            })(),
            podcastTopics: (() => {
                try {
                    const cached = localStorage.getItem('stock_quant_podcast_topics_cache');
                    return cached ? JSON.parse(cached) : [];
                } catch (e) {
                    return [];
                }
            })(),
            podcastLoading: false,
            podcastSelectedTopic: '',
            podcastSelectedChannel: 'ALL',
            podcastSelectedLang: 'ALL',
            podcastSelectedImportance: 'ALL', // 'ALL' | 'heavyweight' | 'important'
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

            // AI Tech Radar & Discovery State
            discoveryCandidates: (() => {
                try {
                    const cached = localStorage.getItem('stock_quant_discovery_candidates_cache');
                    return cached ? JSON.parse(cached) : [];
                } catch (e) {
                    return [];
                }
            })(),
            discoveryPapers: (() => {
                try {
                    const cached = localStorage.getItem('stock_quant_discovery_papers_cache');
                    return cached ? JSON.parse(cached) : [];
                } catch (e) {
                    return [];
                }
            })(),
            discoverySignals: (() => {
                try {
                    const cached = localStorage.getItem('stock_quant_discovery_signals_cache');
                    return cached ? JSON.parse(cached) : [];
                } catch (e) {
                    return [];
                }
            })(),
            discoveryHistory: [],
            discoveryStatus: {
                status: 'success',
                stale: false,
                as_of: '',
                candidate_count: 0,
                paper_count: 0,
                signal_count: 0
            },
            discoveryLoading: false,
            discoverySelectedTheme: 'ALL',
            discoveryExpandedDossiers: {},

            pythonSnippet: `import requests\n\n# 取得三重共振焦點推薦標的\nres = requests.get(f"${window.location.origin}/api/v1/predictions/resonance")\npicks = res.json().get("data", [])\nfor p in picks:\n    print(f"{p['ticker']} | 潛力:{p['potential']}% | 標籤:{p['tags']}")`
        };
    },
    computed: {
        currentStrategyMeta() {
            return this.strategyOptions.find(s => s.id === this.selectedStrategy) || this.strategyOptions[0];
        },
        uniqueDiscoveryCandidates() {
            return this.deduplicateCandidates(this.discoveryCandidates || []);
        },
        discoveryUniqueCandidateCount() {
            return this.uniqueDiscoveryCandidates.length;
        },
        discoveryProposeAddCount() {
            return this.uniqueDiscoveryCandidates.filter(c => c.recommendation === 'propose_add').length;
        },
        discoveryObserveCount() {
            return this.uniqueDiscoveryCandidates.filter(c => c.recommendation === 'observe').length;
        },
        discoveryInPoolCount() {
            return this.uniqueDiscoveryCandidates.filter(c => c.recommendation === 'already_in_pool' || c.in_pool).length;
        },
        discoveryAvgScore() {
            const list = this.uniqueDiscoveryCandidates;
            if (!list.length) return 0;
            const sum = list.reduce((acc, c) => acc + (Number(c.total_score) || 0), 0);
            return sum / list.length;
        },
        discoverySparklinePath() {
            const hist = this.discoveryHistory || [];
            if (!hist.length) {
                return "M 0 16 Q 25 8, 50 14 T 100 10";
            }
            const points = hist.slice(-10).map((h, i, arr) => {
                const x = (i / Math.max(1, arr.length - 1)) * 100;
                const score = Number(h.avg_total_score) || 75;
                const y = Math.max(2, Math.min(22, 24 - (score / 100) * 22));
                return `${x.toFixed(1)},${y.toFixed(1)}`;
            });
            return "M " + points.map((p, idx) => (idx === 0 ? p : `L ${p}`)).join(" ");
        },
        filteredNewCandidates() {
            if (this.discoverySelectedTheme === 'ALL') {
                return this.uniqueDiscoveryCandidates.filter(c => c.recommendation === 'propose_add' || c.recommendation === 'observe');
            }
            const list = (this.discoveryCandidates || []).filter(
                c => c.theme_id === this.discoverySelectedTheme && (c.recommendation === 'propose_add' || c.recommendation === 'observe')
            );
            const deduped = this.deduplicateCandidates(list);
            const globalThemeMap = new Map(this.uniqueDiscoveryCandidates.map(u => [u.ticker, u.all_themes]));
            for (const c of deduped) {
                if (globalThemeMap.has(c.ticker)) {
                    c.all_themes = globalThemeMap.get(c.ticker);
                }
            }
            return deduped;
        },
        filteredInPoolCandidates() {
            if (this.discoverySelectedTheme === 'ALL') {
                return this.uniqueDiscoveryCandidates.filter(c => c.recommendation === 'already_in_pool' || c.in_pool);
            }
            const list = (this.discoveryCandidates || []).filter(
                c => c.theme_id === this.discoverySelectedTheme && (c.recommendation === 'already_in_pool' || c.in_pool)
            );
            const deduped = this.deduplicateCandidates(list);
            const globalThemeMap = new Map(this.uniqueDiscoveryCandidates.map(u => [u.ticker, u.all_themes]));
            for (const c of deduped) {
                if (globalThemeMap.has(c.ticker)) {
                    c.all_themes = globalThemeMap.get(c.ticker);
                }
            }
            return deduped;
        },
        filteredPapers() {
            let list = this.discoveryPapers || [];
            if (this.discoverySelectedTheme !== 'ALL') {
                list = list.filter(p => p.bottleneck_tag === this.discoverySelectedTheme);
            }
            return list;
        },
        filteredSignals() {
            let list = this.discoverySignals || [];
            if (this.discoverySelectedTheme !== 'ALL') {
                list = list.filter(s => s.theme_id === this.discoverySelectedTheme);
            }
            return list;
        },
        dailyHistoryRows() {
            if (!this.historyRecords || !this.historyRecords.length) return [];

            // 1. Group records by Date (YYYY-MM-DD)
            const dateMap = new Map();
            for (const r of this.historyRecords) {
                const rawTs = String(r.forecast_session || r.timestamp || '').trim();
                if (rawTs.length < 10) continue;
                const dateKey = rawTs.slice(0, 10);
                if (!dateMap.has(dateKey)) {
                    dateMap.set(dateKey, {
                        date: dateKey,
                        closePrice: null,
                        timesfmPred: null,
                        timestralPred: null,
                        moiraiPred: null,
                        timesfmActual: null,
                        timestralActual: null,
                        moiraiActual: null,
                        timesfmTarget: null,
                        timestralTarget: null,
                        moiraiTarget: null,
                        timesfmIssued: '',
                        timestralIssued: '',
                        moiraiIssued: '',
                        ma60: null,
                        pe: null,
                        pb: null,
                        pullback_type: null,
                        trust_net_5d: null,
                        foreign_net_5d: null
                    });
                }
                const item = dateMap.get(dateKey);
                const curP = Number(r.current_price);
                if (!isNaN(curP) && curP > 0) item.closePrice = curP;

                const ma60 = Number(r.ma60);
                if (!isNaN(ma60) && ma60 > 0) item.ma60 = ma60;
                if (r.pe !== null && r.pe !== undefined && !isNaN(Number(r.pe))) item.pe = Number(r.pe);
                if (r.pb !== null && r.pb !== undefined && !isNaN(Number(r.pb))) item.pb = Number(r.pb);
                if (r.pullback_type) item.pullback_type = r.pullback_type;
                if (r.trust_net_5d !== null && r.trust_net_5d !== undefined && Number.isFinite(Number(r.trust_net_5d))) item.trust_net_5d = Number(r.trust_net_5d);
                if (r.foreign_net_5d !== null && r.foreign_net_5d !== undefined && Number.isFinite(Number(r.foreign_net_5d))) item.foreign_net_5d = Number(r.foreign_net_5d);

                if (r.timesfm_predicted_price && !item.timesfmIssued) {
                    const tfp = Number(r.timesfm_predicted_price);
                    if (!isNaN(tfp) && tfp > 0) item.timesfmPred = tfp;
                }
                if (r.timestral_predicted_price && !item.timestralIssued) {
                    const tmp = Number(r.timestral_predicted_price);
                    if (!isNaN(tmp) && tmp > 0) item.timestralPred = tmp;
                }

                const predP = Number(r.predicted_price);
                const verified = r.verification_status === 'settled' &&
                    /^\d{4}-\d{2}-\d{2}$/.test(String(r.target_date || '').slice(0, 10)) &&
                    Number.isFinite(Number(r.actual_price)) && Number(r.actual_price) > 0;
                if (!isNaN(predP) && predP > 0 && Number(r.horizon || 1) === 1) {
                    const strat = ((r.strategy_type || '') + ' ' + (r.model_name || '')).toLowerCase();
                    const issued = String(r.timestamp || '').trim();
                    if (strat.includes('timesfm') && issued >= item.timesfmIssued) {
                        item.timesfmPred = predP;
                        item.timesfmIssued = issued;
                        item.timesfmActual = verified ? Number(r.actual_price) : null;
                        item.timesfmTarget = /^\d{4}-\d{2}-\d{2}$/.test(String(r.target_date || '').slice(0, 10)) ? String(r.target_date).slice(0, 10) : null;
                    } else if (strat.includes('timestral') && issued >= item.timestralIssued) {
                        item.timestralPred = predP;
                        item.timestralIssued = issued;
                        item.timestralActual = verified ? Number(r.actual_price) : null;
                        item.timestralTarget = /^\d{4}-\d{2}-\d{2}$/.test(String(r.target_date || '').slice(0, 10)) ? String(r.target_date).slice(0, 10) : null;
                    } else if (strat.includes('moirai') && issued >= item.moiraiIssued) {
                        item.moiraiPred = predP;
                        item.moiraiIssued = issued;
                        item.moiraiActual = verified ? Number(r.actual_price) : null;
                        item.moiraiTarget = /^\d{4}-\d{2}-\d{2}$/.test(String(r.target_date || '').slice(0, 10)) ? String(r.target_date).slice(0, 10) : null;
                    }
                }
            }

            // Actuals only come from the corresponding verified forecast record.
            const sortedDates = Array.from(dateMap.keys()).sort();
            const n = sortedDates.length;
            const rows = [];

            for (let i = 0; i < n; i++) {
                const date = sortedDates[i];
                const item = dateMap.get(date);
                const closePrice = item.closePrice;
                const targets = [item.timesfmTarget, item.timestralTarget, item.moiraiTarget].filter(Boolean);
                const actuals = [item.timesfmActual, item.timestralActual, item.moiraiActual].filter(v => v !== null && v !== undefined);
                const allTargetsMatch = targets.length <= 1 || targets.every(t => t === targets[0]);
                const allActualsMatch = actuals.length <= 1 || actuals.every(a => Math.abs(a - actuals[0]) < 1e-4);
                const sameTarget = allTargetsMatch && allActualsMatch;
                let nextDate = sameTarget && targets.length > 0 ? targets[0] : null;
                let nextActualPrice = sameTarget && actuals.length > 0 ? actuals[0] : null;

                // 🌟 向下相容歷史資料：若無新版 verification 結算資料，則 fallback 使用連續交易日次日真實收盤開獎
                if (nextActualPrice === null && i < n - 1) {
                    const fallbackNextDate = sortedDates[i + 1];
                    const nextItem = dateMap.get(fallbackNextDate);
                    if (nextItem && nextItem.closePrice !== null && nextItem.closePrice > 0) {
                        nextDate = fallbackNextDate;
                        nextActualPrice = nextItem.closePrice;
                    }
                }

                const actualReturnPct = (nextActualPrice !== null && nextActualPrice > 0 && closePrice !== null && closePrice > 0)
                    ? ((nextActualPrice - closePrice) / closePrice) * 100
                    : null;

                // TimesFM metrics
                let timesfmPot = null;
                let timesfmHit = null;
                let timesfmErrorPct = null;
                if (item.timesfmPred !== null && closePrice !== null && closePrice > 0) {
                    timesfmPot = ((item.timesfmPred - closePrice) / closePrice) * 100;
                }
                const tfActual = item.timesfmActual !== null ? item.timesfmActual : nextActualPrice;
                if (tfActual !== null && item.timesfmPred !== null && tfActual > 0) {
                    timesfmErrorPct = Math.abs(item.timesfmPred - tfActual) / tfActual * 100;
                    if (closePrice !== null && timesfmPot !== null) {
                        timesfmHit = Math.sign(item.timesfmPred - closePrice) === Math.sign(tfActual - closePrice);
                    }
                }

                // Timestral metrics
                let timestralPot = null;
                let timestralHit = null;
                let timestralErrorPct = null;
                if (item.timestralPred !== null && closePrice !== null && closePrice > 0) {
                    timestralPot = ((item.timestralPred - closePrice) / closePrice) * 100;
                }
                const tmActual = item.timestralActual !== null ? item.timestralActual : nextActualPrice;
                if (tmActual !== null && item.timestralPred !== null && tmActual > 0) {
                    timestralErrorPct = Math.abs(item.timestralPred - tmActual) / tmActual * 100;
                    if (closePrice !== null && timestralPot !== null) {
                        timestralHit = Math.sign(item.timestralPred - closePrice) === Math.sign(tmActual - closePrice);
                    }
                }

                // MOIRAI-2 metrics
                let moiraiPot = null;
                let moiraiHit = null;
                let moiraiErrorPct = null;
                if (item.moiraiPred !== null && closePrice !== null && closePrice > 0) {
                    moiraiPot = ((item.moiraiPred - closePrice) / closePrice) * 100;
                }
                const mrActual = item.moiraiActual !== null ? item.moiraiActual : nextActualPrice;
                if (mrActual !== null && item.moiraiPred !== null && mrActual > 0) {
                    moiraiErrorPct = Math.abs(item.moiraiPred - mrActual) / mrActual * 100;
                    if (closePrice !== null && moiraiPot !== null) {
                        moiraiHit = Math.sign(item.moiraiPred - closePrice) === Math.sign(mrActual - closePrice);
                    }
                }

                // Daily winner among active predictions
                let winner = null;
                if (nextActualPrice !== null) {
                    const validErrors = [];
                    if (timesfmErrorPct !== null) validErrors.push({ name: 'TimesFM', err: timesfmErrorPct });
                    if (timestralErrorPct !== null) validErrors.push({ name: 'Timestral', err: timestralErrorPct });
                    if (moiraiErrorPct !== null) validErrors.push({ name: 'MOIRAI-2', err: moiraiErrorPct });

                    if (validErrors.length > 1) {
                        validErrors.sort((a, b) => a.err - b.err);
                        if (Math.abs(validErrors[0].err - validErrors[1].err) < 0.05) {
                            winner = '平手';
                        } else {
                            winner = validErrors[0].name + ' 勝';
                        }
                    } else if (validErrors.length === 1) {
                        winner = validErrors[0].name;
                    }
                }

                rows.push({
                    date,
                    closePrice,
                    nextDate,
                    nextActualPrice,
                    actualReturnPct,
                    timesfmPred: item.timesfmPred,
                    timesfmPot,
                    timesfmHit,
                    timesfmErrorPct,
                    timestralPred: item.timestralPred,
                    timestralPot,
                    timestralHit,
                    timestralErrorPct,
                    moiraiPred: item.moiraiPred,
                    moiraiPot,
                    moiraiHit,
                    moiraiErrorPct,
                    winner,
                    ma60: item.ma60,
                    pullback_type: item.pullback_type,
                    pe: item.pe,
                    pb: item.pb,
                    trust_net_5d: item.trust_net_5d,
                    foreign_net_5d: item.foreign_net_5d
                });
            }

            // Return sorted descending (newest first for table display)
            return rows.reverse();
        },
        modelScorecard() {
            const rows = this.dailyHistoryRows;
            if (!rows || !rows.length) return null;

            let tfCount = 0, tfHitCount = 0, tfErrorSum = 0, tfH2HCount = 0;
            let tmCount = 0, tmHitCount = 0, tmErrorSum = 0, tmH2HCount = 0;
            let mrCount = 0, mrHitCount = 0, mrErrorSum = 0, mrH2HCount = 0;
            let headToHeadCount = 0, tfWins = 0, tmWins = 0, mrWins = 0, ties = 0;

            for (const r of rows) {
                if (r.nextActualPrice === null) continue;

                const hasTf = r.timesfmErrorPct !== null;
                const hasTm = r.timestralErrorPct !== null;
                const hasMr = r.moiraiErrorPct !== null;

                if (hasTf) {
                    tfCount++;
                    tfErrorSum += r.timesfmErrorPct;
                    if (r.timesfmHit === true) tfHitCount++;
                }

                if (hasTm) {
                    tmCount++;
                    tmErrorSum += r.timestralErrorPct;
                    if (r.timestralHit === true) tmHitCount++;
                }

                if (hasMr) {
                    mrCount++;
                    mrErrorSum += r.moiraiErrorPct;
                    if (r.moiraiHit === true) mrHitCount++;
                }

                const activeInDay = (hasTf ? 1 : 0) + (hasTm ? 1 : 0) + (hasMr ? 1 : 0);
                if (activeInDay >= 2) {
                    headToHeadCount++;
                    if (hasTf) tfH2HCount++;
                    if (hasTm) tmH2HCount++;
                    if (hasMr) mrH2HCount++;

                    if (r.winner === 'TimesFM 勝') tfWins++;
                    else if (r.winner === 'Timestral 勝') tmWins++;
                    else if (r.winner === 'MOIRAI-2 勝') mrWins++;
                    else if (r.winner === '平手') ties++;
                }
            }

            if (tfCount === 0 && tmCount === 0 && mrCount === 0) return null;

            const tfHitRate = tfCount > 0 ? (tfHitCount / tfCount) * 100 : null;
            const tfMape = tfCount > 0 ? (tfErrorSum / tfCount) : null;

            const tmHitRate = tmCount > 0 ? (tmHitCount / tmCount) * 100 : null;
            const tmMape = tmCount > 0 ? (tmErrorSum / tmCount) : null;

            const mrHitRate = mrCount > 0 ? (mrHitCount / mrCount) * 100 : null;
            const mrMape = mrCount > 0 ? (mrErrorSum / mrCount) : null;

            const candidates = [
                { name: 'TimesFM', count: tfCount, hitRate: tfHitRate, mape: tfMape, wins: tfWins },
                { name: 'Timestral', count: tmCount, hitRate: tmHitRate, mape: tmMape, wins: tmWins },
                { name: 'MOIRAI-2', count: mrCount, hitRate: mrHitRate, mape: mrMape, wins: mrWins }
            ];

            const eligible = candidates.filter(c => c.count >= 2 && c.hitRate !== null && c.hitRate >= 50.0);

            let overallWinner = '模型表現均衡';
            let winningModel = null;

            if (eligible.length === 0) {
                winningModel = null;
                const active = candidates.filter(c => c.count >= 2 && c.hitRate !== null);
                if (active.length > 0 && active.every(c => c.hitRate < 50.0)) {
                    overallWinner = '⚠️ 各模型於該股方向勝率皆低於 50%，建議以均線支撐與籌碼為主';
                } else {
                    overallWinner = '⏳ 驗證樣本累積中（需至少 2 日開獎紀錄且方向勝率 ≥ 50%）';
                }
            } else {
                eligible.sort((a, b) => {
                    if (Math.abs(a.hitRate - b.hitRate) >= 5.0) {
                        return b.hitRate - a.hitRate;
                    }
                    if (a.wins !== b.wins) {
                        return b.wins - a.wins;
                    }
                    return (a.mape || 999) - (b.mape || 999);
                });

                if (eligible.length > 1 && Math.abs(eligible[0].hitRate - eligible[1].hitRate) < 5.0 && eligible[0].wins === eligible[1].wins && Math.abs((eligible[0].mape || 0) - (eligible[1].mape || 0)) < 0.01) {
                    winningModel = null;
                    overallWinner = `⚖️ 該股歷史表現：模型實力相當（方向勝率 ${eligible[0].hitRate.toFixed(1)}% · 平均誤差 ±${eligible[0].mape.toFixed(2)}%）`;
                } else {
                    winningModel = eligible[0].name;
                    overallWinner = `🏆 該股表現較佳：${winningModel}（方向勝率 ${eligible[0].hitRate.toFixed(1)}% · 平均誤差 ±${eligible[0].mape.toFixed(2)}%）`;
                }
            }

            return {
                headToHeadCount,
                ties,
                timesfm: {
                    count: tfCount,
                    hitCount: tfHitCount,
                    hitRate: tfHitRate !== null ? tfHitRate.toFixed(1) : '-',
                    mape: tfMape !== null ? tfMape.toFixed(2) : '-',
                    wins: tfWins,
                    h2hCount: tfH2HCount,
                    isLeader: winningModel === 'TimesFM',
                    lowHitWarning: tfCount >= 2 && tfHitRate !== null && tfHitRate < 50
                },
                timestral: {
                    count: tmCount,
                    hitCount: tmHitCount,
                    hitRate: tmHitRate !== null ? tmHitRate.toFixed(1) : '-',
                    mape: tmMape !== null ? tmMape.toFixed(2) : '-',
                    wins: tmWins,
                    h2hCount: tmH2HCount,
                    isLeader: winningModel === 'Timestral',
                    lowHitWarning: tmCount >= 2 && tmHitRate !== null && tmHitRate < 50
                },
                moirai: {
                    count: mrCount,
                    hitCount: mrHitCount,
                    hitRate: mrHitRate !== null ? mrHitRate.toFixed(1) : '-',
                    mape: mrMape !== null ? mrMape.toFixed(2) : '-',
                    wins: mrWins,
                    h2hCount: mrH2HCount,
                    isLeader: winningModel === 'MOIRAI-2',
                    lowHitWarning: mrCount >= 2 && mrHitRate !== null && mrHitRate < 50
                },
                overallWinner
            };
        }
    },
    methods: {
        formatPerformancePercent(value, digits = 1) {
            const number = Number(value);
            return value === null || value === undefined || !Number.isFinite(number)
                ? '—' : `${(number * 100).toFixed(digits)}%`;
        },
        performanceParams(market = this.performanceMarket) {
            const params = new URLSearchParams({ market, window: String(this.performanceWindow),
                horizon: String(this.performanceHorizon), target_type: 'price', limit: '100' });
            if (this.performanceRevision.trim()) params.set('model_revision', this.performanceRevision.trim());
            return params;
        },
        async loadPerformance() {
            const requestId = ++this.performanceRequestId;
            this.performanceLoading = true;
            this.performanceError = '';
            const params = this.performanceParams().toString();
            try {
                const [stocks, models] = await Promise.all([
                    fetch(`/api/v1/performance/stocks?${params}`),
                    fetch(`/api/v1/performance/models?${params}`)
                ]);
                if (!stocks.ok || !models.ok) throw new Error('目前無法讀取已驗證的預測資料');
                const [stockData, modelData] = await Promise.all([stocks.json(), models.json()]);
                if (requestId !== this.performanceRequestId) return;
                this.performanceStocks = stockData;
                this.performanceModels = modelData;
                if (this.resolvedTicker) this.loadHistoryPerformance(this.resolvedTicker, this.historyRequestId);
            } catch (error) {
                if (requestId !== this.performanceRequestId) return;
                this.performanceStocks = null;
                this.performanceModels = null;
                this.performanceError = '目前無法取得預測成績，請稍後重試。';
            } finally {
                if (requestId === this.performanceRequestId) this.performanceLoading = false;
            }
        },
        async loadHistoryPerformance(ticker, requestId) {
            this.historyPerformance = null;
            this.historyPerformanceError = '';
            const market = /\.(TW|TWO)$/i.test(ticker) || /^\d+$/.test(ticker) ? 'TW'
                : (/\.T$/i.test(ticker) ? 'JP' : 'US');
            try {
                const response = await fetch(`/api/v1/performance/tickers/${encodeURIComponent(ticker)}?${this.performanceParams(market)}`);
                if (!response.ok) throw new Error('performance unavailable');
                const data = await response.json();
                if (requestId === this.historyRequestId) this.historyPerformance = data;
            } catch (error) {
                if (requestId === this.historyRequestId) this.historyPerformanceError = '成績暫時無法取得，請稍後重試。';
            }
        },
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
        toggleLocale() {
            this.currentLocale = this.currentLocale === 'zh' ? 'en' : 'zh';
            localStorage.setItem('stock_quant_locale', this.currentLocale);
            this.showToast(this.currentLocale === 'zh' ? '已切換為繁體中文' : 'Language switched to English');
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
        toggleMacroDetails() {
            this.showMacroDetails = !this.showMacroDetails;
            localStorage.setItem('stock_show_macro_details', this.showMacroDetails);
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
        async syncInstitutionalNow() {
            if (this.instSyncing) return;
            this.instSyncing = true;
            try {
                await ApiClient.syncInstitutionalFlows();
                this.showToast('已在背景啟動三大法人歷史籌碼增量同步，正在更新至最新交易日！', 'success');
                setTimeout(async () => {
                    await Promise.all([
                        this.loadMarketStats(),
                        this.loadAvailableDates()
                    ]);
                    if (this.availableDates.length > 0) {
                        this.instSelectedDate = this.availableDates[0];
                    }
                    await this.loadInstitutionalFlows();
                    this.instSyncing = false;
                }, 3500);
            } catch (e) {
                this.showToast('法人籌碼同步啟動失敗: ' + (e.message || e), 'error');
                this.instSyncing = false;
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
                // 計算候選標的之最新批次時間戳記 (優先使用 API 回傳，備援計算所有記錄之 MAX 時間戳，避免被單一歷史記錄誤導)
                const maxRecordTs = this.records.reduce((max, r) => {
                    const ts = r.data_as_of || r.timestamp || '';
                    return ts > max ? ts : max;
                }, '');
                const effectiveTs = data.latest_batch_timestamp || maxRecordTs || '';
                this.batchTimestamp = effectiveTs ? effectiveTs.replace('T', ' ').substring(0, 19) : '';
                this.batchDate = data.batch_date || (effectiveTs ? effectiveTs.substring(0, 10) : '');
                this.batchStale = Boolean(data.is_stale || this.records.some(r => r.is_stale));
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
            this.historyPerformance = null;
            this.historyPerformanceError = '';
            this.tickerInstHistory = [];
            chartManager.destroy('pricePredictChart');
            chartManager.destroy('instChart');
            this.resolvedTicker = '';
            this.resolvedName = '';
            this.historyCompanyData = null;
            this.historyEarningsList = [];
            this.historyDetailsCollapsed = false;
            this.historyDecisionVerdict = null;
            this.historyDecisionLoading = false;
            this.updateUrl();

            try {
                const resolution = await ApiClient.resolveTicker(query);
                if (requestId !== this.historyRequestId) return;
                this.resolvedTicker = resolution.ticker;
                this.resolvedName = resolution.name || '';
                this.searchTicker = this.formatTicker(resolution.ticker);
                this.updateUrl(true);
                this.loadHistoryPerformance(resolution.ticker, requestId);

                // 若非台股標的且當前子頁籤為標案或股權網絡，自動重設為基本面與新聞
                if (!this.isTwStock(resolution.ticker) && (this.historyDetailSubTab === 'tender' || this.historyDetailSubTab === 'network')) {
                    this.historyDetailSubTab = 'profile';
                }

                // 並行啟動個股情報中心與綜合決策裁判查詢
                this.loadHistoryCompanyData(resolution.ticker, requestId);
                this.loadHistoryDecision(resolution.ticker, requestId);

                const [historyData, instData] = await Promise.all([
                    ApiClient.getPredictionHistory(resolution.ticker, 240).catch(() => []),
                    ApiClient.getInstitutionalHistory(resolution.ticker, 60).catch(() => [])
                ]);

                if (requestId !== this.historyRequestId) return;
                this.historyRecords = historyData;
                this.tickerInstHistory = instData;

                this.$nextTick(() => {
                    if (this.historyRecords.length > 0) {
                        this.renderPricePredictChart();
                    }
                    if (this.tickerInstHistory.length > 0) {
                        this.renderInstChart();
                    }
                });
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
        async loadHistoryDecision(ticker, requestId = null, forceRefresh = false) {
            if (!ticker) return;
            this.historyDecisionLoading = true;
            try {
                const res = await ApiClient.getStockDecision(ticker, forceRefresh);
                if (requestId !== null && requestId !== this.historyRequestId) return;
                if (res && res.verdict) {
                    this.historyDecisionVerdict = res.verdict;
                }
            } catch (e) {
                console.warn("Failed to load stock decision verdict:", e);
                if (requestId !== null && requestId !== this.historyRequestId) return;
            } finally {
                if (!requestId || requestId === this.historyRequestId) {
                    this.historyDecisionLoading = false;
                }
            }
        },
        async loadHistoryCompanyData(ticker, requestId) {
            if (!ticker) return;
            this.historyCompanyLoading = true;

            // Fast-path: SWR 快取即時預先水合
            if (this.companyProfileCache[ticker]) {
                this.historyCompanyData = { ...this.companyProfileCache[ticker] };
            } else {
                this.historyCompanyData = {
                    ticker: ticker,
                    name: this.resolvedName || ticker,
                    industry: '載入中...',
                    business_summary: '正在檢索基本資料與即時新聞...',
                    recent_news: []
                };
            }

            if (this.drawerEarningsMap[ticker]) {
                this.historyEarningsList = this.drawerEarningsMap[ticker];
            } else {
                this.historyEarningsList = [];
            }

            try {
                const isTw = this.isTwStock(ticker);
                const [profRes, netRes, earnRes, tenderRes] = await Promise.allSettled([
                    ApiClient.getCompanyProfile(ticker),
                    isTw ? ApiClient.getCompanyNetwork(ticker) : Promise.resolve(null),
                    ApiClient.getTickerEarningsCalls(ticker),
                    isTw ? ApiClient.getCompanyTender(ticker) : Promise.resolve(null)
                ]);

                if (requestId !== this.historyRequestId) return;

                const prof = (profRes.status === 'fulfilled' && profRes.value) ? profRes.value : (this.companyProfileCache[ticker] || {});
                let net = (netRes.status === 'fulfilled') ? netRes.value : (prof.company_network || null);
                if (net && net.data) net = net.data;
                const earnVal = (earnRes.status === 'fulfilled' && earnRes.value) ? earnRes.value : null;
                const earnList = earnVal ? (earnVal.data || earnVal.items || (Array.isArray(earnVal) ? earnVal : [])) : [];
                let tender = (tenderRes.status === 'fulfilled') ? tenderRes.value : (prof.company_tender || null);
                if (tender && tender.data) tender = tender.data;

                this.historyEarningsList = Array.isArray(earnList) ? earnList : [];
                this.drawerEarningsMap[ticker] = this.historyEarningsList;

                const fullData = {
                    ...prof,
                    ticker: ticker,
                    name: prof.name || this.resolvedName || ticker,
                    company_network: net,
                    company_tender: tender
                };

                if (net) {
                    if (!fullData.links) fullData.links = {};
                    if (net.web_url) fullData.links.network_graph = net.web_url;
                    if (net.image_url) fullData.links.network_image = net.image_url;
                }

                this.historyCompanyData = fullData;
                this.companyProfileCache[ticker] = fullData;
            } catch (e) {
                console.warn(`Failed to load history company data for ${ticker}:`, e);
            } finally {
                if (requestId === this.historyRequestId) {
                    this.historyCompanyLoading = false;
                }
            }
        },
        toggleHistoryDetails() {
            this.historyDetailsCollapsed = !this.historyDetailsCollapsed;
        },
        setHistoryDetailSubTab(tab) {
            this.historyDetailSubTab = tab;
            const ticker = this.resolvedTicker || this.searchTicker;
            if (!ticker) return;
            if (tab === 'tender' && (!this.historyCompanyData || !this.historyCompanyData.company_tender)) {
                this.ensureCompanyTenderLoaded(ticker);
            } else if (tab === 'network' && (!this.historyCompanyData || !this.historyCompanyData.company_network)) {
                this.ensureCompanyNetworkLoaded(ticker);
            }
        },
        togglePriceChartSeries(seriesKey) {
            if (this.priceChartVisibility[seriesKey] === undefined) return;
            this.priceChartVisibility[seriesKey] = !this.priceChartVisibility[seriesKey];
            const updated = typeof chartManager.setDatasetVisibility === 'function' &&
                chartManager.setDatasetVisibility('pricePredictChart', seriesKey, this.priceChartVisibility[seriesKey]);
            if (!updated && typeof chartManager.renderPricePredictChart === 'function') {
                this.renderPricePredictChart();
            }
        },
        setPriceChartRange(range) {
            this.priceChartRange = String(range);
            const ok = typeof chartManager.setPriceChartRange === 'function' &&
                chartManager.setPriceChartRange(range === 'ALL' ? 0 : Number(range));
            if (!ok) {
                this.renderPricePredictChart();
            }
        },
        zoomPriceChartIn() {
            this.priceChartRange = 'CUSTOM';
            if (typeof chartManager.zoomPriceChart === 'function') {
                chartManager.zoomPriceChart(1.3);
            }
        },
        zoomPriceChartOut() {
            this.priceChartRange = 'CUSTOM';
            if (typeof chartManager.zoomPriceChart === 'function') {
                chartManager.zoomPriceChart(0.75);
            }
        },
        panPriceChartLeft() {
            this.priceChartRange = 'CUSTOM';
            if (typeof chartManager.panPriceChart === 'function') {
                chartManager.panPriceChart(-0.35);
            }
        },
        panPriceChartRight() {
            this.priceChartRange = 'CUSTOM';
            if (typeof chartManager.panPriceChart === 'function') {
                chartManager.panPriceChart(0.35);
            }
        },
        resetPriceChartZoom() {
            this.priceChartRange = 'ALL';
            if (typeof chartManager.resetPriceChartZoom === 'function') {
                chartManager.resetPriceChartZoom();
            }
        },
        renderPricePredictChart() {
            this.$nextTick(() => {
                if (typeof chartManager.renderPricePredictChart === 'function') {
                    const rangeDays = (this.priceChartRange === 'ALL' || this.priceChartRange === 'CUSTOM') ? 0 : Number(this.priceChartRange);
                    chartManager.renderPricePredictChart('pricePredictChartCanvas', this.historyRecords, this.isDark, this.priceChartVisibility, rangeDays);
                } else {
                    console.warn("chartManager.renderPricePredictChart is not available on active instance");
                }
            });
        },
        renderInstChart() {
            this.$nextTick(() => {
                if (typeof chartManager.renderInstChart === 'function') {
                    chartManager.renderInstChart('instChartCanvas', this.tickerInstHistory, this.isDark);
                }
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
        async toggleRowDrawer(ticker, targetTab = null) {
            this.hideCompanyHover(true);
            if (this.expandedDrawerTicker === ticker) {
                if (targetTab && this.drawerActiveSubTab !== targetTab) {
                    this.drawerActiveSubTab = targetTab;
                    if (targetTab === 'network' && (!this.expandedDrawerData || !this.expandedDrawerData.company_network)) {
                        this.ensureCompanyNetworkLoaded(ticker);
                    } else if (targetTab === 'tender' && (!this.expandedDrawerData || !this.expandedDrawerData.company_tender)) {
                        this.ensureCompanyTenderLoaded(ticker);
                    }
                    return;
                }
                this.expandedDrawerTicker = null;
                this.expandedDrawerData = null;
                return;
            }

            this.expandedDrawerTicker = ticker;
            this.drawerActiveSubTab = targetTab || 'profile';

            // Also load earnings call history for this ticker
            this.loadTickerEarnings(ticker);

            if (this.isTwStock(ticker)) {
                this.ensureCompanyNetworkLoaded(ticker);
                this.ensureCompanyTenderLoaded(ticker);
            }

            if (this.companyProfileCache[ticker]) {
                this.expandedDrawerData = this.companyProfileCache[ticker];
                this.expandedDrawerLoading = false;
                if (targetTab === 'network' && !this.expandedDrawerData.company_network) {
                    this.ensureCompanyNetworkLoaded(ticker);
                } else if (targetTab === 'tender' && !this.expandedDrawerData.company_tender) {
                    this.ensureCompanyTenderLoaded(ticker);
                }
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
        async ensureCompanyNetworkLoaded(ticker) {
            if (!ticker) return;
            try {
                let net = await ApiClient.getCompanyNetwork(ticker);
                if (net && net.data) net = net.data;
                if (net && (net.name || net.relationships)) {
                    if (this.companyProfileCache[ticker]) {
                        this.companyProfileCache[ticker].company_network = net;
                        if (!this.companyProfileCache[ticker].links) this.companyProfileCache[ticker].links = {};
                        if (net.web_url) this.companyProfileCache[ticker].links.network_graph = net.web_url;
                        if (net.image_url) this.companyProfileCache[ticker].links.network_image = net.image_url;
                    }
                    if (this.hoverVisible && this.hoverData?.ticker === ticker) {
                        this.hoverData.company_network = net;
                    }
                    // 僅在當前展開抽屜仍是該股票時才寫入抽屜狀態，防止切換標的造成串台 (Codex P2)
                    if (this.expandedDrawerTicker === ticker) {
                        if (!this.expandedDrawerData) this.expandedDrawerData = { ticker };
                        this.expandedDrawerData.company_network = net;
                        if (!this.expandedDrawerData.links) this.expandedDrawerData.links = {};
                        if (net.web_url) this.expandedDrawerData.links.network_graph = net.web_url;
                        if (net.image_url) this.expandedDrawerData.links.network_image = net.image_url;
                    }
                    if (this.historyCompanyData && (this.historyCompanyData.ticker === ticker || this.resolvedTicker === ticker || this.searchTicker === ticker)) {
                        this.historyCompanyData.company_network = net;
                        if (!this.historyCompanyData.links) this.historyCompanyData.links = {};
                        if (net.web_url) this.historyCompanyData.links.network_graph = net.web_url;
                        if (net.image_url) this.historyCompanyData.links.network_image = net.image_url;
                    }
                }
            } catch (e) {
                console.debug("Failed to lazily load company network:", e);
            }
        },
        async ensureCompanyTenderLoaded(ticker) {
            if (!ticker || !this.isTwStock(ticker)) return;
            if (this.expandedDrawerData && this.expandedDrawerData.company_tender) return;
            this.companyTenderLoading = true;
            try {
                let tender = await ApiClient.getCompanyTender(ticker);
                if (tender && tender.data) tender = tender.data;
                if (tender) {
                    if (this.companyProfileCache[ticker]) {
                        this.companyProfileCache[ticker].company_tender = tender;
                    }
                    if (this.hoverVisible && this.hoverData?.ticker === ticker) {
                        this.hoverData.company_tender = tender;
                    }
                    // 僅在當前展開抽屜仍是該股票時才寫入抽屜狀態，防止切換標的造成串台 (Codex P2)
                    if (this.expandedDrawerTicker === ticker) {
                        if (!this.expandedDrawerData) this.expandedDrawerData = { ticker };
                        this.expandedDrawerData.company_tender = tender;
                    }
                    if (this.historyCompanyData && (this.historyCompanyData.ticker === ticker || this.resolvedTicker === ticker || this.searchTicker === ticker)) {
                        this.historyCompanyData.company_tender = tender;
                    }
                }
            } catch (e) {
                console.debug("Failed to lazily load company tender:", e);
            } finally {
                this.companyTenderLoading = false;
            }
        },
        async loadUpcomingEarnings() {
            this.earningsLoading = true;
            try {
                const res = await ApiClient.getUpcomingEarningsCalls({
                    market: this.earningsMarket,
                    days: 30,
                    isWeightStock: this.earningsWeightOnly ? true : null,
                    timeFilter: this.earningsTimeFilter,
                    limit: 50
                });
                if (res && res.data) {
                    this.upcomingEarningsCalls = res.data;
                    this.earningsTodayCount = res.today_count || 0;
                    this.earningsUpcomingCount = res.upcoming_count || 0;
                    this.earningsPastCount = res.past_count || 0;
                }
            } catch (e) {
                console.warn("Failed to load upcoming earnings calls:", e);
            } finally {
                this.earningsLoading = false;
            }
        },
        setEarningsTimeFilter(tf) {
            this.earningsTimeFilter = tf;
            this.loadUpcomingEarnings();
        },
        setCalendarTimeFilter(tf) {
            this.calendarTimeFilter = tf;
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
            const d = new Date();
            this.calendarTodayStr = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
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
        isCallMatchingSearch(call, query) {
            if (!query) return true;
            const q = query.trim().toLowerCase();
            if (!q) return true;
            return !!(
                (call.ticker && call.ticker.toLowerCase().includes(q)) ||
                (call.name && call.name.toLowerCase().includes(q)) ||
                (call.remarks && call.remarks.toLowerCase().includes(q)) ||
                (call.financial_period && call.financial_period.toLowerCase().includes(q)) ||
                (call.guidance_summary && call.guidance_summary.toLowerCase().includes(q)) ||
                (call.guidance_outlook && call.guidance_outlook.toLowerCase().includes(q)) ||
                (call.guidance_margin && call.guidance_margin.toLowerCase().includes(q)) ||
                (call.guidance_capex && call.guidance_capex.toLowerCase().includes(q)) ||
                (call.guidance_qa && call.guidance_qa.toLowerCase().includes(q))
            );
        },
        isDayMatchingCalendarSearch(day, query) {
            if (!query) return true;
            const q = query.trim().toLowerCase();
            if (!q) return true;
            if (day.summary_companies && day.summary_companies.toLowerCase().includes(q)) return true;
            if (day.calls && day.calls.some(c => this.isCallMatchingSearch(c, q))) return true;
            return false;
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
            this.searchTicker = this.formatTicker(ticker);
            this.activeTab = 'history';
            this.updateUrl();
            this.loadHistory();
            window.scrollTo({ top: 0, behavior: 'smooth' });
        },
        setTab(tab) {
            this.hideCompanyHover(true);
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
                this.loadPerformance();
                if (this.searchTicker && !this.historyRecords.length) {
                    this.loadHistory();
                } else {
                    this.$nextTick(() => {
                        if (this.historyRecords.length > 0) this.renderPricePredictChart();
                        if (this.tickerInstHistory.length > 0) this.renderInstChart();
                    });
                }
            } else if (tab === 'podcast') {
                this.loadPodcastEpisodes();
                this.loadPodcastChannels();
                this.loadPodcastTopics();
            } else if (tab === 'industry_chain') {
                this.loadIndustryChainData();
            } else if (tab === 'discovery') {
                this.loadDiscoveryData();
            }
        },
        async loadPodcastEpisodes() {
            // 若本地已具有快取資料，則不進入全螢幕空白 loading 阻斷，改在背景靜默更新 (SWR)
            if (!this.podcastEpisodes || !this.podcastEpisodes.length) {
                this.podcastLoading = true;
            }
            try {
                const res = await ApiClient.getPodcastEpisodes({
                    topic: this.podcastSelectedTopic || null,
                    channelId: this.podcastSelectedChannel,
                    lang: this.podcastSelectedLang,
                    limit: 100
                });
                if (res && res.data) {
                    this.podcastEpisodes = res.data;
                    // 當為首頁預設篩選時，自動存入 localStorage 提供下次瞬間 0ms 冷啟動
                    if (!this.podcastSelectedTopic && (!this.podcastSelectedChannel || this.podcastSelectedChannel === 'ALL') && (!this.podcastSelectedLang || this.podcastSelectedLang === 'ALL')) {
                        try {
                            localStorage.setItem('stock_quant_podcast_episodes_cache', JSON.stringify(res.data.slice(0, 100)));
                        } catch (e) {}
                    }
                }
            } catch (e) {
                console.warn("Failed to load podcast episodes:", e);
            } finally {
                this.podcastLoading = false;
            }
        },
        async loadPodcastChannels() {
            try {
                const res = await ApiClient.getPodcastChannels(150);
                if (res && res.data) {
                    this.podcastChannels = res.data;
                    try {
                        localStorage.setItem('stock_quant_podcast_channels_cache', JSON.stringify(res.data));
                    } catch (e) {}
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
                    try {
                        localStorage.setItem('stock_quant_podcast_topics_cache', JSON.stringify(res.data));
                    } catch (e) {}
                }
            } catch (e) {
                console.warn("Failed to load podcast topics:", e);
            }
        },
        togglePodcastSubView(view) {
            this.podcastSubView = view;
        },
        setPodcastImportance(level) {
            this.podcastSelectedImportance = level;
        },
        formatFollowers(count) {
            if (count === null || count === undefined) return '資料不足';
            if (!count) return '0';
            if (count >= 1000000) {
                return (count / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
            }
            if (count >= 1000) {
                return (count / 1000).toFixed(0) + 'K';
            }
            return count.toString();
        },
        formatLots(val) {
            if (val === null || val === undefined) return '—';
            const n = Number(val);
            if (isNaN(n)) return String(val);
            return (n > 0 ? '+' : '') + n.toLocaleString();
        },
        formatPrice(val, ticker = null) {
            if (val === null || val === undefined || val === '') return '—';
            const num = Number(val);
            if (isNaN(num)) return String(val);
            if (ticker && (String(ticker).endsWith('.T') || String(ticker).endsWith('.JP'))) {
                return '¥' + num.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 1 });
            }
            return '$' + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        },
        formatNumber(val, decimals = 2) {
            if (val === null || val === undefined || val === '') return '—';
            const num = Number(val);
            if (isNaN(num) || !isFinite(num)) return String(val);
            const maxDec = Math.max(0, Math.min(20, Math.floor(Number(decimals) || 0)));
            const minDec = Math.min(Number.isInteger(num) ? 0 : 2, maxDec);
            try {
                return num.toLocaleString('en-US', {
                    minimumFractionDigits: minDec,
                    maximumFractionDigits: maxDec,
                });
            } catch (e) {
                return num.toFixed(maxDec);
            }
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
        deduplicateCandidates(rawList) {
            if (!rawList || !rawList.length) return [];
            const map = new Map();
            for (const item of rawList) {
                const key = item.ticker;
                if (!map.has(key)) {
                    map.set(key, {
                        ...item,
                        all_themes: item.theme_name ? [item.theme_name] : [],
                        all_theme_ids: item.theme_id ? [item.theme_id] : [],
                        combined_paper_count: item.paper_count || 0,
                        combined_signal_count: item.signal_count || 0,
                    });
                } else {
                    const existing = map.get(key);
                    if (item.theme_name && !existing.all_themes.includes(item.theme_name)) {
                        existing.all_themes.push(item.theme_name);
                    }
                    if (item.theme_id && !existing.all_theme_ids.includes(item.theme_id)) {
                        existing.all_theme_ids.push(item.theme_id);
                    }
                    existing.combined_paper_count += (item.paper_count || 0);
                    existing.combined_signal_count += (item.signal_count || 0);
                    existing.paper_count = existing.combined_paper_count;
                    existing.signal_count = existing.combined_signal_count;
                    // 當標的跨主題時，若分數更高則覆蓋主要推薦欄位
                    if (Number(item.total_score || 0) > Number(existing.total_score || 0)) {
                        existing.total_score = item.total_score;
                        existing.recommendation = item.recommendation;
                        existing.theme_id = item.theme_id;
                        existing.theme_name = item.theme_name;
                        existing.catalyst_summary = item.catalyst_summary;
                        existing.source_score = item.source_score;
                        existing.purity_score = item.purity_score;
                        existing.paper_score = item.paper_score;
                        existing.novelty_score = item.novelty_score;
                        existing.sentiment_score = item.sentiment_score;
                        existing.setup_score = item.setup_score;
                        existing.risk_score = item.risk_score;
                        existing.us_anchor = item.us_anchor;
                    }
                }
            }
            return Array.from(map.values()).sort((a, b) => (Number(b.total_score) || 0) - (Number(a.total_score) || 0));
        },
        toggleCandidateDossier(id) {
            this.discoveryExpandedDossiers[id] = !this.discoveryExpandedDossiers[id];
        },
        isDossierExpanded(id) {
            return !!this.discoveryExpandedDossiers[id];
        },
        async loadDiscoveryData(force = false) {
            if (!this.discoveryCandidates || !this.discoveryCandidates.length || force) {
                this.discoveryLoading = true;
            }
            if (force) {
                try {
                    this.showToast('🚀 正在向 arXiv 即時同步最新學術論文與前沿題材...');
                    await ApiClient.triggerDiscoverySync();
                    // 等待 2.5 秒讓後台拉取與 DuckDB 快照完成
                    await new Promise(r => setTimeout(r, 2500));
                } catch (e) {
                    console.warn("觸發前沿同步失敗:", e);
                }
            }
            try {
                const [candRes, paperRes, sigRes, statRes, histRes] = await Promise.all([
                    ApiClient.getDiscoveryCandidates({ limit: 100 }).catch(() => null),
                    ApiClient.getDiscoveryPapers({ limit: 40 }).catch(() => null),
                    ApiClient.getDiscoverySignals({ limit: 60 }).catch(() => null),
                    ApiClient.getDiscoveryStatus().catch(() => null),
                    ApiClient.getDiscoveryHistory(30).catch(() => null)
                ]);

                if (candRes && candRes.data && Array.isArray(candRes.data) && candRes.data.length > 0) {
                    this.discoveryCandidates = candRes.data;
                    try {
                        localStorage.setItem('stock_quant_discovery_candidates_cache', JSON.stringify(candRes.data));
                    } catch (e) {}
                } else if (!this.discoveryCandidates || !this.discoveryCandidates.length) {
                    try {
                        const local = localStorage.getItem('stock_quant_discovery_candidates_cache');
                        if (local) this.discoveryCandidates = JSON.parse(local);
                    } catch (e) {}
                }

                if (paperRes && paperRes.data && Array.isArray(paperRes.data) && paperRes.data.length > 0) {
                    this.discoveryPapers = paperRes.data;
                    try {
                        localStorage.setItem('stock_quant_discovery_papers_cache', JSON.stringify(paperRes.data));
                    } catch (e) {}
                } else if (!this.discoveryPapers || !this.discoveryPapers.length) {
                    try {
                        const local = localStorage.getItem('stock_quant_discovery_papers_cache');
                        if (local) this.discoveryPapers = JSON.parse(local);
                    } catch (e) {}
                }

                if (sigRes && sigRes.data && Array.isArray(sigRes.data) && sigRes.data.length > 0) {
                    this.discoverySignals = sigRes.data;
                    try {
                        localStorage.setItem('stock_quant_discovery_signals_cache', JSON.stringify(sigRes.data));
                    } catch (e) {}
                } else if (!this.discoverySignals || !this.discoverySignals.length) {
                    try {
                        const local = localStorage.getItem('stock_quant_discovery_signals_cache');
                        if (local) this.discoverySignals = JSON.parse(local);
                    } catch (e) {}
                }

                if (statRes && statRes.data) {
                    this.discoveryStatus = statRes.data;
                }
                if (histRes && histRes.data) {
                    this.discoveryHistory = histRes.data;
                }
            } catch (err) {
                console.warn("載入 Discovery 資料異常:", err);
            } finally {
                this.discoveryLoading = false;
            }
        },
        setStrategy(stratId) {
            this.selectedStrategy = stratId;
            this.updateUrl();
            this.loadData();
        },
        setViewMode(mode) {
            this.viewMode = mode;
            try {
                localStorage.setItem('stock_quant_view_mode', mode);
            } catch (e) {}
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
                        params.set('ticker', this.formatTicker(this.searchTicker));
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
                } else if (this.activeTab === 'discovery') {
                    if (this.discoverySelectedTheme && this.discoverySelectedTheme !== 'ALL') {
                        params.set('theme', this.discoverySelectedTheme);
                    }
                } else if (this.activeTab === 'industry_chain') {
                    if (this.selectedTheme && this.selectedTheme !== 'ALL') {
                        params.set('theme', this.selectedTheme);
                    }
                    if (this.selectedRelationType && this.selectedRelationType !== 'ALL') {
                        params.set('rel', this.selectedRelationType);
                    }
                    if (this.selectedNodeTicker) {
                        params.set('ticker', this.selectedNodeTicker);
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
                const tickerParam = params.get('ticker');
                if (!tab && tickerParam) {
                    this.activeTab = 'history';
                } else {
                    this.activeTab = (tab && ['dashboard', 'institutional', 'history', 'industry_chain', 'agent_hub', 'podcast', 'discovery'].includes(tab)) ? tab : 'dashboard';
                }

                const strat = params.get('strategy');
                this.selectedStrategy = (strat && ['resonance', 'xuantie', 'timestral_bullish', 'timestral_bearish', 'bullish', 'bearish', 'timesfm_bullish', 'timesfm_bearish', 'moirai_bullish', 'moirai_bearish', 'latest'].includes(strat)) ? strat : 'resonance';

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
                this.discoverySelectedTheme = params.get('theme') || 'ALL';
                if (tab === 'industry_chain') {
                    if (params.get('theme')) this.selectedTheme = params.get('theme');
                    if (params.get('rel')) this.selectedRelationType = params.get('rel');
                    const indTicker = params.get('ticker') || params.get('focus') || params.get('node');
                    this.selectedNodeTicker = indTicker ? indTicker.trim().toUpperCase() : null;
                }
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
            } else if (this.activeTab === 'industry_chain') {
                if (this.industryGraphData && this.industryGraphData.upstream_nodes && this.industryGraphData.upstream_nodes.length) {
                    if (this.selectedNodeTicker) {
                        const allNodes = [
                            ...(this.industryGraphData.upstream_nodes || []),
                            ...(this.industryGraphData.midstream_nodes || []),
                            ...(this.industryGraphData.downstream_nodes || []),
                        ];
                        const targetTicker = this.selectedNodeTicker;
                        const matched = allNodes.find(n => 
                            n.ticker === targetTicker || 
                            n.ticker === targetTicker.split('.')[0] ||
                            (targetTicker.includes('.') && n.ticker.split('.')[0] === targetTicker.split('.')[0])
                        );
                        if (matched) {
                            this.selectedNodeTicker = matched.ticker;
                            this.selectNode(matched.ticker, false);
                        } else {
                            this.clearNodeSelection(false);
                        }
                    } else {
                        this.clearNodeSelection(false);
                    }
                } else {
                    this.loadIndustryChainData();
                }
            } else if (this.activeTab === 'podcast') {
                this.loadPodcastEpisodes();
                this.loadPodcastChannels();
                this.loadPodcastTopics();
            } else if (this.activeTab === 'discovery') {
                this.loadDiscoveryData();
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
            if (typeof window !== 'undefined' && window.matchMedia && !window.matchMedia('(hover: hover) and (pointer: fine)').matches) {
                return;
            }
            if (this.hoverTimeout) {
                clearTimeout(this.hoverTimeout);
                this.hoverTimeout = null;
            }

            const rect = event.currentTarget.getBoundingClientRect();
            const popoverWidth = Math.min(window.innerWidth - 24, 384);
            const popoverHeight = 280;

            let left = rect.left;
            if (left + popoverWidth > window.innerWidth - 12) {
                left = window.innerWidth - popoverWidth - 12;
            }
            if (left < 12) left = 12;

            let top = rect.bottom + 8;
            if (top + popoverHeight > window.innerHeight - 12) {
                top = rect.top - popoverHeight - 8;
            }
            if (top < 12) top = 12;

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
                                    if (prof && prof.name) {
                                        this.records.forEach(r => {
                                            if (r.ticker === ticker && !r.name) r.name = prof.name;
                                        });
                                        if (this.institutionalFlows) {
                                            this.institutionalFlows.forEach(item => {
                                                if (item.ticker === ticker && !item.name) item.name = prof.name;
                                            });
                                        }
                                    }
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
        formatTicker(ticker) {
            if (!ticker) return '';
            return String(ticker).replace(/\.(TW|TWO)$/i, '');
        },
        getMarketLabel(ticker) {
            if (!ticker) return '';
            const t = String(ticker).toUpperCase();
            if (t === '^TWII' || t === 'WTX') return '台股大盤指數';
            if (t === '^N225') return '日經指數';
            if (t.startsWith('^')) return '美股大盤指數';
            if (t.endsWith('.T') || t.endsWith('.JP')) return '日經225 / 日股';
            if (t.endsWith('.TW') || t.endsWith('.TWO') || /^\d{4,5}$/.test(t)) return '台股';
            if (t.endsWith('.HK')) return '港股';
            return '美股';
        },
        isTwStock(ticker) {
            if (!ticker) return false;
            const t = String(ticker).toUpperCase();
            return t.endsWith('.TW') || t.endsWith('.TWO') || /^\d{4,5}$/.test(t);
        },
        stripEmojis(str) {
            if (!str) return '';
            return String(str).replace(/[\p{Extended_Pictographic}\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{2B50}-\u{2B55}\u{FE00}-\u{FE0F}]/gu, '').trim();
        },
        cleanRegimeName(str) {
            if (!str) return this.currentLocale === 'en' ? 'Evaluating…' : '評估中…';
            let s = this.stripEmojis(str);
            if (this.currentLocale === 'en') {
                const match = s.match(/\(([A-Za-z\s_-]+)\)/);
                if (match) return match[1].trim();
                if (s.includes('全面多頭') || s.includes('台股全面多頭')) return 'Strong Bull';
                if (s.includes('多頭回調') || s.includes('台股多頭回調')) return 'Bull Pullback';
                if (s.includes('弱勢整理') || s.includes('加權弱勢整理')) return 'Weak Consolidation';
                if (s.includes('溫和空頭')) return 'Mild Bear';
                if (s.includes('輕微震盪')) return 'Mild Volatility';
                if (s.includes('極度恐慌') || s.includes('全球恐慌避險')) return 'Extreme Panic';
                if (s.includes('劇烈震盪')) return 'High Volatility';
                if (s.includes('空頭恐慌') || s.includes('台美雙破季線')) return 'Bearish Defense';
                if (s.includes('資料不足')) return 'Unavailable';
                return s.replace(/\s*\([A-Za-z\s_-]+\)/g, '').trim();
            }
            // Traditional Chinese: strip all redundant English in parentheses
            return s.replace(/\s*\([A-Za-z\s_-]+\)/g, '').trim();
        },
        formatStrategyName(str) {
            if (!str) return '-';
            let s = this.stripEmojis(str);
            if (s.includes('玄鐵重劍') || s.includes('玄鐵買點') || s.includes('玄鐵')) {
                return '季線支撐';
            }
            if (s.includes('四重共振') || s.includes('三重共振') || s.includes('多維共振') || s.includes('雙重共振')) {
                return '綜合優選';
            }
            if (s.includes('三大模型共振') || s.includes('三模型看漲') || s.includes('三模型共振')) {
                return '三模型看漲';
            }
            if (s.includes('雙大模型共振') || s.includes('雙ML共振') || s.includes('雙時序共振') || s.includes('雙模型共振') || s.includes('雙AI看漲') || s.includes('雙模型看漲')) {
                return '雙模型看漲';
            }
            if (s.toLowerCase().includes('moirai')) {
                return 'MOIRAI-2';
            }
            return s;
        },
        parseTags(tags) {
            if (!tags) return [];
            const list = Array.isArray(tags) ? tags : String(tags).split('|').map(t => t.trim()).filter(t => t.length > 0);
            return list.map(rawTag => {
                let cleanLabel = this.stripEmojis(rawTag)
                    .replace(/\((?:Google|Mistral|Salesforce|888|本地規則)\)/gi, '')
                    .replace(/Google\s+|Mistral\s+|Salesforce\s+/gi, '')
                    .trim();
                cleanLabel = cleanLabel.replace(/^[🤖\s]+/, '').trim();
                if (cleanLabel.startsWith('AI強買')) cleanLabel = cleanLabel.replace(/^AI強買/, '強勢買進');
                else if (cleanLabel.startsWith('AI看好')) cleanLabel = cleanLabel.replace(/^AI看好/, '偏多看好');
                else if (cleanLabel.startsWith('AI建議觀察')) cleanLabel = cleanLabel.replace(/^AI建議觀察/, '建議觀察');
                else if (cleanLabel.startsWith('AI建議避開')) cleanLabel = cleanLabel.replace(/^AI建議避開/, '建議避開');
                else if (cleanLabel.startsWith('AI做多')) cleanLabel = cleanLabel.replace(/^AI做多/, '偏多佈局');
                else if (cleanLabel.startsWith('AI中立')) cleanLabel = cleanLabel.replace(/^AI中立/, '中立觀望');
                cleanLabel = cleanLabel.replace(/^AI\s*/, '').trim();

                if (cleanLabel.includes('玄鐵重劍') || cleanLabel.includes('玄鐵買點') || cleanLabel.includes('玄鐵')) {
                    cleanLabel = cleanLabel.replace(/玄鐵重劍|玄鐵買點|玄鐵/g, '季線支撐');
                }
                if (/四重共振|三重共振|多維共振|雙重共振/.test(cleanLabel)) {
                    cleanLabel = cleanLabel.replace(/四重共振|三重共振|多維共振|雙重共振/g, '綜合優選');
                }
                if (/三大模型共振|三模型看漲|三模型共振/.test(cleanLabel)) {
                    cleanLabel = cleanLabel.replace(/三大模型共振|三模型看漲|三模型共振/g, '三模型看漲');
                }
                if (/雙大模型共振|雙ML共振|雙時序共振|雙模型共振|雙AI看漲/.test(cleanLabel)) {
                    cleanLabel = cleanLabel.replace(/雙大模型共振|雙ML共振|雙時序共振|雙模型共振|雙AI看漲/g, '雙模型看漲');
                }
                let icon = null;
                let badgeType = 'default';

                let tooltip = cleanLabel;
                if (cleanLabel.includes('綜合優選')) {
                    icon = 'military_tech';
                    badgeType = 'resonance_triple';
                    tooltip = '多模型同步看漲 · 具備支撐或籌碼優勢';
                } else if (cleanLabel.includes('三模型看漲')) {
                    icon = 'psychology';
                    badgeType = 'resonance_triple';
                    tooltip = '三大時序模型同步看漲';
                } else if (cleanLabel.includes('雙模型看漲') || cleanLabel.includes('雙AI看漲')) {
                    icon = 'psychology';
                    badgeType = 'resonance_ml';
                    tooltip = '模型同步看漲';
                } else if (cleanLabel.includes('土洋合買')) {
                    icon = 'handshake';
                    badgeType = 'institutional';
                    tooltip = '外資投信同步買超';
                } else if (cleanLabel.includes('連買')) {
                    icon = 'handshake';
                    badgeType = 'institutional';
                    tooltip = '投信連續買超';
                } else if (cleanLabel.includes('買超') || cleanLabel.includes('鎖碼')) {
                    icon = 'handshake';
                    badgeType = 'institutional';
                    tooltip = '法人買超';
                } else if (cleanLabel.includes('季線支撐') || cleanLabel.includes('均線') || cleanLabel.includes('MA60') || cleanLabel.includes('MA120') || cleanLabel.includes('MA20')) {
                    icon = 'show_chart';
                    badgeType = 'technical';
                    tooltip = '均線支撐買點';
                } else if (cleanLabel.includes('Timestral')) {
                    icon = 'smart_toy';
                    badgeType = 'ai_timestral';
                    tooltip = cleanLabel.includes('跌') || cleanLabel.includes('避險') || cleanLabel.includes('防守') ? 'Timestral 看跌' : 'Timestral 看漲';
                } else if (cleanLabel.includes('TimesFM')) {
                    icon = 'auto_awesome';
                    badgeType = 'ai_timesfm';
                    tooltip = cleanLabel.includes('跌') || cleanLabel.includes('避險') || cleanLabel.includes('防守') ? 'TimesFM 看跌' : 'TimesFM 看漲';
                } else if (cleanLabel.includes('MOIRAI')) {
                    icon = 'insights';
                    badgeType = 'ai_moirai';
                    tooltip = cleanLabel.includes('跌') || cleanLabel.includes('避險') || cleanLabel.includes('防守') ? 'MOIRAI-2 看跌' : 'MOIRAI-2 看漲';
                } else if (cleanLabel.includes('高盈虧比') || cleanLabel.includes('高風報比')) {
                    icon = 'balance';
                    badgeType = 'risk_reward';
                    tooltip = '高盈虧比';
                } else if (cleanLabel.includes('低PE') || cleanLabel.includes('低PB')) {
                    icon = 'equalizer';
                    badgeType = 'fundamental';
                    tooltip = '低估值';
                } else if (cleanLabel.includes('防守') || cleanLabel.includes('避險')) {
                    icon = 'shield';
                    badgeType = 'defense';
                    tooltip = '避險防守';
                } else if (cleanLabel.includes('主流板塊')) {
                    icon = 'hub';
                    badgeType = 'technical';
                    tooltip = '強勢主流板塊';
                }

                return {
                    raw: rawTag,
                    label: cleanLabel,
                    icon: icon,
                    type: badgeType,
                    tooltip: tooltip
                };
            });
        },
        getTagClass(tag) {
            if (!tag) return '';
            const t = tag.type || 'default';
            if (this.isDark) {
                switch (t) {
                    case 'resonance_quad':
                        return 'bg-amber-500/20 text-amber-300 border-amber-500/40 font-bold';
                    case 'resonance_triple':
                        return 'bg-amber-500/15 text-amber-400 border-amber-500/30 font-bold';
                    case 'resonance_double':
                    case 'resonance_multi':
                        return 'bg-amber-500/10 text-amber-400 border-amber-500/25 font-bold';
                    case 'resonance_ml':
                        return 'bg-purple-950/50 text-purple-300 border-purple-800/40 font-bold';
                    case 'institutional':
                        return 'bg-emerald-950/50 text-emerald-300 border-emerald-800/40 font-semibold';
                    case 'technical':
                        return 'bg-sky-950/40 text-sky-300 border-sky-800/40 font-medium';
                    case 'ai_timestral':
                        return 'bg-cyan-950/40 text-cyan-300 border-cyan-800/40 font-medium';
                    case 'ai_timesfm':
                        return 'bg-purple-950/40 text-purple-300 border-purple-800/40 font-medium';
                    case 'ai_moirai':
                        return 'bg-sky-950/40 text-sky-300 border-sky-800/40 font-medium';
                    case 'ai_lstm':
                        return 'bg-blue-950/30 text-blue-300 border-blue-800/30 font-medium';
                    case 'risk_reward':
                        return 'bg-teal-950/40 text-teal-300 border-teal-800/40 font-semibold';
                    case 'defense':
                        return 'bg-rose-950/40 text-rose-300 border-rose-800/40 font-medium';
                    default:
                        return 'bg-[#16181D] text-slate-300 border-[#262626] font-medium';
                }
            } else {
                switch (t) {
                    case 'resonance_quad':
                        return 'bg-amber-100 text-amber-900 border-amber-300 font-bold';
                    case 'resonance_triple':
                        return 'bg-amber-50 text-amber-800 border-amber-300 font-bold';
                    case 'resonance_double':
                    case 'resonance_multi':
                        return 'bg-amber-50 text-amber-800 border-amber-200 font-bold';
                    case 'resonance_ml':
                        return 'bg-purple-50 text-purple-800 border-purple-200 font-bold';
                    case 'institutional':
                        return 'bg-emerald-50 text-emerald-800 border-emerald-200 font-semibold';
                    case 'technical':
                        return 'bg-sky-50 text-sky-800 border-sky-200 font-medium';
                    case 'ai_timestral':
                        return 'bg-cyan-50 text-cyan-800 border-cyan-200 font-medium';
                    case 'ai_timesfm':
                        return 'bg-purple-50 text-purple-800 border-purple-200 font-medium';
                    case 'ai_moirai':
                        return 'bg-sky-50 text-sky-800 border-sky-200 font-medium';
                    case 'ai_lstm':
                        return 'bg-blue-50 text-blue-800 border-blue-200 font-medium';
                    case 'risk_reward':
                        return 'bg-teal-50 text-teal-800 border-teal-200 font-semibold';
                    case 'defense':
                        return 'bg-rose-50 text-rose-800 border-rose-200 font-medium';
                    default:
                        return 'bg-slate-100 text-slate-700 border-slate-200 font-medium';
                }
            }
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
                    description: '篩選雙時序共振強勢股（技術面均線 ∩ 雙大模型看漲 ∩ 法人籌碼鎖碼 ∩ 估值合理）。',
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
                    name: 'get_timestral_top_predictions',
                    description: '取得 Timestral 時序大模型 5 日預測漲跌幅排行與蒙地卡羅區間。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            direction: { type: 'string', enum: ['bullish', 'bearish'], description: 'bullish 看漲 / bearish 看跌' },
                            index_name: { type: 'string', description: '指數名稱' },
                            limit: { type: 'integer', description: '回傳筆數' }
                        },
                        required: ['direction']
                    },
                    execute: async ({ direction, index_name, limit }) => JSON.stringify(await ApiClient.getPredictions({ strategy: direction === 'bearish' ? 'timestral_bearish' : 'timestral_bullish', indexName: index_name, limit: limit || 20 }))
                });

                await modelCtx.registerTool({
                    name: 'get_timesfm_top_predictions',
                    description: '取得 TimesFM 時序大模型 5 日預測漲跌幅排行榜與盈虧比。',
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
                    name: 'get_moirai_top_predictions',
                    description: '取得 Salesforce MOIRAI-2 時序大模型隔日目標價與 P10/P90 蒙地卡羅風控區間。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            direction: { type: 'string', enum: ['bullish', 'bearish'], description: 'bullish 看漲 / bearish 看跌' },
                            index_name: { type: 'string', description: '指數名稱' },
                            limit: { type: 'integer', description: '回傳筆數' }
                        },
                        required: ['direction']
                    },
                    execute: async ({ direction, index_name, limit }) => JSON.stringify(await ApiClient.getPredictions({ strategy: direction === 'bearish' ? 'moirai_bearish' : 'moirai_bullish', indexName: index_name, limit: limit || 20 }))
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

                await modelCtx.registerTool({
                    name: 'get_company_network',
                    description: '查詢台股企業之經濟部統編、資本額、董事長、上游控股、下游轉投資企業與股權關係網絡圖。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            ticker: { type: 'string', description: '股票代號或公司名稱（例如 2330.TW, 2454, 台積電）' }
                        },
                        required: ['ticker']
                    },
                    execute: async ({ ticker }) => JSON.stringify(await ApiClient.getCompanyNetwork(ticker))
                });

                await modelCtx.registerTool({
                    name: 'open_company_network',
                    description: '在瀏覽器中開啟特定台股企業之視覺化股權拓撲關係圖譜或於看盤抽屜中檢視。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            company: { type: 'string', description: '公司正式全名、股票代號或別名（例如 2330, 台積電）' },
                            direction: { type: 'string', enum: ['both', 'upstream', 'downstream'], description: '關係展開方向，預設 both' }
                        },
                        required: ['company']
                    },
                    execute: async ({ company, direction }) => {
                        try {
                            const res = await ApiClient.getCompanyNetwork(company);
                            const net = res?.data || res;
                            const url = net?.web_url || `https://taiwan-company-network.david888.com/graph?company=${encodeURIComponent(company)}`;
                            window.open(url, '_blank');
                            return JSON.stringify({ navigated: true, company, url, direction: direction || 'both', message: `已在瀏覽器新分頁中開啟「${company}」之集團股權關係圖譜。` });
                        } catch (e) {
                            return JSON.stringify({ navigated: false, error: e.message });
                        }
                    }
                });

                await modelCtx.registerTool({
                    name: 'get_company_tenders',
                    description: '透過 ToBid 查詢台股企業之政府採購得標統計、得標率、累計金額、主要機關金主與競標對手情報。',
                    inputSchema: {
                        type: 'object',
                        properties: {
                            ticker: { type: 'string', description: '股票代號或公司名稱（例如 2634.TW, 1519, 漢翔）' }
                        },
                        required: ['ticker']
                    },
                    execute: async ({ ticker }) => JSON.stringify(await ApiClient.getCompanyTender(ticker))
                });

                console.log('[Chrome WebMCP] Registered native tools to document.modelContext');
            } catch (e) {
                console.warn('[Chrome WebMCP] Registration warning:', e);
            }
        },

        // =========================================================================
        // AI 產業價值鏈與供應鏈傳導圖譜 (Industry Value Chain Knowledge Graph)
        // =========================================================================

        async loadIndustryChainData() {
            await this.fetchIndustryThemes();
            await this.fetchIndustryGraph();
        },

        async fetchIndustryThemes() {
            try {
                const res = await ApiClient.getIndustryThemes();
                if (res && res.data) {
                    this.industryThemes = res.data;
                }
            } catch (e) {
                console.warn("fetchIndustryThemes error:", e);
            }
        },

        async fetchIndustryGraph() {
            if (!this.industryGraphData || !this.industryGraphData.upstream_nodes || !this.industryGraphData.upstream_nodes.length) {
                this.industryLoading = true;
            }
            if (this._industryGraphReqId === undefined) this._industryGraphReqId = 0;
            const reqId = ++this._industryGraphReqId;

            try {
                const res = await ApiClient.getIndustryGraph(this.selectedTheme, this.selectedRelationType);
                if (reqId !== this._industryGraphReqId) {
                    return; // 忽略被後續篩選請求覆蓋的過期回應
                }
                if (res && res.success) {
                    this.industryGraphData = res;
                    if (this.selectedNodeTicker) {
                        const allNodes = [
                            ...(res.upstream_nodes || []),
                            ...(res.midstream_nodes || []),
                            ...(res.downstream_nodes || []),
                        ];
                        const targetTicker = this.selectedNodeTicker;
                        const matched = allNodes.find(n => 
                            n.ticker === targetTicker || 
                            n.ticker === targetTicker.split('.')[0] ||
                            (targetTicker.includes('.') && n.ticker.split('.')[0] === targetTicker.split('.')[0])
                        );
                        if (matched) {
                            this.selectedNodeTicker = matched.ticker;
                            this.selectNode(matched.ticker, false);
                        } else {
                            this.clearNodeSelection(false);
                        }
                    } else if (this.highlightedNodes && this.highlightedNodes.length) {
                        this.clearNodeSelection(false);
                    }
                    this.$nextTick(() => {
                        this.renderIndustryBeziers();
                    });
                }
            } catch (e) {
                if (reqId !== this._industryGraphReqId) return;
                console.warn("fetchIndustryGraph error:", e);
                this.showToast("載入產業鏈圖譜失敗：" + e.message);
            } finally {
                if (reqId === this._industryGraphReqId) {
                    this.industryLoading = false;
                }
            }
        },

        selectTheme(themeId) {
            if (this.selectedTheme === themeId) return;
            this.selectedTheme = themeId;
            this.clearNodeSelection();
            this.fetchIndustryGraph();
            this.updateUrl();
        },

        selectRelationType(relType) {
            if (this.selectedRelationType === relType) return;
            this.selectedRelationType = relType;
            this.fetchIndustryGraph();
            this.updateUrl();
        },

        selectNode(ticker, shouldUpdateUrl = true) {
            if (this.selectedNodeTicker === ticker && this.highlightedNodes && this.highlightedNodes.length > 0) {
                // 若已經選中同一節點，再次點選切換精簡/展開模式
                this.toggleFocusConnectedOnly();
                return;
            }
            this.selectedNodeTicker = ticker;
            // 點選節點時，預設啟用聚焦精簡模式，只保留關聯鏈路，免去上下滑動困擾
            this.focusConnectedOnly = true;

            const connected = new Set([ticker]);
            const connEdges = [];
            const edges = this.industryGraphData.edges || [];
            for (const e of edges) {
                if (e.source === ticker) {
                    connected.add(e.target);
                    connEdges.push(e);
                }
                if (e.target === ticker) {
                    connected.add(e.source);
                    connEdges.push(e);
                }
            }
            this.highlightedNodes = Array.from(connected);
            this.highlightedEdges = connEdges;

            // 即時同步至網址列，方便一鍵直達與分享他人
            if (shouldUpdateUrl) {
                this.updateUrl();
            }

            // 平滑滾動置頂，讓聚焦卡片立即進入視野
            const wrapperEl = document.getElementById('industry-canvas-wrapper');
            if (wrapperEl && wrapperEl.scrollTop > 50) {
                wrapperEl.scrollTo({ top: 0, behavior: 'smooth' });
            }

            if (connEdges.length === 0) {
                const themeObj = (this.industryThemes || []).find(t => t.id === this.selectedTheme);
                const themeLabel = themeObj ? themeObj.name : this.selectedTheme;
                this.showToast(`提示：${ticker} 在「${themeLabel}」主題下暫無直連邊，可切換至「全部主題」查看完整網絡`);
            }
            this.$nextTick(() => {
                this.renderIndustryBeziers();
                setTimeout(() => this.renderIndustryBeziers(), 60);
                setTimeout(() => this.renderIndustryBeziers(), 200);
            });
        },

        clearNodeSelection(shouldUpdateUrl = true) {
            if (!this.selectedNodeTicker && (!this.highlightedNodes || !this.highlightedNodes.length)) return;
            this.selectedNodeTicker = null;
            this.highlightedNodes = [];
            this.highlightedEdges = [];
            if (shouldUpdateUrl) {
                this.updateUrl();
            }
            this.$nextTick(() => {
                this.renderIndustryBeziers();
                setTimeout(() => this.renderIndustryBeziers(), 60);
                setTimeout(() => this.renderIndustryBeziers(), 200);
            });
        },

        copyIndustryShareUrl() {
            this.updateUrl(true);
            const fullUrl = window.location.href;
            if (!this.selectedNodeTicker) {
                this.copyText(fullUrl, '已複製產業鏈圖譜連結！');
                return;
            }
            const comp = this.getSelectedNodeCompany();
            const label = comp ? `${this.selectedNodeTicker} (${comp})` : this.selectedNodeTicker;
            this.copyText(fullUrl, `✅ 已複製「${label}」產業鏈直達連結！可直接分享給他人`);
        },

        toggleFocusConnectedOnly() {
            this.focusConnectedOnly = !this.focusConnectedOnly;
            this.$nextTick(() => {
                this.renderIndustryBeziers();
                setTimeout(() => this.renderIndustryBeziers(), 60);
                setTimeout(() => this.renderIndustryBeziers(), 200);
            });
        },

        isNodeVisible(ticker) {
            // 1. 若處於「聚焦模式」
            if (this.selectedNodeTicker) {
                if (this.focusConnectedOnly) {
                    return ticker === this.selectedNodeTicker || this.highlightedNodes.includes(ticker);
                }
            }
            // 2. 若開啟「隱藏未連線節點」
            if (this.hideUnconnectedNodes) {
                if (this.selectedNodeTicker) {
                    return ticker === this.selectedNodeTicker || this.highlightedNodes.includes(ticker);
                }
                return this.hasNodeConnections(ticker);
            }
            return true;
        },

        visibleCount(nodes) {
            if (!nodes || !Array.isArray(nodes)) return 0;
            return nodes.filter(n => this.isNodeVisible(n.ticker)).length;
        },

        getSelectedNodeCompany() {
            if (!this.selectedNodeTicker) return '';
            const all = [
                ...(this.industryGraphData.upstream_nodes || []),
                ...(this.industryGraphData.midstream_nodes || []),
                ...(this.industryGraphData.downstream_nodes || [])
            ];
            const found = all.find(n => n.ticker === this.selectedNodeTicker);
            return found ? found.company : '';
        },

        onCanvasBackgroundClick(event) {
            const target = event.target;
            if (target && !target.closest('[data-ticker]')) {
                this.clearNodeSelection();
            }
        },

        hasNodeConnections(ticker) {
            const edges = this.industryGraphData.edges || [];
            return edges.some(e => e.source === ticker || e.target === ticker);
        },

        getNodeCardClasses(ticker) {
            const isDark = this.isDark;
            if (!this.selectedNodeTicker) {
                return isDark
                    ? 'bg-[#0D0E11] border-[#262626] hover:border-slate-500 hover:shadow-md'
                    : 'bg-[#FFF9E8] border-[#DDC58B] hover:border-[#6B5B38] hover:shadow-md';
            }

            if (ticker === this.selectedNodeTicker) {
                return (isDark ? 'bg-[#16181D] ' : 'bg-white ') +
                    'ring-3 ring-amber-500 border-amber-500 shadow-xl shadow-amber-500/25 scale-[1.02] z-30';
            }

            if (this.highlightedNodes.includes(ticker)) {
                return (isDark ? 'bg-[#16181D] ' : 'bg-white ') +
                    'ring-2 ring-sky-600 border-sky-600 shadow-md shadow-sky-600/20 opacity-100 z-20';
            }

            return (isDark ? 'bg-[#0D0E11] border-[#262626] ' : 'bg-[#FFF9E8]/90 border-[#E5D7B7] ') +
                'opacity-65 grayscale-[30%] hover:opacity-100 hover:grayscale-0 transition-opacity';
        },

        zoomIn() {
            if (this.graphZoom < 1.6) {
                this.graphZoom = Math.round((this.graphZoom + 0.1) * 10) / 10;
                this.$nextTick(() => this.renderIndustryBeziers());
                setTimeout(() => this.renderIndustryBeziers(), 160);
            }
        },

        zoomOut() {
            if (this.graphZoom > 0.6) {
                this.graphZoom = Math.round((this.graphZoom - 0.1) * 10) / 10;
                this.$nextTick(() => this.renderIndustryBeziers());
                setTimeout(() => this.renderIndustryBeziers(), 160);
            }
        },

        resetZoom() {
            this.graphZoom = 1.0;
            this.$nextTick(() => this.renderIndustryBeziers());
            setTimeout(() => this.renderIndustryBeziers(), 160);
        },

        renderIndustryBeziers() {
            if (this.activeTab !== 'industry_chain') return;
            const svgEl = document.getElementById('industry-chain-svg');
            const wrapperEl = document.getElementById('industry-canvas-wrapper');
            if (!svgEl || !wrapperEl) return;

            const wrapperRect = wrapperEl.getBoundingClientRect();
            const scrollLeft = wrapperEl.scrollLeft || 0;
            const scrollTop = wrapperEl.scrollTop || 0;

            const edges = this.industryGraphData.edges || [];
            const paths = [];

            for (const edge of edges) {
                const srcEl = document.getElementById('ind-card-' + edge.source);
                const tgtEl = document.getElementById('ind-card-' + edge.target);
                if (!srcEl || !tgtEl) continue;

                // 檢查是否被隱藏
                if (srcEl.offsetParent === null || tgtEl.offsetParent === null) continue;

                const srcRect = srcEl.getBoundingClientRect();
                const tgtRect = tgtEl.getBoundingClientRect();

                // 相對於 canvas wrapper 之錨點坐標
                const x1 = (srcRect.right - wrapperRect.left) + scrollLeft;
                const y1 = (srcRect.top + srcRect.height / 2 - wrapperRect.top) + scrollTop;
                const x2 = (tgtRect.left - wrapperRect.left) + scrollLeft;
                const y2 = (tgtRect.top + tgtRect.height / 2 - wrapperRect.top) + scrollTop;

                // 若源卡片在目標卡片右側 (例如回流反向邊) 或同行
                const isForward = x2 > x1 + 40;
                const dx = isForward ? Math.max(40, (x2 - x1) * 0.5) : 60;
                const cx1 = isForward ? x1 + dx : x1 + dx;
                const cy1 = y1;
                const cx2 = isForward ? x2 - dx : x2 - dx;
                const cy2 = y2;
                const d = `M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${cx1.toFixed(1)} ${cy1.toFixed(1)}, ${cx2.toFixed(1)} ${cy2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`;

                // WCAG 2.2 AA / AAA 色弱友善高對比雙重編碼配色 (高對比色彩 + 拓撲線型 Pattern)
                const isDark = this.isDark;
                let color = isDark ? '#38BDF8' : '#0284C7'; // 產業鏈: Deep Sky Blue
                let dashArray = 'none'; // 實線 (Solid)
                let strokeWidth = 2.4;

                if (edge.relation_type === '資本開支') {
                    color = isDark ? '#34D399' : '#047857'; // 資本開支: Deep Forest Teal
                    dashArray = '8 5'; // 長虛線 (Dashed)
                    strokeWidth = 2.2;
                } else if (edge.relation_type === '大客戶依賴') {
                    color = isDark ? '#FBBF24' : '#B45309'; // 大客戶依賴: Deep Amber-Orange
                    dashArray = '8 3 2 3'; // 點劃線 (Dash-Dot)
                    strokeWidth = 2.6;
                } else if (edge.relation_type === '主題關聯') {
                    color = isDark ? '#C084FC' : '#7E22CE'; // 主題關聯: Deep Royal Purple
                    dashArray = '3 3'; // 密點線 (Dotted)
                    strokeWidth = 2.4;
                }

                let opacity = isDark ? 0.65 : 0.80;
                let width = strokeWidth;
                let isHighlighted = false;

                if (this.selectedNodeTicker) {
                    if (edge.source === this.selectedNodeTicker || edge.target === this.selectedNodeTicker) {
                        opacity = 1.0;
                        width = strokeWidth + 1.4; // 3.6px - 4.0px
                        isHighlighted = true;
                    } else {
                        opacity = isDark ? 0.10 : 0.16;
                        width = 1.2;
                    }
                }

                paths.push({
                    d,
                    color,
                    width,
                    opacity,
                    isHighlighted,
                    dashArray
                });
            }

            this.bezierPaths = paths;
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
            this.loadPerformance();
            if (this.searchTicker) {
                this.loadHistory();
            }
        } else if (this.activeTab === 'podcast') {
            this.loadPodcastEpisodes();
            this.loadPodcastChannels();
            this.loadPodcastTopics();
        } else if (this.activeTab === 'industry_chain') {
            this.loadIndustryChainData();
        } else if (this.activeTab === 'discovery') {
            this.loadDiscoveryData();
        } else {
            this.loadData();
        }

        window.addEventListener('popstate', () => {
            this.syncStateFromUrl();
        });

        window.addEventListener('resize', () => {
            if (this.activeTab === 'industry_chain') {
                this.renderIndustryBeziers();
            }
        });

        window.addEventListener('scroll', () => {
            if (this.hoverVisible) {
                this.hideCompanyHover(true);
            }
            if (this.activeTab === 'industry_chain') {
                this.renderIndustryBeziers();
            }
        }, { passive: true });

        this.registerChromeWebMCP();
    }
});

// Global helper registration to prevent any template rendering exceptions
const globalFormatTicker = (ticker) => {
    if (!ticker) return '';
    return String(ticker).replace(/\.(TW|TWO)$/i, '');
};
app.config.globalProperties.formatTicker = globalFormatTicker;
window.formatTicker = globalFormatTicker;

const globalGetMarketLabel = (ticker) => {
    if (!ticker) return '';
    const t = String(ticker).toUpperCase();
    if (t === '^TWII' || t === 'WTX') return '台股大盤指數';
    if (t === '^N225') return '日經指數';
    if (t.startsWith('^')) return '美股大盤指數';
    if (t.endsWith('.T') || t.endsWith('.JP')) return '日經225 / 日股';
    if (t.endsWith('.TW') || t.endsWith('.TWO') || /^\d{4,5}$/.test(t)) return '台股';
    if (t.endsWith('.HK')) return '港股';
    return '美股';
};
app.config.globalProperties.getMarketLabel = globalGetMarketLabel;
window.getMarketLabel = globalGetMarketLabel;

const globalStripEmojis = (str) => {
    if (!str) return '';
    return String(str).replace(/[\p{Extended_Pictographic}\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{2B50}-\u{2B55}\u{FE00}-\u{FE0F}]/gu, '').trim();
};
app.config.globalProperties.stripEmojis = globalStripEmojis;
window.stripEmojis = globalStripEmojis;

const globalCleanRegimeName = (str) => {
    const locale = localStorage.getItem('stock_quant_locale') || 'zh';
    if (!str) return locale === 'en' ? 'Evaluating…' : '評估中…';
    let s = globalStripEmojis(str);
    if (locale === 'en') {
        const match = s.match(/\(([A-Za-z\s_-]+)\)/);
        if (match) return match[1].trim();
        if (s.includes('全面多頭') || s.includes('台股全面多頭')) return 'Strong Bull';
        if (s.includes('多頭回調') || s.includes('台股多頭回調')) return 'Bull Pullback';
        if (s.includes('弱勢整理') || s.includes('加權弱勢整理')) return 'Weak Consolidation';
        if (s.includes('溫和空頭')) return 'Mild Bear';
        if (s.includes('輕微震盪')) return 'Mild Volatility';
        if (s.includes('極度恐慌') || s.includes('全球恐慌避險')) return 'Extreme Panic';
        if (s.includes('劇烈震盪')) return 'High Volatility';
        if (s.includes('空頭恐慌') || s.includes('台美雙破季線')) return 'Bearish Defense';
        if (s.includes('資料不足')) return 'Unavailable';
        return s.replace(/\s*\([A-Za-z\s_-]+\)/g, '').trim();
    }
    return s.replace(/\s*\([A-Za-z\s_-]+\)/g, '').trim();
};
app.config.globalProperties.cleanRegimeName = globalCleanRegimeName;
window.cleanRegimeName = globalCleanRegimeName;

const globalFormatStrategyName = (str) => {
    if (!str) return '-';
    let s = globalStripEmojis(str);
    if (s.includes('玄鐵重劍') || s.includes('玄鐵買點') || s.includes('玄鐵')) {
        return '季線支撐';
    }
    if (s.includes('四重共振') || s.includes('三重共振') || s.includes('多維共振') || s.includes('雙重共振')) {
        return '綜合優選';
    }
    if (s.includes('三大模型共振') || s.includes('三模型看漲') || s.includes('三模型共振')) {
        return '三模型看漲';
    }
    if (s.includes('雙大模型共振') || s.includes('雙ML共振') || s.includes('雙時序共振') || s.includes('雙模型共振') || s.includes('雙AI看漲') || s.includes('雙模型看漲')) {
        return '雙模型看漲';
    }
    if (s.toLowerCase().includes('moirai')) {
        return 'MOIRAI-2';
    }
    return s;
};
app.config.globalProperties.formatStrategyName = globalFormatStrategyName;
window.formatStrategyName = globalFormatStrategyName;

const globalFormatFollowers = (count) => {
    if (count === null || count === undefined) return '資料不足';
    if (!count) return '0';
    const num = Number(count);
    if (isNaN(num)) return String(count);
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1).replace(/\.0$/, '') + 'M';
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(0) + 'K';
    }
    return num.toString();
};
app.config.globalProperties.formatFollowers = globalFormatFollowers;
window.formatFollowers = globalFormatFollowers;

const globalFormatLots = (val) => {
    if (val === null || val === undefined) return '—';
    const n = Number(val);
    if (isNaN(n)) return String(val);
    return (n > 0 ? '+' : '') + n.toLocaleString();
};
app.config.globalProperties.formatLots = globalFormatLots;
window.formatLots = globalFormatLots;

const globalFormatPrice = (val, ticker = null) => {
    if (val === null || val === undefined || val === '') return '—';
    const num = Number(val);
    if (isNaN(num)) return String(val);
    if (ticker && (String(ticker).endsWith('.T') || String(ticker).endsWith('.JP'))) {
        return '¥' + num.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 1 });
    }
    return '$' + num.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};
app.config.globalProperties.formatPrice = globalFormatPrice;
window.formatPrice = globalFormatPrice;

const globalFormatNumber = (val, decimals = 2) => {
    if (val === null || val === undefined || val === '') return '—';
    const num = Number(val);
    if (isNaN(num) || !isFinite(num)) return String(val);
    const maxDec = Math.max(0, Math.min(20, Math.floor(Number(decimals) || 0)));
    const minDec = Math.min(Number.isInteger(num) ? 0 : 2, maxDec);
    try {
        return num.toLocaleString('en-US', {
            minimumFractionDigits: minDec,
            maximumFractionDigits: maxDec,
        });
    } catch (e) {
        return num.toFixed(maxDec);
    }
};
app.config.globalProperties.formatNumber = globalFormatNumber;
window.formatNumber = globalFormatNumber;

// Auto-mount on load
app.mount('#app');
