/**
 * api-client.js - REST API Client for 888 Stock Quant Engine
 * Encapsulates all backend FastAPI endpoint calls with consistent error handling and AbortController timeout protection.
 */

export async function fetchWithTimeout(url, options = {}, timeoutMs = 12000) {
    const controller = new AbortController();
    let timedOut = false;
    const id = setTimeout(() => {
        timedOut = true;
        controller.abort();
    }, timeoutMs);
    try {
        if (options.signal) {
            if (options.signal.aborted) {
                clearTimeout(id);
                throw options.signal.reason || new DOMException('The user aborted a request.', 'AbortError');
            }
            options.signal.addEventListener('abort', () => controller.abort(options.signal.reason));
        }
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(id);
        if (!response.ok) {
            let detail = `HTTP ${response.status}`;
            try {
                const errData = await response.json();
                if (errData && errData.detail) {
                    detail = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
                } else if (errData && errData.message) {
                    detail = errData.message;
                }
            } catch (_) {}
            const error = new Error(detail);
            error.status = response.status;
            throw error;
        }
        return response;
    } catch (err) {
        clearTimeout(id);
        if (err.name === 'AbortError') {
            if (!timedOut && options.signal && options.signal.aborted) {
                throw err;
            }
            const timeoutError = new Error(`請求逾時 (${timeoutMs}ms)`);
            timeoutError.status = 408;
            throw timeoutError;
        }
        throw err;
    }
}

export const ApiClient = {
    /**
     * Fetch latest market macro risk regime & exposure cap
     */
    async getMacroLatest(forceRefresh = false) {
        const url = forceRefresh ? '/api/v1/macro/latest?force_refresh=true' : '/api/v1/macro/latest';
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return data.success ? (data.data || data) : data;
    },

    /**
     * Refresh Investing.com & Polymarket macro catalysts
     */
    async refreshMacro() {
        const [res, polyRes] = await Promise.all([
            fetch('/api/v1/macro/investing/summary?force_refresh=true'),
            fetch('/api/v1/macro/polymarket/sentiment?force_refresh=true')
        ]);

        let polyData = null;
        if (polyRes.ok) {
            try {
                const pJson = await polyRes.json();
                if (pJson.success && pJson.data) polyData = pJson.data;
            } catch (e) {
                console.warn('Failed to parse Polymarket refresh payload:', e);
            }
        }

        let summaryData = null;
        if (res.ok) {
            const data = await res.json();
            if (data.success && data.data) summaryData = data.data;
        }

        return { summary: summaryData, polymarket: polyData };
    },

    /**
     * DuckDB database records stats summary
     */
    async getStatsSummary() {
        const res = await fetch('/api/v1/stats/summary');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * TWSE/TPEX market statistics
     */
    async getMarketStats() {
        const res = await fetch('/api/v1/market/stats');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return data.success ? (data.data || data) : data;
    },

    /**
     * Available historical trading dates for institutional flow
     */
    async getInstitutionalDates(limit = 120) {
        const res = await fetch(`/api/v1/market/institutional/dates?limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * TOP institutional net buyers / sellers
     */
    async getInstitutionalFlows({ orderBy = 'total_net', sortDir = 'desc', market = 'ALL', date = '', limit = 30 } = {}) {
        let url = `/api/v1/market/institutional/top?order_by=${orderBy}&sort_dir=${sortDir}&market=${market}&limit=${limit}`;
        if (date) {
            url += `&date=${encodeURIComponent(date)}`;
        }
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.success) return data.data || [];
        if (Array.isArray(data)) return data;
        return [];
    },

    /**
     * Fetch active foundation models from backend registry
     */
    async getActiveModels() {
        const res = await fetch('/api/v1/models');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Multi-Factor Quant Screener API
     */
    async screenPredictions({ direction = 'bullish', models = '', peRange = '', minPotential = 0, riskReward = false, ma60 = false, institutional = false, indexName = '', limit = 100 } = {}) {
        const params = new URLSearchParams();
        if (direction) params.append('direction', direction);
        if (models) params.append('models', models);
        if (peRange && peRange !== 'all') params.append('pe_range', peRange);
        if (minPotential > 0) params.append('min_potential', minPotential);
        if (riskReward) params.append('risk_reward', 'true');
        if (ma60) params.append('ma60', 'true');
        if (institutional) params.append('institutional', 'true');
        if (indexName && indexName !== 'ALL' && indexName !== '全部市場') params.append('index_name', indexName);
        params.append('limit', limit);
        const res = await fetch('/api/v1/predictions/screen?' + params.toString());
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Quantitative predictions by strategy & market index
     */
    async getPredictions({ strategy = 'resonance', indexName = '', limit = 60 } = {}) {
        let url = '/api/v1/predictions/latest?limit=' + limit;
        if (strategy === 'resonance') {
            url = '/api/v1/predictions/resonance?limit=' + limit;
        } else if (strategy === 'tri_bullish') {
            url = '/api/v1/predictions/tri-bullish?limit=' + limit;
        } else if (strategy === 'dual_bullish') {
            url = '/api/v1/predictions/dual-bullish?limit=' + limit;
        } else if (strategy === 'dual_bearish') {
            url = '/api/v1/predictions/dual-bearish?limit=' + limit;
        } else if (strategy === 'xuantie') {
            url = '/api/v1/predictions/xuantie?limit=' + limit;
        } else if (strategy === 'timesfm_bullish') {
            url = '/api/v1/predictions/timesfm/top-bullish?limit=' + limit;
        } else if (strategy === 'moirai_bullish') {
            url = '/api/v1/predictions/moirai/top-bullish?limit=' + limit;
        } else if (strategy === 'timestral_bullish') {
            url = '/api/v1/predictions/timestral/top-bullish?limit=' + limit;
        } else if (strategy === 'timestral_bearish') {
            url = '/api/v1/predictions/timestral/top-bearish?limit=' + limit;
        } else if (strategy === 'timesfm_bearish') {
            url = '/api/v1/predictions/timesfm/top-bearish?limit=' + limit;
        } else if (strategy === 'moirai_bearish') {
            url = '/api/v1/predictions/moirai/top-bearish?limit=' + limit;
        } else if (strategy === 'bullish') {
            url = '/api/v1/predictions/timestral/top-bullish?limit=' + limit;
        } else if (strategy === 'bearish') {
            url = '/api/v1/predictions/timestral/top-bearish?limit=' + limit;
        }

        if (indexName && indexName !== 'ALL' && indexName !== '全部市場') {
            url += (url.includes('?') ? '&' : '?') + 'index_name=' + encodeURIComponent(indexName);
        }

        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (!data.success) throw new Error('API returned an unsuccessful response');
        return data;
    },

    /**
     * Resolve fuzzy stock name/alias to standard symbol
     */
    async resolveTicker(query) {
        const res = await fetch(`/api/v1/predictions/resolve/${encodeURIComponent(query.trim())}`);
        if (!res.ok) {
            const detail = await res.json().catch(() => ({}));
            const err = new Error(detail.detail || `查無法辨識的股票或公司名稱：${query}`);
            err.status = res.status;
            throw err;
        }
        return await res.json();
    },

    /**
     * Single stock historical prediction trajectory
     */
    async getPredictionHistory(ticker, limit = 40) {
        const res = await fetch(`/api/v1/predictions/history/${encodeURIComponent(ticker)}?limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        return data.success ? (data.data || []) : (Array.isArray(data) ? data : []);
    },

    /**
     * Single stock historical institutional flows
     */
    async getInstitutionalHistory(ticker, limit = 60) {
        const res = await fetch(`/api/v1/market/institutional/${encodeURIComponent(ticker)}?limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (Array.isArray(data)) return data;
        if (data.data && Array.isArray(data.data)) return data.data;
        return [];
    },

    /**
     * Trigger background synchronization of institutional flows
     */
    async syncInstitutionalFlows(forceFull = false) {
        const res = await fetch(`/api/v1/market/institutional/sync?force_full=${forceFull}`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Broker branch accumulation trend time series
     */
    async getBrokerTrend(ticker, days = 20, topN = 8) {
        const res = await fetch(`/api/v1/market/broker/trend/${encodeURIComponent(ticker)}?days=${days}&top_n=${topN}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Broker branch summary with top 15 buyers and sellers
     */
    async getBrokerSummary(ticker, days = 20) {
        const res = await fetch(`/api/v1/market/broker/summary/${encodeURIComponent(ticker)}?days=${days}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Company profile from 2MD
     */
    async getCompanyProfile(ticker, signal) {
        const res = await fetch(`/api/v1/market/company-profile?ticker=${encodeURIComponent(ticker)}`, { signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Taiwan Company Network (corporate structure & relationships)
     */
    async getCompanyNetwork(ticker, signal) {
        const res = await fetch(`/api/v1/market/company-network?ticker=${encodeURIComponent(ticker)}`, { signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        return json.success ? json.data : (json.data || json);
    },

    /**
     * Government Procurement & Tender Intelligence (ToBid MCP)
     */
    async getCompanyTender(ticker, signal) {
        const res = await fetch(`/api/v1/market/company-tender?ticker=${encodeURIComponent(ticker)}`, { signal });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        return json.success ? json.data : null;
    },

    /**
     * Batch prefetch company profiles
     */
    async getCompanyProfilesBatch(tickers) {
        const res = await fetch('/api/v1/market/company-profiles/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Polymarket macro prediction market sentiment
     */
    async getPolymarketSentiment(category = 'all', forceRefresh = false) {
        const params = new URLSearchParams();
        if (category && category !== 'all') params.set('category', category);
        if (forceRefresh) params.set('force_refresh', 'true');
        const query = params.toString();
        const url = '/api/v1/macro/polymarket/sentiment' + (query ? '?' + query : '');
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Upcoming earnings calls
     */
    async getUpcomingEarningsCalls({ market = 'ALL', days = 14, isWeightStock = null, timeFilter = 'ALL', limit = 30 } = {}) {
        const params = new URLSearchParams();
        if (market) params.set('market', market);
        if (days) params.set('days', days.toString());
        if (isWeightStock !== null && isWeightStock !== undefined) params.set('is_weight_stock', isWeightStock ? 'true' : 'false');
        if (timeFilter && timeFilter !== 'ALL') params.set('time_filter', timeFilter);
        if (limit) params.set('limit', limit.toString());
        const res = await fetch(`/api/v1/earnings-call/upcoming?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Ticker specific earnings calls
     */
    async getTickerEarningsCalls(ticker) {
        const res = await fetch(`/api/v1/earnings-call/${encodeURIComponent(ticker)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Monthly earnings calls calendar summary
     */
    async getEarningsCalendar(year, month, market = 'ALL') {
        const params = new URLSearchParams();
        if (year) params.set('year', year.toString());
        if (month) params.set('month', month.toString());
        if (market) params.set('market', market);
        const res = await fetch(`/api/v1/earnings-call/calendar?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Podcast Radar episodes with stock quote enrichment
     */
    async getPodcastEpisodes({ topic = null, ticker = null, channelId = null, lang = 'ALL', limit = 100, offset = 0 } = {}) {
        const params = new URLSearchParams();
        if (topic) params.set('topic', topic);
        if (ticker) params.set('ticker', ticker);
        if (channelId && channelId !== 'ALL') params.set('channel_id', channelId);
        if (lang && lang !== 'ALL') params.set('lang', lang);
        if (limit) params.set('limit', limit.toString());
        if (offset) params.set('offset', offset.toString());

        const res = await fetch(`/api/v1/podcast/episodes?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Podcast subscribed channels list
     */
    async getPodcastChannels(limit = 200) {
        const res = await fetch(`/api/v1/podcast/channels?limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Podcast aggregated hot topics/entities frequency list
     */
    async getPodcastTopics(limit = 30) {
        const res = await fetch(`/api/v1/podcast/topics?limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Trigger background synchronization of podcast feeds
     */
    async syncPodcast() {
        const res = await fetch('/api/v1/podcast/sync', { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get Clef System One stock decision verdict
     */
    async getStockDecision(ticker, forceRefresh = false) {
        const url = `/api/v1/decision/stock/${encodeURIComponent(ticker)}${forceRefresh ? '?force_refresh=true' : ''}`;
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    // ==================== Discovery Tech Radar Operations ====================

    /**
     * Get Discovery candidate stocks
     */
    async getDiscoveryCandidates(params = {}) {
        const query = new URLSearchParams();
        if (params.recommendation) query.set('recommendation', params.recommendation);
        if (params.themeId) query.set('theme_id', params.themeId);
        if (params.asOf) query.set('as_of', params.asOf);
        if (params.limit) query.set('limit', params.limit);
        if (params.offset) query.set('offset', params.offset);

        const qs = query.toString();
        const res = await fetch(`/api/v1/discovery/candidates${qs ? '?' + qs : ''}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get Discovery arXiv technical papers
     */
    async getDiscoveryPapers(params = {}) {
        const query = new URLSearchParams();
        if (params.themeId) query.set('theme_id', params.themeId);
        if (params.category) query.set('category', params.category);
        if (params.asOf) query.set('as_of', params.asOf);
        if (params.limit) query.set('limit', params.limit);
        if (params.offset) query.set('offset', params.offset);

        const qs = query.toString();
        const res = await fetch(`/api/v1/discovery/papers${qs ? '?' + qs : ''}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get Discovery live industry signals
     */
    async getDiscoverySignals(params = {}) {
        const query = new URLSearchParams();
        if (params.ticker) query.set('ticker', params.ticker);
        if (params.themeId) query.set('theme_id', params.themeId);
        if (params.limit) query.set('limit', params.limit);
        if (params.offset) query.set('offset', params.offset);

        const qs = query.toString();
        const res = await fetch(`/api/v1/discovery/signals${qs ? '?' + qs : ''}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get Discovery timeline history trends
     */
    async getDiscoveryHistory(limit = 30) {
        const res = await fetch(`/api/v1/discovery/history?limit=${limit}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get Discovery engine health & freshness status
     */
    async getDiscoveryStatus() {
        const res = await fetch('/api/v1/discovery/status');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Trigger background sync of Discovery engine with arXiv
     */
    async triggerDiscoverySync() {
        const res = await fetch('/api/v1/discovery/sync', { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get available AI industry value chain themes
     */
    async getIndustryThemes() {
        const res = await fetch('/api/v1/industry-chain/themes');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get AI industry value chain graph topology with quant indicators
     */
    async getIndustryGraph(theme = 'ALL', relationType = 'ALL') {
        const params = new URLSearchParams();
        if (theme) params.set('theme', theme);
        if (relationType) params.set('relation_type', relationType);
        const res = await fetch(`/api/v1/industry-chain/graph?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get real-time cross-market trading sessions and freshness status
     */
    async getMarketLiveSessions() {
        const res = await fetch('/api/v1/market/calendar/sessions');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    },

    /**
     * Get user watchlist from cloud DuckDB
     */
    async getWatchlist(profileId = 'default', apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = {};
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout(`/api/v1/watchlist?profile_id=${encodeURIComponent(profileId)}`, { headers });
        return await res.json();
    },

    /**
     * Add ticker to cloud DuckDB watchlist
     */
    async addWatchlist(ticker, note = '', sortOrder = null, apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = { 'Content-Type': 'application/json' };
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout('/api/v1/watchlist', {
            method: 'POST',
            headers,
            body: JSON.stringify({ ticker, note, sort_order: sortOrder })
        });
        return await res.json();
    },

    /**
     * Remove ticker from cloud DuckDB watchlist
     */
    async removeWatchlist(ticker, apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = {};
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout(`/api/v1/watchlist/${encodeURIComponent(ticker)}`, {
            method: 'DELETE',
            headers
        });
        return await res.json();
    },

    /**
     * Batch import tickers to cloud DuckDB watchlist
     */
    async batchImportWatchlist(tickers, apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = { 'Content-Type': 'application/json' };
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout('/api/v1/watchlist/batch', {
            method: 'POST',
            headers,
            body: JSON.stringify({ tickers })
        });
        return await res.json();
    },

    /**
     * Resolve ticker or query
     */
    async resolveTicker(query) {
        const res = await fetchWithTimeout(`/api/v1/predictions/resolve/${encodeURIComponent(query)}`);
        return await res.json();
    },

    /**
     * Autocomplete stock tickers and names (Taiwan & US markets)
     */
    async autocompleteTickers(query, limit = 12, signal = null) {
        if (!query || !query.trim()) {
            return { query: query || '', results: [] };
        }
        const params = new URLSearchParams({ q: query.trim(), limit: String(limit) });
        const res = await fetchWithTimeout(`/api/v1/predictions/autocomplete?${params.toString()}`, { signal });
        return await res.json();
    },

    /**
     * Get Signal Alignment score & factor breakdown
     */
    async getSignalAlignment(ticker) {
        const res = await fetchWithTimeout(`/api/v1/predictions/alignment/${encodeURIComponent(ticker)}`);
        return await res.json();
    },

    /**
     * Get user price alerts
     */
    async getAlerts(apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = {};
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout('/api/v1/alerts', { headers });
        return await res.json();
    },

    /**
     * Create user price alert
     */
    async createAlert(alertPayload, apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = { 'Content-Type': 'application/json' };
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout('/api/v1/alerts', {
            method: 'POST',
            headers,
            body: JSON.stringify(alertPayload)
        });
        return await res.json();
    },

    /**
     * Delete user price alert
     */
    async deleteAlert(alertId, apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = {};
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout(`/api/v1/alerts/${encodeURIComponent(alertId)}`, {
            method: 'DELETE',
            headers
        });
        return await res.json();
    },

    /**
     * Trigger evaluation and dispatch of pending quant price alerts
     */
    async checkAlerts(apiKey = null) {
        const key = apiKey || localStorage.getItem('sq_watchlist_api_key_v1');
        const headers = { 'Content-Type': 'application/json' };
        if (key) headers['X-API-Key'] = key;
        const res = await fetchWithTimeout('/api/v1/alerts/check', {
            method: 'POST',
            headers
        });
        return await res.json();
    }
};

// Global fallback for non-module usage
if (typeof window !== 'undefined') {
    window.ApiClient = ApiClient;
}


