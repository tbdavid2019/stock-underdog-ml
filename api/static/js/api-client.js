/**
 * api-client.js - REST API Client for 888 Stock Quant Engine
 * Encapsulates all backend FastAPI endpoint calls with consistent error handling.
 */

export const ApiClient = {
    /**
     * Fetch latest market macro risk regime & exposure cap
     */
    async getMacroLatest() {
        const res = await fetch('/api/v1/macro/latest');
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
     * Quantitative predictions by strategy & market index
     */
    async getPredictions({ strategy = 'resonance', indexName = '', limit = 60 } = {}) {
        let url = '/api/v1/predictions/latest?limit=' + limit;
        if (strategy === 'resonance') {
            url = '/api/v1/predictions/resonance?limit=' + limit;
        } else if (strategy === 'xuantie') {
            url = '/api/v1/predictions/xuantie?limit=' + limit;
        } else if (strategy === 'timesfm_bullish') {
            url = '/api/v1/predictions/timesfm/top-bullish?limit=' + limit;
        } else if (strategy === 'bullish') {
            url = '/api/v1/predictions/lstm/top-bullish?limit=' + limit;
        } else if (strategy === 'bearish') {
            url = '/api/v1/predictions/lstm/top-bearish?limit=' + limit;
        } else if (strategy === 'timesfm_bearish') {
            url = '/api/v1/predictions/timesfm/top-bearish?limit=' + limit;
        }

        if (indexName) {
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
    async getUpcomingEarningsCalls({ market = 'ALL', days = 14, isWeightStock = null, limit = 30 } = {}) {
        const params = new URLSearchParams();
        if (market) params.set('market', market);
        if (days) params.set('days', days.toString());
        if (isWeightStock !== null && isWeightStock !== undefined) params.set('is_weight_stock', isWeightStock ? 'true' : 'false');
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
    }
};

// Global fallback for non-module usage
if (typeof window !== 'undefined') {
    window.ApiClient = ApiClient;
}

