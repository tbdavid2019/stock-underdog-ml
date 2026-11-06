/**
 * risk-xray.js - Quantitative Portfolio Risk X-Ray & Tax-Aware Allocation Engine Client
 * Version: 2.6.0
 * Part of 888 StockQuant Wave 2 Quant Risk & Validation System
 */

export async function fetchPortfolioRiskXray({ tickers = [], weights = null, window = 252, asOfDate = null } = {}) {
    if (!Array.isArray(tickers) || tickers.length === 0) {
        throw new Error('請至少選取或輸入 1 檔標的');
    }

    const payload = {
        tickers: tickers.map(t => String(t).trim().toUpperCase()).filter(Boolean),
        window: Math.max(30, Math.min(504, Number(window) || 252)),
    };
    if (weights && typeof weights === 'object') {
        payload.weights = weights;
    }
    if (asOfDate) {
        payload.as_of_date = String(asOfDate).trim();
    }

    const response = await fetch('/api/v1/risk/xray', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        let errorDetail = `伺服器回應異常 (HTTP ${response.status})`;
        try {
            const errJson = await response.json();
            if (errJson && errJson.detail) {
                errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
            }
        } catch (_) {}
        throw new Error(errorDetail);
    }

    return await response.json();
}

export async function fetchOptimizedWeights({
    tickers = [],
    currentWeights = null,
    gammaBuy = 0.001425,
    gammaSell = 0.004425,
    riskAversion = 1.0,
    maxSingleWeight = 0.10,
    window = 252,
    asOfDate = null,
} = {}) {
    if (!Array.isArray(tickers) || tickers.length === 0) {
        throw new Error('請至少選取或輸入 1 檔標的進行權重優化');
    }

    const payload = {
        tickers: tickers.map(t => String(t).trim().toUpperCase()).filter(Boolean),
        gamma_buy: (Number.isFinite(Number(gammaBuy)) && Number(gammaBuy) >= 0) ? Number(gammaBuy) : 0.001425,
        gamma_sell: (Number.isFinite(Number(gammaSell)) && Number(gammaSell) >= 0) ? Number(gammaSell) : 0.004425,
        risk_aversion: Number(riskAversion) >= 0 ? Number(riskAversion) : 1.0,
        max_single_weight: Math.min(1.0, Math.max(0.01, Number(maxSingleWeight) || 0.10)),
        window: Math.max(30, Math.min(504, Number(window) || 252)),
    };
    if (currentWeights && typeof currentWeights === 'object') {
        payload.current_weights = currentWeights;
    }
    if (asOfDate) {
        payload.as_of_date = String(asOfDate).trim();
    }

    const response = await fetch('/api/v1/risk/optimize-weights', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        let errorDetail = `權重優化運算失敗 (HTTP ${response.status})`;
        try {
            const errJson = await response.json();
            if (errJson && errJson.detail) {
                errorDetail = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
            }
        } catch (_) {}
        throw new Error(errorDetail);
    }

    return await response.json();
}

export function formatRiskPercent(val, decimals = 2) {
    if (val === null || val === undefined || !Number.isFinite(Number(val))) {
        return '—';
    }
    return (Number(val) * 100).toFixed(decimals) + '%';
}

if (typeof window !== 'undefined') {
    window.RiskXRay = {
        fetchPortfolioRiskXray,
        fetchOptimizedWeights,
        formatRiskPercent,
    };
}
