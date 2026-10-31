/**
 * chart-manager.js - Chart.js Lifecycle & Theme Management for 888 Stock Quant
 * Manages institutional flow and broker branch accumulation trend charts with dark/light mode reactivity.
 * Features an internal instances map to avoid canvas memory leaks and ensure proper teardown.
 */

export class ChartManager {
    constructor() {
        this.instances = {};
    }

    getThemeTokens(isDark) {
        return {
            gridColor: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)',
            textColor: isDark ? '#9CA3AF' : '#334155',
            tooltipBg: isDark ? '#0D0E11' : '#FFFFFF',
            tooltipTitle: isDark ? '#F9FAFB' : '#0F172A',
            tooltipBody: isDark ? '#E5E7EB' : '#1E293B',
            tooltipBorder: isDark ? '#374151' : '#CBD5E1'
        };
    }

    /**
     * Safely destroy a tracked Chart instance by key
     * @param {string} key
     */
    destroy(key) {
        if (this.instances[key]) {
            try {
                this.instances[key].destroy();
            } catch (e) {
                console.debug(`Error destroying chart ${key}:`, e);
            }
            delete this.instances[key];
        }
    }

    /**
     * Destroy all active Chart.js instances (e.g. on view teardown or tab switch)
     */
    destroyAll() {
        for (const key of Object.keys(this.instances)) {
            this.destroy(key);
        }
    }

    /**
     * Render Single Stock Institutional Flow Cumulative Trend Line Chart
     * @param {HTMLCanvasElement|string} canvas - Canvas element or ID
     * @param {Array} rawHistory - Array of daily institutional flow objects
     * @param {boolean} isDark - Dark mode active flag
     */
    renderInstChart(canvas, rawHistory, isDark) {
        const el = typeof canvas === 'string' ? document.getElementById(canvas) : canvas;
        if (!el || !rawHistory || !rawHistory.length || typeof Chart === 'undefined') return null;

        const chartKey = 'instChart';
        this.destroy(chartKey);

        // Chronological order (earliest to latest)
        const sorted = [...rawHistory].reverse();
        const labels = sorted.map(d => d.date);

        let cumForeign = 0, cumTrust = 0, cumDealer = 0, cumTotal = 0;
        const dataForeign = [], dataTrust = [], dataDealer = [], dataTotal = [];

        for (const row of sorted) {
            cumForeign += (Number(row.foreign_net) || 0);
            cumTrust += (Number(row.trust_net) || 0);
            cumDealer += (Number(row.dealer_net) || 0);
            cumTotal += (Number(row.total_net) || 0);

            dataForeign.push(Math.round(cumForeign));
            dataTrust.push(Math.round(cumTrust));
            dataDealer.push(Math.round(cumDealer));
            dataTotal.push(Math.round(cumTotal));
        }

        const theme = this.getThemeTokens(isDark);

        const chart = new Chart(el, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: '三大法人合計',
                        data: dataTotal,
                        borderColor: isDark ? '#34D399' : '#059669',
                        backgroundColor: isDark ? 'rgba(52, 211, 153, 0.08)' : 'rgba(5, 150, 105, 0.06)',
                        borderWidth: 2.8,
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 2,
                        pointHoverRadius: 5,
                        fill: true
                    },
                    {
                        label: '外資累計',
                        data: dataForeign,
                        borderColor: isDark ? '#60A5FA' : '#1D4ED8',
                        borderDash: [5, 4],
                        borderWidth: 2.2,
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: '投信累計',
                        data: dataTrust,
                        borderColor: isDark ? '#FB7185' : '#E11D48',
                        borderDash: [2, 3],
                        borderWidth: 2.2,
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: '自營商累計',
                        data: dataDealer,
                        borderColor: isDark ? '#F59E0B' : '#D97706',
                        borderWidth: 1.8,
                        borderDash: [6, 3, 2, 3],
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 1,
                        pointHoverRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: theme.tooltipBg,
                        titleColor: theme.tooltipTitle,
                        bodyColor: theme.tooltipBody,
                        borderColor: theme.tooltipBorder,
                        borderWidth: 1,
                        padding: 10,
                        bodyFont: { family: 'JetBrains Mono' },
                        titleFont: { family: 'JetBrains Mono', weight: 'bold' },
                        callbacks: {
                            label(ctx) {
                                const val = ctx.parsed.y;
                                const sign = val > 0 ? '+' : '';
                                return ` ${ctx.dataset.label}: ${sign}${val.toLocaleString()} 張`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: theme.gridColor },
                        ticks: {
                            color: theme.textColor,
                            font: { family: 'JetBrains Mono', size: 10 },
                            maxTicksLimit: (typeof window !== 'undefined' && window.innerWidth < 640) ? 5 : 8,
                            callback: function(val, index) {
                                const raw = this.getLabelForValue ? this.getLabelForValue(val) : String(val);
                                if (typeof window !== 'undefined' && window.innerWidth < 640 && typeof raw === 'string') {
                                    const match = raw.match(/\d{4}-(\d{2}-\d{2})/);
                                    if (match) {
                                        return match[1].replace('-', '/');
                                    }
                                }
                                return raw;
                            }
                        }
                    },
                    y: {
                        grid: { color: theme.gridColor },
                        ticks: {
                            color: theme.textColor,
                            font: { family: 'JetBrains Mono', size: 10 },
                            callback: (v) => v.toLocaleString() + ' 張'
                        }
                    }
                }
            }
        });

        this.instances[chartKey] = chart;
        return chart;
    }

    /**
     * Toggle or set dataset visibility for a specific series key
     * @param {string} chartKey
     * @param {string} seriesKey
     * @param {boolean} isVisible
     * @returns {boolean} Whether dataset was found and updated
     */
    setDatasetVisibility(chartKey, seriesKey, isVisible) {
        const chart = this.instances[chartKey];
        if (!chart || !chart.data || !chart.data.datasets) return false;
        if (seriesKey === 'timestralBands') {
            let found = false;
            for (const d of chart.data.datasets) {
                if (d.key === 'timestralBandsUpper' || d.key === 'timestralBandsLower') {
                    d.hidden = !isVisible;
                    found = true;
                }
            }
            if (found) {
                chart.update();
                return true;
            }
        }
        const ds = chart.data.datasets.find(d => d.key === seriesKey);
        if (ds) {
            ds.hidden = !isVisible;
            chart.update();
            return true;
        }
        return false;
    }

    /**
     * Render Stock Price vs AI Target Price (LSTM, TimesFM, Resonance, MA60) Trajectory Chart
     * @param {HTMLCanvasElement|string} canvas - Canvas element or ID
     * @param {Array} historyRecords - Array of raw prediction history records
     * @param {boolean} isDark - Dark mode active flag
     * @param {Object} visibility - Map of seriesKey -> boolean
     */
    renderPricePredictChart(canvas, historyRecords, isDark, visibility = {}) {
        const el = typeof canvas === 'string' ? document.getElementById(canvas) : canvas;
        if (!el || !historyRecords || !historyRecords.length || typeof Chart === 'undefined') return null;

        const chartKey = 'pricePredictChart';
        this.destroy(chartKey);

        // Sort chronologically (earliest to latest)
        const sorted = [...historyRecords].sort((a, b) => String(a.timestamp || '').localeCompare(String(b.timestamp || '')));

        // Group by Date (YYYY-MM-DD)
        const dateMap = new Map();
        for (const r of sorted) {
            const rawTs = String(r.timestamp || '').trim();
            if (rawTs.length < 10) continue;
            const dateKey = rawTs.slice(0, 10);
            if (!dateMap.has(dateKey)) {
                dateMap.set(dateKey, {
                    date: dateKey,
                    price: null,
                    ma60: null,
                    lstm: null,
                    timesfm: null,
                    timestral: null,
                    timestralP10: null,
                    timestralP90: null,
                    resonance: null
                });
            }
            const item = dateMap.get(dateKey);
            const curPrice = Number(r.current_price);
            if (!isNaN(curPrice) && curPrice > 0) item.price = curPrice;
            const ma60 = Number(r.ma60);
            if (!isNaN(ma60) && ma60 > 0) item.ma60 = ma60;

            if (r.timestral_predicted_price) {
                const tp = Number(r.timestral_predicted_price);
                if (!isNaN(tp) && tp > 0) item.timestral = tp;
            }
            if (r.timestral_p10) {
                const p10 = Number(r.timestral_p10);
                if (!isNaN(p10) && p10 > 0) item.timestralP10 = p10;
            }
            if (r.timestral_p90) {
                const p90 = Number(r.timestral_p90);
                if (!isNaN(p90) && p90 > 0) item.timestralP90 = p90;
            }
            if (r.timesfm_predicted_price) {
                const tfp = Number(r.timesfm_predicted_price);
                if (!isNaN(tfp) && tfp > 0) item.timesfm = tfp;
            }

            const predPrice = Number(r.predicted_price);
            if (!isNaN(predPrice) && predPrice > 0) {
                const strat = ((r.strategy_type || '') + ' ' + (r.model_name || '')).toLowerCase();
                // Check resonance first to prevent Resonance from being misclassified
                if (strat.includes('共振') || strat.includes('resonance')) {
                    item.resonance = predPrice;
                } else if (strat.includes('timestral')) {
                    item.timestral = predPrice;
                } else if (strat.includes('timesfm')) {
                    item.timesfm = predPrice;
                } else if (strat.includes('lstm')) {
                    item.lstm = predPrice;
                }
            }
        }

        const dates = Array.from(dateMap.keys());
        if (!dates.length) return null;
        const n = dates.length;

        // Future anchor point label for T+1 prediction target day
        const labels = [...dates, '隔日預估 (T+1)'];

        const theme = this.getThemeTokens(isDark);

        // Actual price and MA60: valid on 0 ... n-1, null at n (T+1 anchor)
        const dataPrice = [...dates.map(d => dateMap.get(d).price), null];
        const dataMa60 = [...dates.map(d => dateMap.get(d).ma60), null];

        // AI predictions shifted by +1 (Target realization alignment):
        // Index 0: null (no prediction from day -1)
        // Index i + 1 (1 ... n): prediction made on day i (dates[i]) targeting day i + 1
        const dataTimestral = [null];
        const dataTimestralP10 = [null];
        const dataTimestralP90 = [null];
        const dataTimesfm = [null];
        const dataResonance = [null];
        const dataLstm = [null];

        for (let i = 0; i < n; i++) {
            const item = dateMap.get(dates[i]);
            dataTimestral.push(item.timestral);
            dataTimestralP10.push(item.timestralP10);
            dataTimestralP90.push(item.timestralP90);
            dataTimesfm.push(item.timesfm);
            let resVal = item.resonance;
            if (!resVal && item.timesfm && item.timestral) {
                resVal = Number(((item.timesfm + item.timestral) / 2).toFixed(2));
            }
            dataResonance.push(resVal);
            dataLstm.push(item.lstm);
        }

        // WCAG 2.1 Level AA & AAA Accessible Color System for Color Vision Deficiency (CVD):
        // 1. Real Price: High contrast Deep Navy Blue (#1D4ED8 light) / Sky Cobalt (#60A5FA dark) - Solid line, circle markers
        // 2. Timestral: Deep Warm Amber/Orange (#D97706 light / #F59E0B dark) - Dashed line, square markers
        // 3. TimesFM: Deep Royal Purple (#7C3AED light / #C084FC dark) - Dotted line, diamond markers
        // 4. Resonance: Deep Teal (#0F766E light / #2DD4BF dark) - Dash-Dot line, triangle markers
        // 5. MA60: Slate (#475569 light / #94A3B8 dark) - Long dashed line, no points
        // 6. Miss/Opposite direction: Vivid Rose/Crimson (#E11D48 light / #FB7185 dark) - 'crossRot' (✕) markers
        const colorPrice = isDark ? '#60A5FA' : '#1D4ED8';
        const colorTimestral = isDark ? '#F59E0B' : '#D97706';
        const colorTimestralBands = isDark ? 'rgba(245, 158, 11, 0.5)' : 'rgba(217, 119, 6, 0.45)';
        const colorTimestralFill = isDark ? 'rgba(245, 158, 11, 0.16)' : 'rgba(217, 119, 6, 0.12)';
        const colorTimesfm = isDark ? '#C084FC' : '#7C3AED';
        const colorResonance = isDark ? '#2DD4BF' : '#0F766E';
        const colorMa60 = isDark ? '#94A3B8' : '#475569';
        const colorLstm = isDark ? '#FB923C' : '#EA580C';
        const colorOpposite = isDark ? '#FB7185' : '#E11D48';

        /**
         * Dynamic Point Marker Generator for Prediction Datasets:
         * - 方向命中 (Hit): 實心幾何形狀 (Square / Diamond / Triangle)
         * - 方向相反 / 偏離 (Miss/Opposite): ✕ 叉號警示標點 ('crossRot')，標記預測漲但實際跌或預測跌但實際漲
         * - 隔日預估 (T+1 Pending): ★ 星號標點 ('star')，醒目標示最新尚未開獎之預估點
         */
        const computeMarkerStyles = (predArray, defaultStyle, modelColor) => {
            const pointStyles = [];
            const pointRadii = [];
            const pointHoverRadii = [];
            const pointBgColors = [];
            const pointBorderColors = [];
            const pointBorderWidths = [];

            const isDense = labels.length > 40;
            const baseHitRadius = isDense ? 2.5 : 4.5;
            const baseMissRadius = isDense ? 4.5 : 6.5;
            const baseT1Radius = isDense ? 5.5 : 7.5;

            for (let idx = 0; idx <= n; idx++) {
                const val = predArray[idx];
                if (val === null || val === undefined || isNaN(val)) {
                    pointStyles.push(defaultStyle);
                    pointRadii.push(0);
                    pointHoverRadii.push(0);
                    pointBgColors.push('transparent');
                    pointBorderColors.push('transparent');
                    pointBorderWidths.push(0);
                    continue;
                }

                // T+1 Upcoming target anchor (idx === n)
                if (idx === n) {
                    pointStyles.push('star');
                    pointRadii.push(baseT1Radius);
                    pointHoverRadii.push(baseT1Radius + 2.5);
                    pointBgColors.push(modelColor);
                    pointBorderColors.push(isDark ? '#FFFFFF' : '#0F172A');
                    pointBorderWidths.push(2);
                    continue;
                }

                // Historical realized prediction point (idx > 0 && idx < n)
                if (idx > 0) {
                    const targetDate = dates[idx];
                    const srcDate = dates[idx - 1];
                    const basePrice = srcDate ? dateMap.get(srcDate)?.price : null;
                    const actualPrice = targetDate ? dateMap.get(targetDate)?.price : null;

                    if (basePrice && basePrice > 0 && actualPrice && actualPrice > 0) {
                        const predDiff = val - basePrice;
                        const actDiff = actualPrice - basePrice;
                        // Direction hit if both positive/flat or both negative/flat
                        const isHit = (predDiff >= 0 && actDiff >= 0) || (predDiff <= 0 && actDiff <= 0);

                        if (isHit) {
                            // 方向命中 (Hit): 實心幾何形狀
                            pointStyles.push(defaultStyle);
                            pointRadii.push(baseHitRadius);
                            pointHoverRadii.push(baseHitRadius + 2.5);
                            pointBgColors.push(modelColor);
                            pointBorderColors.push(isDark ? '#0D0E11' : '#FFFFFF');
                            pointBorderWidths.push(1.5);
                        } else {
                            // 方向相反 / 偏離 (Miss / Opposite direction): ✕ 叉號標點
                            pointStyles.push('crossRot');
                            pointRadii.push(baseMissRadius);
                            pointHoverRadii.push(baseMissRadius + 3);
                            pointBgColors.push(colorOpposite);
                            pointBorderColors.push(colorOpposite);
                            pointBorderWidths.push(2.5);
                        }
                        continue;
                    }
                }

                // Fallback for missing baseline/actual price
                pointStyles.push(defaultStyle);
                pointRadii.push(baseHitRadius);
                pointHoverRadii.push(baseHitRadius + 2);
                pointBgColors.push(modelColor);
                pointBorderColors.push(isDark ? '#0D0E11' : '#FFFFFF');
                pointBorderWidths.push(1.5);
            }

            return {
                pointStyle: pointStyles,
                pointRadius: pointRadii,
                pointHoverRadius: pointHoverRadii,
                pointBackgroundColor: pointBgColors,
                pointBorderColor: pointBorderColors,
                pointBorderWidth: pointBorderWidths
            };
        };

        const timestralMarkers = computeMarkerStyles(dataTimestral, 'rect', colorTimestral);
        const timesfmMarkers = computeMarkerStyles(dataTimesfm, 'rectRot', colorTimesfm);
        const resonanceMarkers = computeMarkerStyles(dataResonance, 'triangle', colorResonance);
        const lstmMarkers = computeMarkerStyles(dataLstm, 'circle', colorLstm);

        const chart = new Chart(el, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        key: 'price',
                        label: '真實股價',
                        data: dataPrice,
                        borderColor: colorPrice,
                        backgroundColor: isDark ? 'rgba(96, 165, 250, 0.08)' : 'rgba(29, 78, 216, 0.05)',
                        borderWidth: 2.8,
                        tension: 0.25,
                        pointRadius: labels.length > 40 ? 1.5 : 3.5,
                        pointHoverRadius: 6,
                        pointBackgroundColor: colorPrice,
                        pointBorderColor: isDark ? '#0D0E11' : '#FFFFFF',
                        pointBorderWidth: 1.5,
                        pointStyle: 'circle',
                        fill: true,
                        spanGaps: true,
                        hidden: visibility.price === false
                    },
                    {
                        key: 'timestral',
                        label: 'Timestral 目標價 (P50)',
                        data: dataTimestral,
                        borderColor: colorTimestral,
                        backgroundColor: 'transparent',
                        borderDash: [5, 4],
                        borderWidth: 2.4,
                        tension: 0.25,
                        ...timestralMarkers,
                        spanGaps: true,
                        hidden: visibility.timestral === false
                    },
                    {
                        key: 'timestralBandsUpper',
                        label: 'Timestral P90 (樂觀上限)',
                        data: dataTimestralP90,
                        borderColor: colorTimestralBands,
                        backgroundColor: colorTimestralFill,
                        borderWidth: 1.2,
                        borderDash: [2, 2],
                        tension: 0.25,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        fill: '+1',
                        spanGaps: true,
                        hidden: visibility.timestralBands === false
                    },
                    {
                        key: 'timestralBandsLower',
                        label: 'Timestral P10 (悲觀下限)',
                        data: dataTimestralP10,
                        borderColor: colorTimestralBands,
                        backgroundColor: 'transparent',
                        borderWidth: 1.2,
                        borderDash: [2, 2],
                        tension: 0.25,
                        pointRadius: 0,
                        pointHoverRadius: 0,
                        fill: false,
                        spanGaps: true,
                        hidden: visibility.timestralBands === false
                    },
                    {
                        key: 'timesfm',
                        label: 'TimesFM 目標價',
                        data: dataTimesfm,
                        borderColor: colorTimesfm,
                        backgroundColor: 'transparent',
                        borderDash: [2, 3],
                        borderWidth: 2.4,
                        tension: 0.25,
                        ...timesfmMarkers,
                        spanGaps: true,
                        hidden: visibility.timesfm === false
                    },
                    {
                        key: 'resonance',
                        label: '雙模型平均目標價',
                        data: dataResonance,
                        borderColor: colorResonance,
                        backgroundColor: 'transparent',
                        borderDash: [6, 3, 2, 3],
                        borderWidth: 2.4,
                        tension: 0.25,
                        ...resonanceMarkers,
                        spanGaps: true,
                        hidden: visibility.resonance !== true
                    },
                    {
                        key: 'ma60',
                        label: 'MA60 季線',
                        data: dataMa60,
                        borderColor: colorMa60,
                        backgroundColor: 'transparent',
                        borderDash: [8, 4],
                        borderWidth: 1.8,
                        tension: 0.2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointBackgroundColor: colorMa60,
                        spanGaps: true,
                        hidden: visibility.ma60 !== true
                    },
                    {
                        key: 'lstm',
                        label: 'LSTM 目標價 (舊版)',
                        data: dataLstm,
                        borderColor: colorLstm,
                        backgroundColor: 'transparent',
                        borderDash: [4, 4],
                        borderWidth: 1.8,
                        tension: 0.25,
                        ...lstmMarkers,
                        spanGaps: true,
                        hidden: visibility.lstm !== true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: theme.tooltipBg,
                        titleColor: theme.tooltipTitle,
                        bodyColor: theme.tooltipBody,
                        borderColor: theme.tooltipBorder,
                        borderWidth: 1,
                        padding: 10,
                        bodyFont: { family: 'JetBrains Mono' },
                        titleFont: { family: 'JetBrains Mono', weight: 'bold' },
                        callbacks: {
                            title(items) {
                                if (!items || !items.length) return '';
                                const label = items[0].label;
                                if (label === '隔日預估 (T+1)') {
                                    return '預測目標: 隔日預估 (T+1)';
                                }
                                return `交易日期: ${label}`;
                            },
                            label(ctx) {
                                const val = ctx.parsed.y;
                                if (val === null || val === undefined || isNaN(val)) return null;
                                const formatted = '$' + Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 2 });
                                const label = ctx.dataset.label;
                                const idx = ctx.dataIndex;

                                if (ctx.dataset.key === 'price') {
                                    let diffStr = '';
                                    if (idx > 0 && idx < n) {
                                        const prevPrice = dateMap.get(dates[idx - 1])?.price;
                                        if (prevPrice && prevPrice > 0) {
                                            const ret = ((val - prevPrice) / prevPrice) * 100;
                                            const sign = ret >= 0 ? '+' : '';
                                            const arrow = ret >= 0 ? '↑' : '↓';
                                            diffStr = ` (${sign}${ret.toFixed(2)}% ${arrow})`;
                                        }
                                    }
                                    return ` ${label}: ${formatted} (當日收盤${diffStr})`;
                                }

                                if (ctx.dataset.key === 'ma60') {
                                    return ` ${label}: ${formatted}`;
                                }

                                if (ctx.dataset.key === 'timestralBandsUpper' || ctx.dataset.key === 'timestralBandsLower') {
                                    if (idx > 0 && idx <= n) {
                                        const srcDate = dates[idx - 1];
                                        const shortDate = srcDate && srcDate.length >= 10 ? srcDate.slice(5).replace('-', '/') : srcDate;
                                        return ` ${label} (${shortDate}預測): ${formatted}`;
                                    }
                                    return ` ${label}: ${formatted}`;
                                }

                                // Prediction datasets: TimesFM, Timestral, Resonance, LSTM
                                if (idx > 0 && idx <= n) {
                                    const srcDate = dates[idx - 1];
                                    const shortDate = srcDate && srcDate.length >= 10 ? srcDate.slice(5).replace('-', '/') : srcDate;
                                    const srcItem = srcDate ? dateMap.get(srcDate) : null;
                                    const basePrice = srcItem ? srcItem.price : null;
                                    const actualPrice = (idx < n) ? dateMap.get(dates[idx])?.price : null;

                                    if (idx === n) {
                                        if (basePrice && basePrice > 0) {
                                            const expDiff = ((val - basePrice) / basePrice) * 100;
                                            const expSign = expDiff >= 0 ? '+' : '';
                                            const expArrow = expDiff >= 0 ? '↑' : '↓';
                                            return [
                                                ` ${label} (${shortDate}預測): ${formatted}`,
                                                `   ↳ 預期${expDiff >= 0 ? '漲' : '跌'} ${expSign}${expDiff.toFixed(2)}% ${expArrow} (基準: ${shortDate}收盤 $${basePrice.toFixed(2)})`,
                                                `   ↳ 狀態: ⏳ 待下個交易日收盤開獎驗證`
                                            ];
                                        }
                                        return [
                                            ` ${label} (${shortDate}預測): ${formatted}`,
                                            `   ↳ 狀態: ⏳ 待下個交易日收盤開獎驗證`
                                        ];
                                    }

                                    let predStr = '';
                                    let predDiff = 0;
                                    if (basePrice && basePrice > 0) {
                                        predDiff = ((val - basePrice) / basePrice) * 100;
                                        const expSign = predDiff >= 0 ? '+' : '';
                                        const expArrow = predDiff >= 0 ? '↑' : '↓';
                                        predStr = `預期 ${expSign}${predDiff.toFixed(2)}% ${expArrow}`;
                                    }

                                    let actStr = '';
                                    let actDiff = 0;
                                    if (basePrice && basePrice > 0 && actualPrice && actualPrice > 0) {
                                        actDiff = ((actualPrice - basePrice) / basePrice) * 100;
                                        const actSign = actDiff >= 0 ? '+' : '';
                                        const actArrow = actDiff >= 0 ? '↑' : '↓';
                                        actStr = `實際 ${actSign}${actDiff.toFixed(2)}% ${actArrow}`;
                                    }

                                    let errStr = '';
                                    if (actualPrice && actualPrice > 0) {
                                        const errDiff = Math.abs(val - actualPrice) / actualPrice * 100;
                                        errStr = `誤差 ${errDiff.toFixed(2)}%`;
                                    }

                                    if (basePrice && basePrice > 0 && actualPrice && actualPrice > 0) {
                                        const isHit = (predDiff >= 0 && actDiff >= 0) || (predDiff <= 0 && actDiff <= 0);
                                        let outcomeLine = '';
                                        if (isHit) {
                                            outcomeLine = `   ↳ 🎯 方向命中 (${predStr} · ${actStr} ｜ ${errStr})`;
                                        } else {
                                            const predDir = predDiff >= 0 ? '預測漲' : '預測跌';
                                            const actDir = actDiff >= 0 ? '實際跌' : '實際漲';
                                            outcomeLine = `   ↳ ❌ 方向相反 (${predDir}但${actDir} · ${predStr} vs ${actStr} ｜ ${errStr})`;
                                        }
                                        return [
                                            ` ${label} (${shortDate}預測): ${formatted}`,
                                            outcomeLine
                                        ];
                                    } else if (errStr) {
                                        return [
                                            ` ${label} (${shortDate}預測): ${formatted}`,
                                            `   ↳ 誤差: ${errStr}`
                                        ];
                                    }
                                    return ` ${label} (${shortDate}預測): ${formatted}`;
                                }

                                return ` ${label}: ${formatted}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: theme.gridColor },
                        ticks: {
                            color: theme.textColor,
                            font: { family: 'JetBrains Mono', size: 10 },
                            maxTicksLimit: (typeof window !== 'undefined' && window.innerWidth < 640) ? 5 : 10,
                            callback: function(val, index) {
                                const raw = this.getLabelForValue ? this.getLabelForValue(val) : String(val);
                                if (typeof window !== 'undefined' && window.innerWidth < 640 && typeof raw === 'string') {
                                    if (raw.includes('隔日預估') || raw.includes('T+1')) {
                                        return 'T+1';
                                    }
                                    const match = raw.match(/\d{4}-(\d{2}-\d{2})/);
                                    if (match) {
                                        return match[1].replace('-', '/');
                                    }
                                }
                                return raw;
                            }
                        }
                    },
                    y: {
                        grid: { color: theme.gridColor },
                        ticks: {
                            color: theme.textColor,
                            font: { family: 'JetBrains Mono', size: 10 },
                            callback: (v) => '$' + v.toLocaleString()
                        }
                    }
                }
            }
        });

        this.instances[chartKey] = chart;
        return chart;
    }

    /**
     * Render Broker Branches Multi-Line Accumulation Trend Chart
     * @param {HTMLCanvasElement|string} canvas - Canvas element or ID
     * @param {Array} dates - Array of date strings
     * @param {Array} series - Array of series objects: { broker_name, data: [] }
     * @param {boolean} isDark - Dark mode active flag
     */
    renderBrokerChart(canvas, dates, series, isDark) {
        const el = typeof canvas === 'string' ? document.getElementById(canvas) : canvas;
        if (!el || !dates || !dates.length || !series || !series.length || typeof Chart === 'undefined') return null;

        const chartKey = 'brokerChart';
        this.destroy(chartKey);

        const theme = this.getThemeTokens(isDark);

        const neonPalette = [
            '#38BDF8', // Sky Blue
            '#F43F5E', // Rose Red
            '#10B981', // Emerald
            '#F59E0B', // Amber Gold
            '#A855F7', // Violet
            '#EC4899', // Pink
            '#06B6D4', // Cyan
            '#84CC16', // Lime
            '#6366F1', // Indigo
            '#F97316'  // Orange
        ];

        const datasets = series.map((item, idx) => {
            const color = neonPalette[idx % neonPalette.length];
            return {
                label: item.broker_name,
                data: item.data,
                borderColor: color,
                backgroundColor: 'transparent',
                borderWidth: idx < 3 ? 2.5 : 1.8,
                tension: 0.3,
                pointRadius: dates.length > 40 ? 0 : 2,
                pointHoverRadius: 5
            };
        });

        const chart = new Chart(el, {
            type: 'line',
            data: {
                labels: dates,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: {
                            color: theme.textColor,
                            boxWidth: 10,
                            font: { family: 'Noto Sans TC', size: 11 }
                        }
                    },
                    tooltip: {
                        backgroundColor: theme.tooltipBg,
                        titleColor: theme.tooltipTitle,
                        bodyColor: theme.tooltipBody,
                        borderColor: theme.tooltipBorder,
                        borderWidth: 1,
                        padding: 10,
                        bodyFont: { family: 'JetBrains Mono' },
                        titleFont: { family: 'JetBrains Mono', weight: 'bold' },
                        callbacks: {
                            label(ctx) {
                                const val = ctx.parsed.y;
                                const sign = val > 0 ? '+' : '';
                                return ` ${ctx.dataset.label}: ${sign}${val.toLocaleString()} 張`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: theme.gridColor },
                        ticks: {
                            color: theme.textColor,
                            font: { family: 'JetBrains Mono', size: 10 },
                            maxTicksLimit: (typeof window !== 'undefined' && window.innerWidth < 640) ? 5 : 8,
                            callback: function(val, index) {
                                const raw = this.getLabelForValue ? this.getLabelForValue(val) : String(val);
                                if (typeof window !== 'undefined' && window.innerWidth < 640 && typeof raw === 'string') {
                                    const match = raw.match(/\d{4}-(\d{2}-\d{2})/);
                                    if (match) {
                                        return match[1].replace('-', '/');
                                    }
                                }
                                return raw;
                            }
                        }
                    },
                    y: {
                        grid: { color: theme.gridColor },
                        ticks: {
                            color: theme.textColor,
                            font: { family: 'JetBrains Mono', size: 10 },
                            callback: (v) => v.toLocaleString() + ' 張'
                        }
                    }
                }
            }
        });

        this.instances[chartKey] = chart;
        return chart;
    }

    /**
     * Update colors on all active chart instances when theme toggles
     * @param {boolean} isDark
     */
    updateTheme(isDark) {
        const theme = this.getThemeTokens(isDark);
        const charts = Object.values(this.instances).filter(Boolean);

        for (const chart of charts) {
            if (chart.options?.scales?.x?.grid) chart.options.scales.x.grid.color = theme.gridColor;
            if (chart.options?.scales?.x?.ticks) chart.options.scales.x.ticks.color = theme.textColor;
            if (chart.options?.scales?.y?.grid) chart.options.scales.y.grid.color = theme.gridColor;
            if (chart.options?.scales?.y?.ticks) chart.options.scales.y.ticks.color = theme.textColor;

            if (chart.options?.plugins?.tooltip) {
                chart.options.plugins.tooltip.backgroundColor = theme.tooltipBg;
                chart.options.plugins.tooltip.titleColor = theme.tooltipTitle;
                chart.options.plugins.tooltip.bodyColor = theme.tooltipBody;
                chart.options.plugins.tooltip.borderColor = theme.tooltipBorder;
            }
            if (chart.options?.plugins?.legend?.labels) {
                chart.options.plugins.legend.labels.color = theme.textColor;
            }

            // Theme-dependent dataset styling for pricePredictChart
            if (chart.data?.datasets) {
                const priceDs = chart.data.datasets.find(d => d.key === 'price');
                if (priceDs) {
                    priceDs.backgroundColor = isDark ? 'rgba(56, 189, 248, 0.08)' : 'rgba(14, 165, 233, 0.05)';
                }
                const ma60Ds = chart.data.datasets.find(d => d.key === 'ma60');
                if (ma60Ds) {
                    ma60Ds.borderColor = isDark ? '#94A3B8' : '#64748B';
                    ma60Ds.pointBackgroundColor = isDark ? '#94A3B8' : '#64748B';
                }
                // Institutional chart total net fill update
                const instTotalDs = chart.data.datasets.find(d => d.label === '三大法人合計');
                if (instTotalDs) {
                    instTotalDs.backgroundColor = isDark ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.05)';
                }
            }

            try {
                chart.update();
            } catch (err) {
                console.debug('Chart theme update deferred:', err);
            }
        }
    }
}

// Global fallback for non-module usage
if (typeof window !== 'undefined') {
    window.ChartManager = ChartManager;
}
