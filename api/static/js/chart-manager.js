/**
 * chart-manager.js - Chart.js Lifecycle & Theme Management for 888 Stock Quant
 * Manages institutional flow and broker branch accumulation trend charts with dark/light mode reactivity.
 * Features an internal instances map to avoid canvas memory leaks and ensure proper teardown.
 */

/** Build chart points from recorded forecast and target sessions, without inferred dates. */
export function buildVerifiedPriceChartData(historyRecords) {
    const points = new Map();
    let legacyCount = 0;
    const empty = date => ({date, price: null, ma60: null, timesfm: null, timestral: null, moirai: null,
        timestralP10: null, timestralP90: null, resonance: null, lstm: null, meta: {}});
    const ensure = date => {
        if (!points.has(date)) points.set(date, empty(date));
        return points.get(date);
    };
    const validDate = value => /^\d{4}-\d{2}-\d{2}$/.test(String(value || '').slice(0, 10));
    const positive = value => value !== null && value !== undefined && Number.isFinite(Number(value)) && Number(value) > 0
        ? Number(value) : null;
    const sorted = [...(historyRecords || [])].sort((a, b) => String(a.timestamp || '').localeCompare(String(b.timestamp || '')));
    for (const row of sorted) {
        const source = String(row.forecast_session || row.timestamp || '').slice(0, 10);
        if (!validDate(source)) continue;
        const sourcePoint = ensure(source);
        const base = positive(row.current_price);
        if (base !== null) sourcePoint.price = base;
        const ma60 = positive(row.ma60);
        if (ma60 !== null) sourcePoint.ma60 = ma60;
        const name = `${row.strategy_type || ''} ${row.model_name || ''}`.toLowerCase();
        let key = null;
        if (name.includes('共振') || name.includes('resonance')) key = 'resonance';
        else if (name.includes('timestral')) key = 'timestral';
        else if (name.includes('timesfm')) key = 'timesfm';
        else if (name.includes('moirai')) key = 'moirai';
        else if (name.includes('lstm')) key = 'lstm';
        const prediction = positive(row.predicted_price);
        if (!key || prediction === null || Number(row.horizon || 1) !== 1) continue;
        let target = validDate(row.target_date) ? String(row.target_date).slice(0, 10) : null;
        if (!target) {
            // 向下相容歷史資料：無 target_date 時，預測值直接錨定在發布日 source 上，確保走勢圖完整呈現歷史預測軌跡
            target = source;
            legacyCount += 1;
        }
        const point = ensure(target);
        point[key] = prediction;
        const actual = row.verification_status === 'settled' ? positive(row.actual_price) : null;
        point.meta[key] = {source, target, base, actual, status: row.verification_status || 'pending',
            revision: row.model_revision || null};
        if (actual !== null) point.price = actual;
        if (key === 'timestral') {
            point.timestralP10 = positive(row.p10 ?? row.timestral_p10);
            point.timestralP90 = positive(row.p90 ?? row.timestral_p90);
        }
    }
    const dates = [...points.keys()].sort();
    const values = key => dates.map(date => points.get(date)[key]);
    const meta = key => dates.map(date => points.get(date).meta[key] || null);
    return { dates, price: values('price'), ma60: values('ma60'), timesfm: values('timesfm'), moirai: values('moirai'),
        timestral: values('timestral'), timestralP10: values('timestralP10'),
        timestralP90: values('timestralP90'), resonance: values('resonance'), lstm: values('lstm'),
        meta: {timesfm: meta('timesfm'), timestral: meta('timestral'), moirai: meta('moirai'), resonance: meta('resonance'), lstm: meta('lstm')},
        legacy_count: legacyCount };
}

export class ChartManager {
    constructor() {
        this.instances = {};
        if (typeof Chart !== 'undefined' && typeof ChartZoom !== 'undefined' && Chart.register) {
            try {
                Chart.register(ChartZoom);
            } catch (_) {}
        }
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
                if (chartKey === 'pricePredictChart') {
                    this.autoFitPriceChartYScale(chart);
                }
                chart.update();
                return true;
            }
        }
        const ds = chart.data.datasets.find(d => d.key === seriesKey);
        if (ds) {
            ds.hidden = !isVisible;
            if (chartKey === 'pricePredictChart') {
                this.autoFitPriceChartYScale(chart);
            }
            chart.update();
            return true;
        }
        return false;
    }

    /**
     * Dynamically auto-fit Y-axis scale and point radii to visible X-range in price chart
     * @param {Object} [chartInstance] - Optional Chart.js instance, defaults to this.instances['pricePredictChart']
     */
    autoFitPriceChartYScale(chartInstance) {
        const chart = chartInstance || this.instances['pricePredictChart'];
        if (!chart || !chart.data || !chart.data.datasets || !chart.data.labels) return;

        const totalLabels = chart.data.labels.length;
        if (totalLabels <= 0) return;

        let rawMin = chart.scales?.x?.min !== undefined ? chart.scales.x.min : chart.options?.scales?.x?.min;
        let rawMax = chart.scales?.x?.max !== undefined ? chart.scales.x.max : chart.options?.scales?.x?.max;

        if (typeof rawMin === 'string') {
            rawMin = chart.data.labels.indexOf(rawMin);
        }
        if (typeof rawMax === 'string') {
            rawMax = chart.data.labels.indexOf(rawMax);
        }

        const minIdx = Math.max(0, Math.floor(rawMin ?? 0));
        const maxIdx = Math.min(totalLabels - 1, Math.ceil(rawMax ?? (totalLabels - 1)));

        // Collect all numeric values across active visible datasets within visible window
        const vals = [];
        for (const ds of chart.data.datasets) {
            if (ds.hidden) continue;
            if (!Array.isArray(ds.data)) continue;
            for (let i = minIdx; i <= maxIdx; i++) {
                const v = ds.data[i];
                if (v !== null && v !== undefined && !isNaN(v) && typeof v === 'number' && v > 0) {
                    vals.push(v);
                }
            }
        }

        if (vals.length > 0) {
            const minVal = Math.min(...vals);
            const maxVal = Math.max(...vals);
            const pad = Math.max((maxVal - minVal) * 0.08, minVal * 0.012, 0.02);

            if (!chart.options.scales) chart.options.scales = {};
            if (!chart.options.scales.y) chart.options.scales.y = {};
            chart.options.scales.y.min = Math.max(0, Number((minVal - pad).toFixed(2)));
            chart.options.scales.y.max = Number((maxVal + pad).toFixed(2));
            if (chart.scales && chart.scales.y) {
                chart.scales.y.min = chart.options.scales.y.min;
                chart.scales.y.max = chart.options.scales.y.max;
            }
        } else if (chart.options?.scales?.y) {
            delete chart.options.scales.y.min;
            delete chart.options.scales.y.max;
        }

        // Dynamically adjust point sizes when zoomed in (visibleCount <= 45 trading days)
        const visibleCount = maxIdx - minIdx + 1;
        const isRelaxed = visibleCount <= 45;
        for (const ds of chart.data.datasets) {
            if (ds._relaxedPointRadius !== undefined && ds._densePointRadius !== undefined) {
                ds.pointRadius = isRelaxed ? ds._relaxedPointRadius : ds._densePointRadius;
            } else if (ds.key === 'price') {
                ds.pointRadius = isRelaxed ? 3.5 : 1.5;
            }
        }
    }

    /**
     * Render Stock Price vs AI Target Price (LSTM, TimesFM, Resonance, MA60) Trajectory Chart
     * @param {HTMLCanvasElement|string} canvas - Canvas element or ID
     * @param {Array} historyRecords - Array of raw prediction history records
     * @param {boolean} isDark - Dark mode active flag
     * @param {Object} visibility - Map of seriesKey -> boolean
     * @param {number|string} rangeDays - Timeframe window in trading days (0 or 'ALL' for full range)
     */
    renderPricePredictChart(canvas, historyRecords, isDark, visibility = {}, rangeDays = 0) {
        const el = typeof canvas === 'string' ? document.getElementById(canvas) : canvas;
        if (!el || !historyRecords || !historyRecords.length || typeof Chart === 'undefined') return null;

        if (typeof ChartZoom !== 'undefined' && Chart.register) {
            try {
                Chart.register(ChartZoom);
            } catch (_) {}
        }

        const chartKey = 'pricePredictChart';
        this.destroy(chartKey);

        const series = buildVerifiedPriceChartData(historyRecords);
        const dates = series.dates;
        if (!dates.length) return null;
        const n = dates.length;

        // Each prediction stays on its recorded target date, including missing sessions.
        const labels = [...dates];
        const totalPoints = labels.length;
        let xMin = undefined;
        let xMax = undefined;
        if (rangeDays && rangeDays !== 'ALL' && Number(rangeDays) > 0 && totalPoints > Number(rangeDays) + 1) {
            xMin = Math.max(0, totalPoints - 1 - Number(rangeDays));
            xMax = totalPoints - 1;
        }

        const theme = this.getThemeTokens(isDark);

        const dataPrice = series.price;
        const dataMa60 = series.ma60;
        const dataTimestral = series.timestral;
        const dataTimestralP10 = series.timestralP10;
        const dataTimestralP90 = series.timestralP90;
        const dataTimesfm = series.timesfm;
        const dataMoirai = series.moirai;
        const dataResonance = series.resonance;
        const dataLstm = series.lstm;

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
        const colorMoirai = isDark ? '#2DD4BF' : '#0F766E';
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
        const computeMarkerStyles = (predArray, defaultStyle, modelColor, modelKey) => {
            const pointStyles = [];
            const denseRadii = [];
            const relaxedRadii = [];
            const pointHoverRadii = [];
            const pointBgColors = [];
            const pointBorderColors = [];
            const pointBorderWidths = [];

            const denseHit = 2.5, relaxedHit = 4.5;
            const denseMiss = 4.5, relaxedMiss = 6.5;
            const denseT1 = 5.5, relaxedT1 = 7.5;

            for (let idx = 0; idx < n; idx++) {
                const val = predArray[idx];
                if (val === null || val === undefined || isNaN(val)) {
                    pointStyles.push(defaultStyle);
                    denseRadii.push(0);
                    relaxedRadii.push(0);
                    pointHoverRadii.push(0);
                    pointBgColors.push('transparent');
                    pointBorderColors.push('transparent');
                    pointBorderWidths.push(0);
                    continue;
                }

                const evidence = series.meta[modelKey]?.[idx];
                if (evidence && evidence.actual === null) {
                    pointStyles.push('star');
                    denseRadii.push(denseT1);
                    relaxedRadii.push(relaxedT1);
                    pointHoverRadii.push(relaxedT1 + 2.5);
                    pointBgColors.push(modelColor);
                    pointBorderColors.push(isDark ? '#FFFFFF' : '#0F172A');
                    pointBorderWidths.push(2);
                    continue;
                }

                if (evidence) {
                    const basePrice = evidence.base;
                    const actualPrice = evidence.actual;

                    if (basePrice && basePrice > 0 && actualPrice && actualPrice > 0) {
                        const predDiff = val - basePrice;
                        const actDiff = actualPrice - basePrice;
                        const isHit = Math.sign(predDiff) === Math.sign(actDiff);

                        if (isHit) {
                            // 方向命中 (Hit): 實心幾何形狀
                            pointStyles.push(defaultStyle);
                            denseRadii.push(denseHit);
                            relaxedRadii.push(relaxedHit);
                            pointHoverRadii.push(relaxedHit + 2.5);
                            pointBgColors.push(modelColor);
                            pointBorderColors.push(isDark ? '#0D0E11' : '#FFFFFF');
                            pointBorderWidths.push(1.5);
                        } else {
                            // 方向相反 / 偏離 (Miss / Opposite direction): ✕ 叉號標點
                            pointStyles.push('crossRot');
                            denseRadii.push(denseMiss);
                            relaxedRadii.push(relaxedMiss);
                            pointHoverRadii.push(relaxedMiss + 3);
                            pointBgColors.push(colorOpposite);
                            pointBorderColors.push(colorOpposite);
                            pointBorderWidths.push(2.5);
                        }
                        continue;
                    }
                }

                // Fallback for missing baseline/actual price
                pointStyles.push(defaultStyle);
                denseRadii.push(denseHit);
                relaxedRadii.push(relaxedHit);
                pointHoverRadii.push(relaxedHit + 2);
                pointBgColors.push(modelColor);
                pointBorderColors.push(isDark ? '#0D0E11' : '#FFFFFF');
                pointBorderWidths.push(1.5);
            }

            const isDense = labels.length > 40;
            return {
                pointStyle: pointStyles,
                pointRadius: isDense ? [...denseRadii] : [...relaxedRadii],
                _densePointRadius: denseRadii,
                _relaxedPointRadius: relaxedRadii,
                pointHoverRadius: pointHoverRadii,
                pointBackgroundColor: pointBgColors,
                pointBorderColor: pointBorderColors,
                pointBorderWidth: pointBorderWidths
            };
        };

        const timestralMarkers = computeMarkerStyles(dataTimestral, 'rect', colorTimestral, 'timestral');
        const timesfmMarkers = computeMarkerStyles(dataTimesfm, 'rectRot', colorTimesfm, 'timesfm');
        const moiraiMarkers = computeMarkerStyles(dataMoirai, 'circle', colorMoirai, 'moirai');
        const resonanceMarkers = computeMarkerStyles(dataResonance, 'triangle', colorResonance, 'resonance');
        const lstmMarkers = computeMarkerStyles(dataLstm, 'circle', colorLstm, 'lstm');

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
                        _densePointRadius: 1.5,
                        _relaxedPointRadius: 3.5,
                        pointHoverRadius: 6,
                        pointBackgroundColor: colorPrice,
                        pointBorderColor: isDark ? '#0D0E11' : '#FFFFFF',
                        pointBorderWidth: 1.5,
                        pointStyle: 'circle',
                        fill: true,
                        spanGaps: false,
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
                        spanGaps: false,
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
                        spanGaps: false,
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
                        spanGaps: false,
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
                        spanGaps: false,
                        hidden: visibility.timesfm === false
                    },
                    {
                        key: 'moirai',
                        label: 'MOIRAI-2 目標價',
                        data: dataMoirai,
                        borderColor: colorMoirai,
                        backgroundColor: 'transparent',
                        borderDash: [8, 4],
                        borderWidth: 2.4,
                        tension: 0.25,
                        ...moiraiMarkers,
                        spanGaps: false,
                        hidden: visibility.moirai === false
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
                        spanGaps: false,
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
                        spanGaps: false,
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
                        spanGaps: false,
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
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x',
                            threshold: 4,
                            onPanComplete: ({ chart }) => {
                                this.autoFitPriceChartYScale(chart);
                                chart.update('none');
                            }
                        },
                        zoom: {
                            wheel: {
                                enabled: true,
                                speed: 0.12
                            },
                            pinch: {
                                enabled: true
                            },
                            mode: 'x',
                            onZoomComplete: ({ chart }) => {
                                this.autoFitPriceChartYScale(chart);
                                chart.update('none');
                            }
                        },
                        limits: {
                            x: { min: 0, max: labels.length - 1, minRange: 6 }
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
                            title(items) {
                                return items && items.length ? `交易日期: ${items[0].label}` : '';
                            },
                            label(ctx) {
                                const val = ctx.parsed.y;
                                if (val === null || val === undefined || !Number.isFinite(Number(val))) return null;
                                const formatted = '$' + Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 2 });
                                const key = ctx.dataset.key;
                                const evidenceKey = key === 'timestralBandsUpper' || key === 'timestralBandsLower' ? 'timestral' : key;
                                const evidence = series.meta[evidenceKey]?.[ctx.dataIndex];
                                if (!evidence) return ` ${ctx.dataset.label}: ${formatted}`;
                                const first = ` ${ctx.dataset.label} (${evidence.source} 預測): ${formatted}`;
                                if (evidence.actual === null) return [first, `   目標日 ${evidence.target} · 待驗證`];
                                if (key === 'timestralBandsUpper' || key === 'timestralBandsLower') return first;
                                const error = Math.abs(val - evidence.actual) / evidence.actual * 100;
                                if (!evidence.base) return [first, `   已驗證 · 誤差 ${error.toFixed(2)}%`];
                                const hit = Math.sign(val - evidence.base) === Math.sign(evidence.actual - evidence.base);
                                return [first, `   ${hit ? '方向命中' : '方向偏離'} · 誤差 ${error.toFixed(2)}%`];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        min: xMin,
                        max: xMax,
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
        this.autoFitPriceChartYScale(chart);
        chart.update('none');
        return chart;
    }

    /**
     * Slice or zoom the price chart to a specific trading day timeframe
     * @param {number|string} rangeDays - Number of trading days (e.g. 22, 45, 66, 130) or 0/'ALL'
     * @returns {boolean}
     */
    setPriceChartRange(rangeDays) {
        const chart = this.instances['pricePredictChart'];
        if (!chart || !chart.data || !chart.data.labels) return false;
        const total = chart.data.labels.length;
        if (!total) return false;

        if (!rangeDays || rangeDays === 'ALL' || Number(rangeDays) <= 0 || total <= Number(rangeDays) + 1) {
            if (typeof chart.resetZoom === 'function') {
                try {
                    chart.resetZoom('none');
                } catch (e) {
                    console.debug('resetZoom error fallback:', e);
                }
            }
            if (chart.options.scales && chart.options.scales.x) {
                delete chart.options.scales.x.min;
                delete chart.options.scales.x.max;
            }
            if (chart.scales && chart.scales.x) {
                chart.scales.x.min = 0;
                chart.scales.x.max = total - 1;
            }
        } else {
            const days = Number(rangeDays);
            const min = Math.max(0, total - 1 - days);
            const max = total - 1;
            if (!chart.options.scales) chart.options.scales = {};
            if (!chart.options.scales.x) chart.options.scales.x = {};
            chart.options.scales.x.min = min;
            chart.options.scales.x.max = max;
            if (chart.scales && chart.scales.x) {
                chart.scales.x.min = min;
                chart.scales.x.max = max;
            }
        }

        this.autoFitPriceChartYScale(chart);
        chart.update('none');
        return true;
    }

    /**
     * Zoom in or out by a scaling factor
     * @param {number} factor - Zoom factor (> 1 zooms in, < 1 zooms out)
     * @returns {boolean}
     */
    zoomPriceChart(factor = 1.25) {
        const chart = this.instances['pricePredictChart'];
        if (!chart || !chart.data || !chart.data.labels) return false;

        const total = chart.data.labels.length;
        let handled = false;
        if (typeof chart.zoom === 'function') {
            try {
                chart.zoom(factor);
                handled = true;
                if (chart.scales?.x) {
                    const syncedMin = Math.max(0, Math.floor(chart.scales.x.min ?? 0));
                    const syncedMax = Math.min(total - 1, Math.ceil(chart.scales.x.max ?? (total - 1)));
                    if (!chart.options.scales) chart.options.scales = {};
                    if (!chart.options.scales.x) chart.options.scales.x = {};
                    chart.options.scales.x.min = syncedMin;
                    chart.options.scales.x.max = syncedMax;
                }
            } catch (e) {
                console.debug('chart.zoom error, using fallback:', e);
            }
        }

        if (!handled) {
            let min = chart.scales?.x?.min !== undefined ? chart.scales.x.min : (chart.options?.scales?.x?.min ?? 0);
            let max = chart.scales?.x?.max !== undefined ? chart.scales.x.max : (chart.options?.scales?.x?.max ?? (total - 1));
            if (typeof min === 'string') min = chart.data.labels.indexOf(min);
            if (typeof max === 'string') max = chart.data.labels.indexOf(max);
            if (min < 0 || isNaN(min)) min = 0;
            if (max >= total || isNaN(max)) max = total - 1;

            const curRange = Math.max(6, max - min);
            const newRange = Math.max(6, Math.min(total - 1, Math.round(curRange / factor)));
            const newMin = Math.max(0, max - newRange);

            if (!chart.options.scales) chart.options.scales = {};
            if (!chart.options.scales.x) chart.options.scales.x = {};
            chart.options.scales.x.min = newMin;
            chart.options.scales.x.max = max;
            if (chart.scales && chart.scales.x) {
                chart.scales.x.min = newMin;
                chart.scales.x.max = max;
            }
        }

        this.autoFitPriceChartYScale(chart);
        chart.update('none');
        return true;
    }

    /**
     * Pan visible window horizontally by ratio
     * @param {number} stepRatio - Fraction of current window to move (+ moves forward/right, - moves backward/left)
     * @returns {boolean}
     */
    panPriceChart(stepRatio = 0.25) {
        const chart = this.instances['pricePredictChart'];
        if (!chart || !chart.data || !chart.data.labels) return false;
        const total = chart.data.labels.length;
        let min = chart.scales?.x?.min !== undefined ? chart.scales.x.min : (chart.options?.scales?.x?.min ?? 0);
        let max = chart.scales?.x?.max !== undefined ? chart.scales.x.max : (chart.options?.scales?.x?.max ?? (total - 1));
        if (typeof min === 'string') min = chart.data.labels.indexOf(min);
        if (typeof max === 'string') max = chart.data.labels.indexOf(max);
        if (min < 0 || isNaN(min)) min = 0;
        if (max >= total || isNaN(max)) max = total - 1;

        const curRange = max - min;
        const delta = Math.round(curRange * stepRatio);
        let newMin = min + delta;
        let newMax = max + delta;

        if (newMin < 0) {
            newMax += (0 - newMin);
            newMin = 0;
        }
        if (newMax > total - 1) {
            newMin -= (newMax - (total - 1));
            newMax = total - 1;
            newMin = Math.max(0, newMin);
        }

        if (!chart.options.scales) chart.options.scales = {};
        if (!chart.options.scales.x) chart.options.scales.x = {};
        chart.options.scales.x.min = newMin;
        chart.options.scales.x.max = newMax;
        if (chart.scales && chart.scales.x) {
            chart.scales.x.min = newMin;
            chart.scales.x.max = newMax;
        }
        this.autoFitPriceChartYScale(chart);
        chart.update('none');
        return true;
    }

    /**
     * Reset zoom and pan to display full date range
     * @returns {boolean}
     */
    resetPriceChartZoom() {
        return this.setPriceChartRange('ALL');
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
                    priceDs.backgroundColor = isDark ? 'rgba(96, 165, 250, 0.08)' : 'rgba(29, 78, 216, 0.05)';
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
