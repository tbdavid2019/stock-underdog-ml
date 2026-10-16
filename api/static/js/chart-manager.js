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
            gridColor: isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)',
            textColor: isDark ? '#9CA3AF' : '#6B7280',
            tooltipBg: isDark ? '#0D0E11' : '#FFFFFF',
            tooltipTitle: isDark ? '#F9FAFB' : '#111827',
            tooltipBody: isDark ? '#E5E7EB' : '#374151',
            tooltipBorder: isDark ? '#262626' : '#E5E7EB'
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
                        borderColor: '#10B981',
                        backgroundColor: isDark ? 'rgba(16, 185, 129, 0.08)' : 'rgba(16, 185, 129, 0.05)',
                        borderWidth: 2.5,
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 2,
                        pointHoverRadius: 5,
                        fill: true
                    },
                    {
                        label: '外資累計',
                        data: dataForeign,
                        borderColor: '#38BDF8',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: '投信累計',
                        data: dataTrust,
                        borderColor: '#F43F5E',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: labels.length > 40 ? 0 : 2,
                        pointHoverRadius: 5
                    },
                    {
                        label: '自營商累計',
                        data: dataDealer,
                        borderColor: '#F59E0B',
                        borderWidth: 1.5,
                        borderDash: [3, 3],
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
                            maxTicksLimit: 8
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
     */
    setDatasetVisibility(chartKey, seriesKey, isVisible) {
        const chart = this.instances[chartKey];
        if (!chart || !chart.data || !chart.data.datasets) return;
        const ds = chart.data.datasets.find(d => d.key === seriesKey);
        if (ds) {
            ds.hidden = !isVisible;
            chart.update();
        }
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
        const sorted = [...historyRecords].sort((a, b) => (a.timestamp || '').localeCompare(b.timestamp || ''));

        // Group by Date (YYYY-MM-DD)
        const dateMap = new Map();
        for (const r of sorted) {
            if (!r.timestamp) continue;
            const dateKey = r.timestamp.slice(0, 10);
            if (!dateMap.has(dateKey)) {
                dateMap.set(dateKey, {
                    date: dateKey,
                    price: null,
                    ma60: null,
                    lstm: null,
                    timesfm: null,
                    resonance: null
                });
            }
            const item = dateMap.get(dateKey);
            const curPrice = Number(r.current_price);
            if (!isNaN(curPrice) && curPrice > 0) item.price = curPrice;
            const ma60 = Number(r.ma60);
            if (!isNaN(ma60) && ma60 > 0) item.ma60 = ma60;
            const predPrice = Number(r.predicted_price);
            if (!isNaN(predPrice) && predPrice > 0) {
                const strat = ((r.strategy_type || '') + ' ' + (r.model_name || '')).toLowerCase();
                if (strat.includes('lstm')) item.lstm = predPrice;
                else if (strat.includes('timesfm')) item.timesfm = predPrice;
                else if (strat.includes('共振')) item.resonance = predPrice;
            }
        }

        const labels = Array.from(dateMap.keys());
        if (!labels.length) return null;

        const theme = this.getThemeTokens(isDark);

        const dataPrice = labels.map(d => dateMap.get(d).price);
        const dataLstm = labels.map(d => dateMap.get(d).lstm);
        const dataTimesfm = labels.map(d => dateMap.get(d).timesfm);
        const dataResonance = labels.map(d => dateMap.get(d).resonance);
        const dataMa60 = labels.map(d => dateMap.get(d).ma60);

        const chart = new Chart(el, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        key: 'price',
                        label: '真實股價',
                        data: dataPrice,
                        borderColor: '#38BDF8',
                        backgroundColor: isDark ? 'rgba(56, 189, 248, 0.08)' : 'rgba(14, 165, 233, 0.05)',
                        borderWidth: 2.8,
                        tension: 0.25,
                        pointRadius: labels.length > 40 ? 1.5 : 3.5,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#38BDF8',
                        fill: true,
                        spanGaps: true,
                        hidden: visibility.price === false
                    },
                    {
                        key: 'lstm',
                        label: 'LSTM 目標價',
                        data: dataLstm,
                        borderColor: '#F59E0B',
                        backgroundColor: 'transparent',
                        borderDash: [5, 4],
                        borderWidth: 2.2,
                        tension: 0.25,
                        pointRadius: labels.length > 40 ? 2 : 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#F59E0B',
                        pointStyle: 'circle',
                        spanGaps: true,
                        hidden: visibility.lstm === false
                    },
                    {
                        key: 'timesfm',
                        label: 'TimesFM 目標價',
                        data: dataTimesfm,
                        borderColor: '#A855F7',
                        backgroundColor: 'transparent',
                        borderDash: [3, 3],
                        borderWidth: 2.2,
                        tension: 0.25,
                        pointRadius: labels.length > 40 ? 2.5 : 4.5,
                        pointHoverRadius: 6.5,
                        pointBackgroundColor: '#A855F7',
                        pointStyle: 'rectRot',
                        spanGaps: true,
                        hidden: visibility.timesfm === false
                    },
                    {
                        key: 'resonance',
                        label: '多維共振目標價',
                        data: dataResonance,
                        borderColor: '#10B981',
                        backgroundColor: 'transparent',
                        borderWidth: 2.4,
                        tension: 0.25,
                        pointRadius: labels.length > 40 ? 3 : 5,
                        pointHoverRadius: 7,
                        pointBackgroundColor: '#10B981',
                        pointStyle: 'triangle',
                        spanGaps: true,
                        hidden: visibility.resonance === false
                    },
                    {
                        key: 'ma60',
                        label: 'MA60 季線',
                        data: dataMa60,
                        borderColor: isDark ? '#94A3B8' : '#64748B',
                        backgroundColor: 'transparent',
                        borderDash: [2, 4],
                        borderWidth: 1.6,
                        tension: 0.2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        pointBackgroundColor: isDark ? '#94A3B8' : '#64748B',
                        spanGaps: true,
                        hidden: visibility.ma60 !== true
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
                                return `交易日期: ${items[0].label}`;
                            },
                            label(ctx) {
                                const val = ctx.parsed.y;
                                if (val === null || val === undefined || isNaN(val)) return null;
                                const formatted = '$' + Number(val).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 2 });
                                const label = ctx.dataset.label;
                                const dateIdx = ctx.dataIndex;
                                const priceDs = ctx.chart.data.datasets.find(d => d.key === 'price');
                                const actual = priceDs ? priceDs.data[dateIdx] : null;

                                if (actual && actual > 0 && ctx.dataset.key !== 'price' && ctx.dataset.key !== 'ma60') {
                                    const diff = ((val - actual) / actual) * 100;
                                    const sign = diff > 0 ? '+' : '';
                                    return ` ${label}: ${formatted} (${sign}${diff.toFixed(2)}% vs 現價)`;
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
                            maxTicksLimit: 10
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
                            maxTicksLimit: 8
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
            if (chart.options && chart.options.scales) {
                if (chart.options.scales.x) {
                    chart.options.scales.x.grid.color = theme.gridColor;
                    chart.options.scales.x.ticks.color = theme.textColor;
                }
                if (chart.options.scales.y) {
                    chart.options.scales.y.grid.color = theme.gridColor;
                    chart.options.scales.y.ticks.color = theme.textColor;
                }
            }
            if (chart.options && chart.options.plugins && chart.options.plugins.tooltip) {
                chart.options.plugins.tooltip.backgroundColor = theme.tooltipBg;
                chart.options.plugins.tooltip.titleColor = theme.tooltipTitle;
                chart.options.plugins.tooltip.bodyColor = theme.tooltipBody;
                chart.options.plugins.tooltip.borderColor = theme.tooltipBorder;
            }
            if (chart.options && chart.options.plugins && chart.options.plugins.legend && chart.options.plugins.legend.labels) {
                chart.options.plugins.legend.labels.color = theme.textColor;
            }

            // Theme-dependent dataset styling for pricePredictChart
            if (chart.data && chart.data.datasets) {
                const priceDs = chart.data.datasets.find(d => d.key === 'price');
                if (priceDs) {
                    priceDs.backgroundColor = isDark ? 'rgba(56, 189, 248, 0.08)' : 'rgba(14, 165, 233, 0.05)';
                }
                const ma60Ds = chart.data.datasets.find(d => d.key === 'ma60');
                if (ma60Ds) {
                    ma60Ds.borderColor = isDark ? '#94A3B8' : '#64748B';
                    ma60Ds.pointBackgroundColor = isDark ? '#94A3B8' : '#64748B';
                }
            }

            chart.update();
        }
    }
}

// Global fallback for non-module usage
if (typeof window !== 'undefined') {
    window.ChartManager = ChartManager;
}
