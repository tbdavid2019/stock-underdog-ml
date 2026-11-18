/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './api/templates/**/*.html',
    './api/static/js/**/*.js',
  ],
  theme: {
    extend: {
      fontSize: {
        'xs': ['13px', { lineHeight: '18px' }],
        'sm': ['14px', { lineHeight: '20px' }],
      },
      fontFamily: {
        sans: ['"Noto Sans TC"', '-apple-system', 'BlinkMacSystemFont', '"PingFang TC"', '"Microsoft JhengHei"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      colors: {
        terminal: {
          'dark-bg': '#030302',
          'dark-card': '#0D0E11',
          'dark-card-inner': '#16181D',
          'dark-border': '#262626',
          'amber': '#FFB010',
          'amber-hover': '#F2A600',
          'profit': '#24D97A',
          'loss': '#EF4444',
        },
      },
    },
  },
  safelist: [
    'bg-emerald-950/40',
    'bg-emerald-950/50',
    'bg-rose-950/40',
    'bg-amber-950/40',
    'bg-indigo-950/40',
    'bg-sky-950/40',
    'bg-cyan-950/40',
    'bg-purple-950/40',
    'bg-teal-950/40',
    'border-emerald-800/40',
    'border-rose-800/40',
    'border-amber-800/40',
    'border-indigo-800/40',
    'border-sky-800/40',
    'border-cyan-800/40',
    'border-purple-800/40',
    'border-teal-800/40',
    'shadow-amber-500/25',
    'shadow-sky-600/20',
  ],
  plugins: [],
};
