# 888-stock-quant

Zero-dependency multi-agent AI Skill installer, updater, and quantitative market intelligence CLI for **888 Stock Quant** (`https://stockdata.david888.com`).

Integrates **Macro Risk Regime**, **Xuantie MA60/MA120 Pullbacks**, **Mistral Timestral-209M**, **Google TimesFM**, **Salesforce MOIRAI-2**, **Multi-Model Resonance (`👑 四重共振 / 🏆 三重共振`)**, **TWSE/TPEx Institutional Flows**, and **Portfolio Risk X-Ray** directly into your AI coding agent or terminal — with **zero Python dependencies**.

## Supported AI Agents

`npx -y 888-stock-quant@latest install` automatically detects installed agents and writes `stock-quant/SKILL.md` into their native discovery paths:

| Agent | Flag | Global Skill Path | Project Path (`--project`) |
| --- | --- | --- | --- |
| **Antigravity (AGY)** | `--agent agy` | `~/.gemini/skills/stock-quant/SKILL.md` | `.gemini/skills/stock-quant/SKILL.md` |
| **Codex CLI** | `--agent codex` | `~/.codex/skills/stock-quant/SKILL.md` | `.codex/skills/stock-quant/SKILL.md` |
| **OpenCode** | `--agent opencode` | `~/.config/opencode/skills/stock-quant/SKILL.md` | `.opencode/skills/stock-quant/SKILL.md` |
| **Claude Code** | `--agent claude` | `~/.claude/skills/stock-quant/SKILL.md` | `.claude/skills/stock-quant/SKILL.md` |
| **Hermes Agent** | `--agent hermes` | `${HERMES_HOME:-~/.hermes}/skills/stock-quant/SKILL.md` | `.hermes/skills/stock-quant/SKILL.md` |
| **OpenClaw** | `--agent openclaw` | `~/.openclaw/workspace/skills/stock-quant/SKILL.md` | `./skills/stock-quant/SKILL.md` |
| **OMP & Shared** | `--agent omp` / `agents` | `~/.agents/skills/stock-quant/SKILL.md` | `.agents/skills/stock-quant/SKILL.md` |

## Mandatory Pre-Use Update Check (For LLM Agents)

The 888 Stock Quant brain evolves continuously with new time-series models, REST endpoints, and MCP tools. Installed LLM Agents should **check for updates before every session/task**:

```bash
# Check both npm @latest CLI version and live /skill SHA-256 fingerprint
npx -y 888-stock-quant@latest check-update

# Update all installed local SKILL.md copies from https://stockdata.david888.com/skill
npx -y 888-stock-quant@latest update
```

## Install & Sync Commands

```bash
# Auto-detect installed agents and install latest SKILL.md
npx -y 888-stock-quant@latest install

# Install for specific agent(s) or every supported agent
npx -y 888-stock-quant@latest install --agent codex,opencode,hermes,openclaw
npx -y 888-stock-quant@latest install --all

# Install into current workspace instead of home directory
npx -y 888-stock-quant@latest install --project

# Install into a custom directory / preview without writing
npx -y 888-stock-quant@latest install --dir /path/to/skills --dry-run

# Print live or bundled SKILL.md / llms.txt to stdout
npx -y 888-stock-quant@latest skill
npx -y 888-stock-quant@latest skill --llms
```

## Quant Market CLI (Zero Python Required)

Every command supports `--json` for direct LLM/script consumption and `--base-url <url>` (or `STOCK_QUANT_BASE_URL`) for self-hosted deployments:

```bash
# 1. Macro risk regime & recommended equity exposure
npx -y 888-stock-quant@latest macro --json

# 2. Multi-model & multi-strategy resonance picks
npx -y 888-stock-quant@latest resonance --index TW_TOP50 --limit 10 --json

# 3. Multi-factor quant screener
npx -y 888-stock-quant@latest screen --direction bull --models 2 --potential 5 --limit 15 --json

# 4. Xuantie MA60/MA120 trend pullback setups
npx -y 888-stock-quant@latest xuantie --index TW_TOP50 --limit 10

# 5. Ticker prediction trajectory, multi-source alignment & name resolution
npx -y 888-stock-quant@latest resolve 台積電
npx -y 888-stock-quant@latest query 2330.TW --limit 20 --json
npx -y 888-stock-quant@latest align 2330.TW --json

# 6. Portfolio covariance & VaR/CVaR Risk X-Ray (POST /api/v1/risk/xray)
npx -y 888-stock-quant@latest xray 2330.TW 2454.TW NVDA --window 252 --json

# 7. Cross-market live trading sessions & data-loader canary health
npx -y 888-stock-quant@latest sessions --json
npx -y 888-stock-quant@latest canary --json
```

## Links & Standards

- Live Web Dashboard & Agent Hub: [https://stockdata.david888.com](https://stockdata.david888.com)
- Live Skill Specification: [`/skill`](https://stockdata.david888.com/skill)
- LLM Discovery Index: [`/llms.txt`](https://stockdata.david888.com/llms.txt) | [`/llms-full.txt`](https://stockdata.david888.com/llms-full.txt)
- WebMCP Manifest: [`/.well-known/mcp.json`](https://stockdata.david888.com/.well-known/mcp.json)
- Source Repository: [GitHub (`tbdavid2019/stock-underdog-ml`)](https://github.com/tbdavid2019/stock-underdog-ml)
