# 888-stock-quant

Zero-dependency multi-agent AI Skill installer and quantitative market intelligence CLI for **888 Stock Quant** (`https://stockdata.david888.com`).

## Supported Agents

`npx -y 888-stock-quant@latest` installs `stock-quant/SKILL.md` into native discovery paths for:

| Agent | Global Path | Project Path (`--project`) |
| --- | --- | --- |
| **Antigravity (AGY)** | `~/.gemini/skills/stock-quant/SKILL.md` | `.gemini/skills/stock-quant/SKILL.md` |
| **Codex CLI** | `~/.codex/skills/stock-quant/SKILL.md` | `.codex/skills/stock-quant/SKILL.md` |
| **OpenCode** | `~/.config/opencode/skills/stock-quant/SKILL.md` | `.opencode/skills/stock-quant/SKILL.md` |
| **Claude Code** | `~/.claude/skills/stock-quant/SKILL.md` | `.claude/skills/stock-quant/SKILL.md` |
| **Hermes Agent** | `${HERMES_HOME:-~/.hermes}/skills/stock-quant/SKILL.md` | `.hermes/skills/stock-quant/SKILL.md` |
| **OpenClaw** | `~/.openclaw/workspace/skills/stock-quant/SKILL.md` | `./skills/stock-quant/SKILL.md` |
| **OMP (oh-my-pi) & Shared** | `~/.agents/skills/stock-quant/SKILL.md` | `.agents/skills/stock-quant/SKILL.md` |

## Pre-Use Update Check (Recommended for All Agents)

Always check whether a newer CLI version or live `/skill` spec exists before using the skill:

```bash
npx -y 888-stock-quant@latest check-update
npx -y 888-stock-quant@latest update
```

## Quick Start

```bash
# Auto-detect installed agents and install latest SKILL.md
npx -y 888-stock-quant@latest install

# Target specific agent(s) or all agents
npx -y 888-stock-quant@latest install --agent codex,opencode,hermes,openclaw
npx -y 888-stock-quant@latest install --all

# Query live quant endpoints without Python
npx -y 888-stock-quant@latest macro --json
npx -y 888-stock-quant@latest resonance --index TW_TOP50 --limit 10
npx -y 888-stock-quant@latest screen --direction bull --models 2 --limit 15
npx -y 888-stock-quant@latest query 2330.TW --limit 20
npx -y 888-stock-quant@latest align 2330.TW
npx -y 888-stock-quant@latest xray 2330.TW 2454.TW NVDA --window 252
```
