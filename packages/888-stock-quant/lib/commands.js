import fs from 'node:fs';
import os from 'node:os';
import {
  listSupportedAgentKeys,
  resolveInstallTargets,
  writeSkillToTargets,
} from './agents.js';
import {
  DEFAULT_BASE_URL,
  DEFAULT_NPM_REGISTRY_URL,
  apiRequest,
  compareSemver,
  fetchLiveLlmsTxt,
  fetchLiveSkill,
  fetchNpmLatest,
  readBundledSkill,
  readPackageMeta,
  sha256Short,
  validateBaseUrl,
} from './client.js';

const VALUE_FLAGS = new Set([
  'agent',
  'dir',
  'base-url',
  'index',
  'limit',
  'direction',
  'models',
  'pe',
  'potential',
  'rr',
  'ma60',
  'inst',
  'window',
  'registry-url',
]);

export function parseArgs(argv) {
  const positionals = [];
  const flags = {
    agent: [],
    json: false,
    offline: false,
    dryRun: false,
    project: false,
    all: false,
    llms: false,
    llmsFull: false,
    help: false,
    version: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') {
      flags.help = true;
    } else if (arg === '--version' || arg === '-v') {
      flags.version = true;
    } else if (arg === '--json') {
      flags.json = true;
    } else if (arg === '--offline') {
      flags.offline = true;
    } else if (arg === '--dry-run') {
      flags.dryRun = true;
    } else if (arg === '--project') {
      flags.project = true;
    } else if (arg === '--all') {
      flags.all = true;
      flags.agent.push('all');
    } else if (arg === '--llms') {
      flags.llms = true;
    } else if (arg === '--llms-full') {
      flags.llmsFull = true;
    } else if (arg.startsWith('--')) {
      const eqIdx = arg.indexOf('=');
      const rawKey = eqIdx === -1 ? arg.slice(2) : arg.slice(2, eqIdx);
      let val = eqIdx === -1 ? null : arg.slice(eqIdx + 1);
      if (!VALUE_FLAGS.has(rawKey)) {
        throw new Error(`Unknown flag "--${rawKey}"`);
      }
      if (val === null) {
        if (i + 1 >= argv.length) {
          throw new Error(`Flag "--${rawKey}" requires a value`);
        }
        val = argv[i + 1];
        i += 1;
      }
      if (rawKey === 'agent') {
        flags.agent.push(val);
      } else {
        flags[rawKey] = val;
      }
    } else {
      positionals.push(arg);
    }
  }

  return { positionals, flags };
}

export function formatHelpText(version) {
  return `888-stock-quant v${version} — Multi-Agent AI Skill Installer & Quant Brain CLI

Usage:
  npx -y 888-stock-quant@latest [command] [options]

Skill & Update Commands:
  install            Install/sync stock-quant SKILL.md into agent directories (default)
  update             Fetch latest /skill & update installed agent SKILL.md files
  check-update       Check npm latest version and live /skill SHA-256 for updates
  skill              Print SKILL.md (or pass --llms / --llms-full for llms.txt)

Quant API Commands (zero Python required):
  macro              Market macro regime & circuit-breaker risk score
  resonance          Multi-model resonance stocks (--index TW_TOP50 --limit 15)
  screen             Screener (--direction bull --models 2 --pe any --potential 5 --limit 20)
  xuantie            Xuantie MA pullback setups (--index TW_TOP50 --limit 15)
  query <ticker>     Prediction history & price summary (--limit 30)
  align <ticker>     Cross-market & institutional chip alignment
  resolve <query>    Resolve Chinese/English company name to ticker
  xray <tickers...>  Portfolio covariance X-Ray (--window 252)
  sessions           TW/US market holiday & trading session calendar
  canary             Canary data-loader health metrics

Install Options:
  --agent <name>     Target agent(s): ${listSupportedAgentKeys().join(', ')}, all
  --project          Install into current project directory instead of global home
  --dir <path>       Install into custom skills directory (e.g. ~/.openclaw/workspace/skills)
  --offline          Use bundled SKILL.md without network fetch
  --dry-run          Preview target paths without writing files
  --base-url <url>   Override API base URL (default: $STOCK_QUANT_BASE_URL or ${DEFAULT_BASE_URL})
  --json             Machine-readable JSON output
  -h, --help         Show this help
  -v, --version      Show CLI package version

IMPORTANT FOR LLM AGENTS:
  Before relying on cached SKILL.md rules or endpoint schemas, run:
    npx -y 888-stock-quant@latest check-update
  and upgrade with \`npx -y 888-stock-quant@latest update\` if an update is available.
`;
}

function getEffectiveBaseUrl(flags, env) {
  return validateBaseUrl(flags['base-url'] || env.STOCK_QUANT_BASE_URL || DEFAULT_BASE_URL);
}

function getRegistryUrl(flags, env) {
  return validateBaseUrl(flags['registry-url'] || env.STOCK_QUANT_NPM_REGISTRY || DEFAULT_NPM_REGISTRY_URL);
}

async function resolveSkillContent(baseUrl, { offline = false, fetchImpl = globalThis.fetch } = {}) {
  const bundled = readBundledSkill();
  if (offline) {
    return { content: bundled, source: 'bundled', hash: sha256Short(bundled), fallbackReason: null };
  }
  try {
    const live = await fetchLiveSkill(baseUrl, { fetchImpl });
    return { content: live, source: 'live', hash: sha256Short(live), fallbackReason: null };
  } catch (err) {
    return {
      content: bundled,
      source: 'bundled',
      hash: sha256Short(bundled),
      fallbackReason: err.message,
    };
  }
}

async function handleInstallOrUpdate(mode, flags, context) {
  const { env, cwd, home, fetchImpl, out } = context;
  const baseUrl = getEffectiveBaseUrl(flags, env);
  const pkg = readPackageMeta();

  const targets = resolveInstallTargets({
    agents: flags.agent,
    project: flags.project,
    dir: flags.dir,
    cwd,
    home,
    env,
  });

  const skillInfo = await resolveSkillContent(baseUrl, { offline: flags.offline, fetchImpl });
  const results = writeSkillToTargets(targets, skillInfo.content, { dryRun: flags.dryRun });

  let npmLatest = null;
  let npmUpdateAvailable = false;
  if (!flags.offline && mode === 'update') {
    try {
      const latest = await fetchNpmLatest(getRegistryUrl(flags, env), { fetchImpl });
      npmLatest = latest.version;
      if (npmLatest && compareSemver(npmLatest, pkg.version) > 0) {
        npmUpdateAvailable = true;
      }
    } catch {
      // Non-fatal in offline/firewalled envs
    }
  }

  const payload = {
    mode,
    version: pkg.version,
    npmLatest,
    npmUpdateAvailable,
    baseUrl,
    skillSource: skillInfo.source,
    skillHash: skillInfo.hash,
    fallbackReason: skillInfo.fallbackReason,
    dryRun: flags.dryRun,
    targets: results.map((r) => ({
      path: r.skillFile,
      agents: r.agents,
      status: r.status,
    })),
  };

  if (flags.json) {
    out.write(`${JSON.stringify(payload, null, 2)}\n`);
    return 0;
  }

  out.write(`888-stock-quant v${pkg.version} — ${mode === 'update' ? 'Skill Update' : 'Skill Install'}\n`);
  out.write(`Source: ${skillInfo.source} (sha256:${skillInfo.hash}, base=${baseUrl})\n`);
  if (skillInfo.fallbackReason) {
    out.write(`Notice: Live /skill fetch skipped/failed (${skillInfo.fallbackReason}); used bundled SKILL.md.\n`);
  }
  if (npmUpdateAvailable) {
    out.write(`NPM CLI UPDATE AVAILABLE: v${pkg.version} -> v${npmLatest}. Run: npx -y 888-stock-quant@latest update\n`);
  }
  out.write('\nTargets:\n');
  for (const item of payload.targets) {
    out.write(`  - [${item.status}] ${item.path} (${item.agents.join(', ')})\n`);
  }
  out.write('\nReminder for LLM Agents:\n');
  out.write('  Before each session/task, check for updates via: npx -y 888-stock-quant@latest check-update\n');
  return 0;
}

async function handleCheckUpdate(flags, context) {
  const { env, cwd, home, fetchImpl, out } = context;
  const pkg = readPackageMeta();
  const baseUrl = getEffectiveBaseUrl(flags, env);
  const bundled = readBundledSkill();
  const bundledHash = sha256Short(bundled);

  let npmLatest = null;
  let npmError = null;
  let liveSkillHash = null;
  let liveError = null;

  if (!flags.offline) {
    try {
      const latest = await fetchNpmLatest(getRegistryUrl(flags, env), { fetchImpl });
      npmLatest = latest.version;
    } catch (err) {
      npmError = err.message;
    }
    try {
      const liveSkill = await fetchLiveSkill(baseUrl, { fetchImpl });
      liveSkillHash = sha256Short(liveSkill);
    } catch (err) {
      liveError = err.message;
    }
  }

  const targets = resolveInstallTargets({
    agents: flags.agent,
    project: flags.project,
    dir: flags.dir,
    cwd,
    home,
    env,
  });

  const installedTargets = targets.map((t) => {
    if (!fs.existsSync(t.skillFile)) {
      return { path: t.skillFile, agents: t.agents, installed: false, hash: null, stale: false };
    }
    const localHash = sha256Short(fs.readFileSync(t.skillFile, 'utf8'));
    const compareHash = liveSkillHash || bundledHash;
    return {
      path: t.skillFile,
      agents: t.agents,
      installed: true,
      hash: localHash,
      stale: localHash !== compareHash,
    };
  });

  const npmUpdateAvailable = Boolean(npmLatest && compareSemver(npmLatest, pkg.version) > 0);
  const skillUpdateAvailable = Boolean(
    (liveSkillHash && liveSkillHash !== bundledHash) || installedTargets.some((t) => t.stale)
  );
  const anyUpdateAvailable = npmUpdateAvailable || skillUpdateAvailable;

  const payload = {
    currentVersion: pkg.version,
    npmLatest,
    npmUpdateAvailable,
    npmError,
    bundledSkillHash: bundledHash,
    liveSkillHash,
    liveError,
    skillUpdateAvailable,
    anyUpdateAvailable,
    recommendedCommand: anyUpdateAvailable ? 'npx -y 888-stock-quant@latest update' : null,
    targets: installedTargets,
  };

  if (flags.json) {
    out.write(`${JSON.stringify(payload, null, 2)}\n`);
    return 0;
  }

  out.write(`888-stock-quant v${pkg.version} Update Check\n`);
  out.write(`  npm @latest   : ${npmLatest || (npmError ? `unavailable (${npmError})` : 'offline')}\n`);
  out.write(`  bundled skill : sha256:${bundledHash}\n`);
  out.write(`  live /skill   : ${liveSkillHash ? `sha256:${liveSkillHash}` : liveError ? `unavailable (${liveError})` : 'offline'}\n`);
  for (const t of installedTargets) {
    const state = !t.installed ? 'not-installed' : t.stale ? 'STALE' : 'up-to-date';
    out.write(`  target        : [${state}] ${t.path}\n`);
  }
  out.write(
    anyUpdateAvailable
      ? '\nUPDATE AVAILABLE -> Run: npx -y 888-stock-quant@latest update\n'
      : '\nAll checks up-to-date.\n'
  );
  return 0;
}

async function handleSkillCommand(flags, context) {
  const { env, fetchImpl, out } = context;
  const baseUrl = getEffectiveBaseUrl(flags, env);

  if (flags.llms || flags.llmsFull) {
    const content = await fetchLiveLlmsTxt(baseUrl, Boolean(flags.llmsFull), { fetchImpl });
    out.write(content.endsWith('\n') ? content : `${content}\n`);
    return 0;
  }

  const skillInfo = await resolveSkillContent(baseUrl, { offline: flags.offline, fetchImpl });
  if (flags.json) {
    out.write(`${JSON.stringify({ source: skillInfo.source, hash: skillInfo.hash, content: skillInfo.content }, null, 2)}\n`);
    return 0;
  }
  out.write(skillInfo.content.endsWith('\n') ? skillInfo.content : `${skillInfo.content}\n`);
  return 0;
}

function formatTableOrJson(data, flags, out, summaryBuilder) {
  if (flags.json) {
    out.write(`${JSON.stringify(data, null, 2)}\n`);
    return 0;
  }
  out.write(`${summaryBuilder(data)}\n`);
  return 0;
}

async function handleApiCommand(command, args, flags, context) {
  const { env, fetchImpl, out } = context;
  const baseUrl = getEffectiveBaseUrl(flags, env);
  const reqOpts = { fetchImpl };

  if (command === 'macro') {
    const data = await apiRequest(baseUrl, '/api/v1/macro/latest', reqOpts);
    return formatTableOrJson(data, flags, out, (d) =>
      JSON.stringify(d, null, 2)
    );
  }

  if (command === 'resonance') {
    const data = await apiRequest(baseUrl, '/api/v1/predictions/resonance', {
      ...reqOpts,
      query: { index_name: flags.index, limit: flags.limit },
    });
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'screen') {
    const data = await apiRequest(baseUrl, '/api/v1/predictions/screen', {
      ...reqOpts,
      query: {
        direction: flags.direction || 'bull',
        models: flags.models,
        pe_range: flags.pe,
        min_potential: flags.potential,
        risk_reward: flags.rr,
        ma60: flags.ma60,
        institutional: flags.inst,
        index_name: flags.index,
        limit: flags.limit || '20',
      },
    });
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'xuantie') {
    const data = await apiRequest(baseUrl, '/api/v1/predictions/xuantie', {
      ...reqOpts,
      query: { index_name: flags.index, limit: flags.limit },
    });
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'query' || command === 'history') {
    const ticker = args[0];
    if (!ticker) throw new Error(`Command "${command}" requires <ticker>`);
    const data = await apiRequest(baseUrl, `/api/v1/predictions/history/${encodeURIComponent(ticker)}`, {
      ...reqOpts,
      query: { limit: flags.limit },
    });
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'align') {
    const ticker = args[0];
    if (!ticker) throw new Error('Command "align" requires <ticker>');
    const data = await apiRequest(baseUrl, `/api/v1/predictions/alignment/${encodeURIComponent(ticker)}`, reqOpts);
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'resolve') {
    const q = args.join(' ');
    if (!q) throw new Error('Command "resolve" requires <query>');
    const data = await apiRequest(baseUrl, `/api/v1/predictions/resolve/${encodeURIComponent(q)}`, reqOpts);
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'xray') {
    if (args.length === 0) throw new Error('Command "xray" requires one or more <ticker> arguments');
    const tickers = args.flatMap((a) => a.split(',')).map((t) => t.trim()).filter(Boolean);
    const body = { tickers };
    if (flags.window) body.window = Number.parseInt(flags.window, 10);
    const data = await apiRequest(baseUrl, '/api/v1/risk/xray', {
      ...reqOpts,
      method: 'POST',
      body,
    });
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'sessions') {
    const data = await apiRequest(baseUrl, '/api/v1/market/calendar/sessions', reqOpts);
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  if (command === 'canary') {
    const data = await apiRequest(baseUrl, '/api/v1/stats/canary-health', reqOpts);
    return formatTableOrJson(data, flags, out, (d) => JSON.stringify(d, null, 2));
  }

  return null;
}

export async function runCli(
  argv,
  {
    env = process.env,
    cwd = process.cwd(),
    home = os.homedir(),
    fetchImpl = globalThis.fetch,
    out = process.stdout,
    err = process.stderr,
  } = {}
) {
  const pkg = readPackageMeta();
  let parsed;
  try {
    parsed = parseArgs(argv);
  } catch (e) {
    err.write(`Error: ${e.message}\n`);
    return 1;
  }

  const { positionals, flags } = parsed;
  if (flags.version) {
    out.write(`${pkg.version}\n`);
    return 0;
  }
  const command = (positionals[0] || 'install').toLowerCase();
  if (flags.help || command === 'help') {
    out.write(formatHelpText(pkg.version));
    return 0;
  }

  const context = { env, cwd, home, fetchImpl, out, err };

  try {
    if (command === 'install' || command === 'update') {
      return await handleInstallOrUpdate(command, flags, context);
    }
    if (command === 'check-update') {
      return await handleCheckUpdate(flags, context);
    }
    if (command === 'skill' || command === 'docs') {
      return await handleSkillCommand(flags, context);
    }
    const apiResult = await handleApiCommand(command, positionals.slice(1), flags, context);
    if (apiResult !== null) return apiResult;

    err.write(`Unknown command "${command}". Run "888-stock-quant --help" for usage.\n`);
    return 1;
  } catch (e) {
    err.write(`Error: ${e.message}\n`);
    return 1;
  }
}
