import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const SKILL_FOLDER_NAME = 'stock-quant';

export const AGENT_TARGETS = {
  agy: {
    label: 'Antigravity (AGY)',
    detectGlobalRoot: (home) => path.join(home, '.gemini'),
    globalSkillsDir: (home) => path.join(home, '.gemini', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.gemini', 'skills'),
  },
  codex: {
    label: 'Codex CLI',
    detectGlobalRoot: (home) => path.join(home, '.codex'),
    globalSkillsDir: (home) => path.join(home, '.codex', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.codex', 'skills'),
  },
  opencode: {
    label: 'OpenCode',
    detectGlobalRoot: (home) => path.join(home, '.config', 'opencode'),
    globalSkillsDir: (home) => path.join(home, '.config', 'opencode', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.opencode', 'skills'),
  },
  claude: {
    label: 'Claude Code',
    detectGlobalRoot: (home) => path.join(home, '.claude'),
    globalSkillsDir: (home) => path.join(home, '.claude', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.claude', 'skills'),
  },
  hermes: {
    label: 'Hermes Agent',
    detectGlobalRoot: (home, env) => env.HERMES_HOME || path.join(home, '.hermes'),
    globalSkillsDir: (home, env) => path.join(env.HERMES_HOME || path.join(home, '.hermes'), 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.hermes', 'skills'),
  },
  openclaw: {
    label: 'OpenClaw',
    detectGlobalRoot: (home) => path.join(home, '.openclaw'),
    globalSkillsDir: (home) => path.join(home, '.openclaw', 'workspace', 'skills'),
    altGlobalSkillsDir: (home) => path.join(home, '.openclaw', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, 'skills'),
  },
  omp: {
    label: 'OMP (oh-my-pi)',
    detectGlobalRoot: (home) => path.join(home, '.omp'),
    globalSkillsDir: (home) => path.join(home, '.agents', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.agents', 'skills'),
  },
  agents: {
    label: 'Cross-Agent (.agents/skills)',
    detectGlobalRoot: (home) => path.join(home, '.agents'),
    globalSkillsDir: (home) => path.join(home, '.agents', 'skills'),
    projectSkillsDir: (cwd) => path.join(cwd, '.agents', 'skills'),
  },
};

export function listSupportedAgentKeys() {
  return Object.keys(AGENT_TARGETS);
}

export function normalizeAgentSelection(agentInputs) {
  if (!agentInputs || agentInputs.length === 0) return [];
  const expanded = agentInputs
    .flatMap((entry) => String(entry).split(','))
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);

  if (expanded.includes('all')) {
    return listSupportedAgentKeys();
  }

  for (const key of expanded) {
    if (!AGENT_TARGETS[key]) {
      throw new Error(
        `Unknown agent "${key}". Supported agents: ${listSupportedAgentKeys().join(', ')}, all`
      );
    }
  }
  return [...new Set(expanded)];
}

function addTarget(targetMap, skillsBaseDir, agentKey, label) {
  const skillFile = path.join(skillsBaseDir, SKILL_FOLDER_NAME, 'SKILL.md');
  const resolved = path.resolve(skillFile);
  if (!targetMap.has(resolved)) {
    targetMap.set(resolved, {
      skillFile: resolved,
      skillDir: path.dirname(resolved),
      agents: [],
      labels: [],
    });
  }
  const entry = targetMap.get(resolved);
  if (!entry.agents.includes(agentKey)) entry.agents.push(agentKey);
  if (!entry.labels.includes(label)) entry.labels.push(label);
}

/**
 * Resolve destination SKILL.md paths across supported coding agents.
 */
export function resolveInstallTargets({
  agents = [],
  project = false,
  dir = null,
  cwd = process.cwd(),
  home = os.homedir(),
  env = process.env,
} = {}) {
  const targetMap = new Map();

  if (dir) {
    const base = path.resolve(cwd, dir);
    const skillsBase = path.basename(base) === SKILL_FOLDER_NAME ? path.dirname(base) : base;
    addTarget(targetMap, skillsBase, 'custom', `Custom (${base})`);
    return [...targetMap.values()];
  }

  const selected = normalizeAgentSelection(agents);
  if (selected.length > 0) {
    for (const key of selected) {
      const def = AGENT_TARGETS[key];
      if (project) {
        addTarget(targetMap, def.projectSkillsDir(cwd), key, def.label);
      } else {
        addTarget(targetMap, def.globalSkillsDir(home, env), key, def.label);
        if (def.altGlobalSkillsDir) {
          const altDir = def.altGlobalSkillsDir(home, env);
          if (fs.existsSync(altDir)) {
            addTarget(targetMap, altDir, key, `${def.label} (legacy ~/.openclaw/skills)`);
          }
        }
      }
    }
    return [...targetMap.values()];
  }

  if (project) {
    // Auto-detect project agent folders, defaulting to standard .agents/skills
    let matched = false;
    for (const [key, def] of Object.entries(AGENT_TARGETS)) {
      if (key === 'openclaw') continue; // avoid creating generic ./skills unless requested
      const projectSkills = def.projectSkillsDir(cwd);
      const projectRoot = path.dirname(projectSkills);
      if (fs.existsSync(projectRoot)) {
        addTarget(targetMap, projectSkills, key, def.label);
        matched = true;
      }
    }
    if (!matched) {
      addTarget(targetMap, AGENT_TARGETS.agents.projectSkillsDir(cwd), 'agents', AGENT_TARGETS.agents.label);
    }
    return [...targetMap.values()];
  }

  // Global auto-detection
  for (const [key, def] of Object.entries(AGENT_TARGETS)) {
    const rootDir = def.detectGlobalRoot(home, env);
    if (fs.existsSync(rootDir)) {
      addTarget(targetMap, def.globalSkillsDir(home, env), key, def.label);
      if (def.altGlobalSkillsDir) {
        const altDir = def.altGlobalSkillsDir(home, env);
        if (fs.existsSync(altDir)) {
          addTarget(targetMap, altDir, key, `${def.label} (legacy ~/.openclaw/skills)`);
        }
      }
    }
  }

  // Always include ~/.agents/skills as the cross-agent canonical fallback
  addTarget(targetMap, AGENT_TARGETS.agents.globalSkillsDir(home, env), 'agents', AGENT_TARGETS.agents.label);
  return [...targetMap.values()];
}

export function writeSkillToTargets(targets, content, { dryRun = false } = {}) {
  const results = [];
  for (const target of targets) {
    const existed = fs.existsSync(target.skillFile);
    const prev = existed ? fs.readFileSync(target.skillFile, 'utf8') : null;
    const unchanged = prev === content;

    if (!dryRun && !unchanged) {
      fs.mkdirSync(target.skillDir, { recursive: true });
      fs.writeFileSync(target.skillFile, content, 'utf8');
    }

    results.push({
      ...target,
      status: unchanged ? 'unchanged' : existed ? 'updated' : 'installed',
      written: !dryRun && !unchanged,
    });
  }
  return results;
}
