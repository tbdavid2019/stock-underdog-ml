import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, test } from 'node:test';
import { runCli } from '../lib/commands.js';
import { readBundledSkill } from '../lib/client.js';

const tmpDirs = [];

function makeTmpDir(prefix = '888sq-test-') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  tmpDirs.push(dir);
  return dir;
}

function makeIo() {
  let stdout = '';
  let stderr = '';
  return {
    out: {
      write(chunk) {
        stdout += String(chunk);
      },
    },
    err: {
      write(chunk) {
        stderr += String(chunk);
      },
    },
    getStdout: () => stdout,
    getStderr: () => stderr,
  };
}

afterEach(() => {
  while (tmpDirs.length > 0) {
    const dir = tmpDirs.pop();
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('install --agent all installs stock-quant SKILL.md into global home dirs with deduplication', async () => {
  const home = makeTmpDir('home-');
  const cwd = makeTmpDir('cwd-');
  const io = makeIo();

  const exit = await runCli(['install', '--agent', 'all', '--offline', '--json'], {
    home,
    cwd,
    env: { HERMES_HOME: path.join(home, 'custom-hermes') },
    out: io.out,
    err: io.err,
  });

  assert.equal(exit, 0, io.getStderr());
  const result = JSON.parse(io.getStdout());
  assert.equal(result.skillSource, 'bundled');

  const expectedPaths = [
    path.join(home, '.gemini', 'skills', 'stock-quant', 'SKILL.md'),
    path.join(home, '.codex', 'skills', 'stock-quant', 'SKILL.md'),
    path.join(home, '.config', 'opencode', 'skills', 'stock-quant', 'SKILL.md'),
    path.join(home, '.claude', 'skills', 'stock-quant', 'SKILL.md'),
    path.join(home, 'custom-hermes', 'skills', 'stock-quant', 'SKILL.md'),
    path.join(home, '.openclaw', 'workspace', 'skills', 'stock-quant', 'SKILL.md'),
    path.join(home, '.agents', 'skills', 'stock-quant', 'SKILL.md'),
  ];

  for (const filePath of expectedPaths) {
    assert.ok(fs.existsSync(filePath), `Expected installed SKILL.md at ${filePath}`);
    assert.equal(fs.readFileSync(filePath, 'utf8'), readBundledSkill());
  }

  // omp and agents both resolve to ~/.agents/skills/stock-quant/SKILL.md and should be deduplicated
  const targetPaths = result.targets.map((t) => t.path);
  assert.equal(new Set(targetPaths).size, targetPaths.length);
  const sharedEntry = result.targets.find((t) => t.path.endsWith(path.join('.agents', 'skills', 'stock-quant', 'SKILL.md')));
  assert.ok(sharedEntry.agents.includes('omp'));
  assert.ok(sharedEntry.agents.includes('agents'));
});

test('install --project installs into project directories and supports --dir override', async () => {
  const home = makeTmpDir('home-');
  const cwd = makeTmpDir('proj-');
  const io = makeIo();

  const exitProj = await runCli(['install', '--agent', 'codex,opencode,openclaw', '--project', '--offline', '--json'], {
    home,
    cwd,
    env: {},
    out: io.out,
    err: io.err,
  });
  assert.equal(exitProj, 0, io.getStderr());

  assert.ok(fs.existsSync(path.join(cwd, '.codex', 'skills', 'stock-quant', 'SKILL.md')));
  assert.ok(fs.existsSync(path.join(cwd, '.opencode', 'skills', 'stock-quant', 'SKILL.md')));
  assert.ok(fs.existsSync(path.join(cwd, 'skills', 'stock-quant', 'SKILL.md')));

  const customDir = path.join(cwd, 'my-custom-skills');
  const ioCustom = makeIo();
  const exitCustom = await runCli(['install', '--dir', customDir, '--offline', '--dry-run', '--json'], {
    home,
    cwd,
    env: {},
    out: ioCustom.out,
    err: ioCustom.err,
  });
  assert.equal(exitCustom, 0);
  const customJson = JSON.parse(ioCustom.getStdout());
  assert.equal(customJson.dryRun, true);
  assert.equal(customJson.targets[0].path, path.join(customDir, 'stock-quant', 'SKILL.md'));
  assert.equal(fs.existsSync(path.join(customDir, 'stock-quant', 'SKILL.md')), false);
});

test('install rejects HTML challenge responses from /skill and safely falls back to bundled SKILL.md', async () => {
  const home = makeTmpDir('home-');
  const cwd = makeTmpDir('cwd-');
  const io = makeIo();

  const htmlChallengeFetch = async () =>
    new Response('<!DOCTYPE html><html><body>Just a moment...</body></html>', {
      status: 200,
      headers: { 'content-type': 'text/html; charset=utf-8' },
    });

  const exit = await runCli(['install', '--agent', 'codex', '--json'], {
    home,
    cwd,
    env: {},
    fetchImpl: htmlChallengeFetch,
    out: io.out,
    err: io.err,
  });

  assert.equal(exit, 0, io.getStderr());
  const parsed = JSON.parse(io.getStdout());
  assert.equal(parsed.skillSource, 'bundled');
  assert.match(parsed.fallbackReason, /text\/html/);
  assert.equal(
    fs.readFileSync(path.join(home, '.codex', 'skills', 'stock-quant', 'SKILL.md'), 'utf8'),
    readBundledSkill()
  );
});

test('check-update and update detect newer npm version and live /skill changes', async () => {
  const home = makeTmpDir('home-');
  const cwd = makeTmpDir('cwd-');
  const ioInstall = makeIo();

  await runCli(['install', '--agent', 'codex', '--offline'], {
    home,
    cwd,
    env: {},
    out: ioInstall.out,
    err: ioInstall.err,
  });

  const liveUpdatedSkill = `${readBundledSkill()}\n<!-- live-v2.7.0-marker -->\n`;
  const mockFetch = async (url) => {
    const href = String(url);
    if (href.includes('/skill')) {
      return new Response(liveUpdatedSkill, {
        status: 200,
        headers: { 'content-type': 'text/markdown; charset=utf-8' },
      });
    }
    if (href.includes('registry.npmjs.org')) {
      return new Response(JSON.stringify({ version: '9.9.9' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      });
    }
    throw new Error(`Unexpected URL ${href}`);
  };

  const ioCheck = makeIo();
  const checkExit = await runCli(['check-update', '--agent', 'codex', '--json'], {
    home,
    cwd,
    env: {},
    fetchImpl: mockFetch,
    out: ioCheck.out,
    err: ioCheck.err,
  });
  assert.equal(checkExit, 0, ioCheck.getStderr());
  const checkJson = JSON.parse(ioCheck.getStdout());
  assert.equal(checkJson.npmLatest, '9.9.9');
  assert.equal(checkJson.npmUpdateAvailable, true);
  assert.equal(checkJson.skillUpdateAvailable, true);
  assert.equal(checkJson.anyUpdateAvailable, true);
  assert.equal(checkJson.recommendedCommand, 'npx -y 888-stock-quant@latest update');
  assert.equal(checkJson.targets[0].stale, true);

  const ioUpdate = makeIo();
  const updateExit = await runCli(['update', '--agent', 'codex', '--json'], {
    home,
    cwd,
    env: {},
    fetchImpl: mockFetch,
    out: ioUpdate.out,
    err: ioUpdate.err,
  });
  assert.equal(updateExit, 0, ioUpdate.getStderr());
  const updateJson = JSON.parse(ioUpdate.getStdout());
  assert.equal(updateJson.skillSource, 'live');
  assert.equal(updateJson.targets[0].status, 'updated');
  assert.equal(
    fs.readFileSync(path.join(home, '.codex', 'skills', 'stock-quant', 'SKILL.md'), 'utf8'),
    liveUpdatedSkill
  );
});

test('quant API subcommands call expected REST endpoints and reject unsafe base URLs', async () => {
  const home = makeTmpDir('home-');
  const cwd = makeTmpDir('cwd-');
  const seen = [];

  const mockFetch = async (url, init = {}) => {
    seen.push({ url: String(url), method: init.method || 'GET', body: init.body || null });
    return new Response(JSON.stringify({ ok: true, url: String(url) }), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
  };

  const cases = [
    ['macro', '--json'],
    ['resonance', '--index', 'TW_TOP50', '--limit', '5', '--json'],
    ['screen', '--direction', 'bull', '--models', '2', '--limit', '10', '--json'],
    ['query', '2330.TW', '--limit', '15', '--json'],
    ['align', '2330.TW', '--json'],
    ['xray', '2330.TW', 'NVDA', '--window', '120', '--json'],
    ['sessions', '--json'],
    ['canary', '--json'],
  ];

  for (const args of cases) {
    const io = makeIo();
    const exit = await runCli(args, {
      home,
      cwd,
      env: { STOCK_QUANT_BASE_URL: 'https://example.quant.local' },
      fetchImpl: mockFetch,
      out: io.out,
      err: io.err,
    });
    assert.equal(exit, 0, `Command ${args.join(' ')} failed: ${io.getStderr()}`);
  }

  assert.equal(seen[0].url, 'https://example.quant.local/api/v1/macro/latest');
  assert.equal(seen[1].url, 'https://example.quant.local/api/v1/predictions/resonance?index_name=TW_TOP50&limit=5');
  assert.equal(
    seen[2].url,
    'https://example.quant.local/api/v1/predictions/screen?direction=bull&models=2&limit=10'
  );
  assert.equal(seen[3].url, 'https://example.quant.local/api/v1/predictions/history/2330.TW?limit=15');
  assert.equal(seen[4].url, 'https://example.quant.local/api/v1/predictions/alignment/2330.TW');
  assert.equal(seen[5].method, 'POST');
  assert.equal(seen[5].url, 'https://example.quant.local/api/v1/risk/xray');
  assert.deepEqual(JSON.parse(seen[5].body), { tickers: ['2330.TW', 'NVDA'], window: 120 });
  assert.equal(seen[6].url, 'https://example.quant.local/api/v1/market/calendar/sessions');
  assert.equal(seen[7].url, 'https://example.quant.local/api/v1/stats/canary-health');

  // Security check: disallow file:// URLs
  const badIo = makeIo();
  const badExit = await runCli(['macro', '--base-url', 'file:///etc/passwd'], {
    home,
    cwd,
    env: {},
    fetchImpl: mockFetch,
    out: badIo.out,
    err: badIo.err,
  });
  assert.equal(badExit, 1);
  assert.match(badIo.getStderr(), /Only http: and https: are allowed/);
});
