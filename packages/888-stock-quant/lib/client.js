import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const DEFAULT_BASE_URL = 'https://stockdata.david888.com';
export const DEFAULT_NPM_REGISTRY_URL = 'https://registry.npmjs.org/888-stock-quant/latest';
const DEFAULT_TIMEOUT_MS = 12000;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const PACKAGE_ROOT = path.resolve(__dirname, '..');
export const BUNDLED_SKILL_PATH = path.join(PACKAGE_ROOT, 'skills', 'stock-quant', 'SKILL.md');
export const PACKAGE_JSON_PATH = path.join(PACKAGE_ROOT, 'package.json');

export function readPackageMeta() {
  return JSON.parse(fs.readFileSync(PACKAGE_JSON_PATH, 'utf8'));
}

export function readBundledSkill() {
  return fs.readFileSync(BUNDLED_SKILL_PATH, 'utf8');
}

export function sha256Short(text) {
  return crypto.createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 12);
}

/**
 * Validate base URL strictly allows only http/https schemes.
 */
export function validateBaseUrl(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error(`Invalid base URL: "${rawUrl}"`);
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new Error(`Unsupported URL protocol "${parsed.protocol}". Only http: and https: are allowed.`);
  }
  return parsed.toString().replace(/\/+$/, '');
}

export function isValidSkillMarkdown(text) {
  if (typeof text !== 'string') return false;
  const trimmed = text.trimStart();
  if (!trimmed.startsWith('---')) return false;
  if (/^\s*<(!doctype|html)/i.test(trimmed)) return false;
  return /^name:\s*stock-quant\b/m.test(trimmed);
}

export function compareSemver(a, b) {
  const parse = (v) =>
    String(v || '0.0.0')
      .replace(/^v/i, '')
      .split('-')[0]
      .split('.')
      .map((n) => Number.parseInt(n, 10) || 0);
  const pa = parse(a);
  const pb = parse(b);
  const len = Math.max(pa.length, pb.length, 3);
  for (let i = 0; i < len; i += 1) {
    const va = pa[i] || 0;
    const vb = pb[i] || 0;
    if (va > vb) return 1;
    if (va < vb) return -1;
  }
  return 0;
}

export async function fetchWithTimeout(
  url,
  options = {},
  { timeoutMs = DEFAULT_TIMEOUT_MS, fetchImpl = globalThis.fetch } = {}
) {
  if (typeof fetchImpl !== 'function') {
    throw new Error('Global fetch is unavailable. Node >= 18 is required.');
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetchImpl(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchLiveSkill(baseUrl, opts = {}) {
  const normalizedBase = validateBaseUrl(baseUrl);
  const res = await fetchWithTimeout(
    `${normalizedBase}/skill`,
    {
      method: 'GET',
      headers: { Accept: 'text/markdown, text/plain;q=0.9' },
    },
    opts
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch /skill (HTTP ${res.status})`);
  }
  const contentType = res.headers.get('content-type') || '';
  if (contentType.toLowerCase().includes('text/html')) {
    throw new Error(`Unexpected content-type "${contentType}" from /skill`);
  }
  const body = await res.text();
  if (!isValidSkillMarkdown(body)) {
    throw new Error('Remote /skill response failed SKILL.md frontmatter validation');
  }
  return body;
}

export async function fetchLiveLlmsTxt(baseUrl, full = false, opts = {}) {
  const normalizedBase = validateBaseUrl(baseUrl);
  const endpoint = full ? '/llms-full.txt' : '/llms.txt';
  const res = await fetchWithTimeout(
    `${normalizedBase}${endpoint}`,
    {
      method: 'GET',
      headers: { Accept: 'text/plain, text/markdown;q=0.9' },
    },
    opts
  );
  if (!res.ok) {
    throw new Error(`Failed to fetch ${endpoint} (HTTP ${res.status})`);
  }
  return res.text();
}

export async function fetchNpmLatest(registryUrl = DEFAULT_NPM_REGISTRY_URL, opts = {}) {
  const normalizedUrl = validateBaseUrl(registryUrl);
  const res = await fetchWithTimeout(
    normalizedUrl,
    {
      method: 'GET',
      headers: { Accept: 'application/json' },
    },
    opts
  );
  if (!res.ok) {
    throw new Error(`npm registry returned HTTP ${res.status}`);
  }
  const data = await res.json();
  return {
    version: data?.version || null,
    distTag: 'latest',
  };
}

export async function apiRequest(
  baseUrl,
  pathname,
  { method = 'GET', query = null, body = null, timeoutMs = DEFAULT_TIMEOUT_MS, fetchImpl = globalThis.fetch } = {}
) {
  const normalizedBase = validateBaseUrl(baseUrl);
  const url = new URL(`${normalizedBase}${pathname}`);
  if (query && typeof query === 'object') {
    for (const [key, val] of Object.entries(query)) {
      if (val !== undefined && val !== null && val !== '') {
        url.searchParams.set(key, String(val));
      }
    }
  }
  const headers = { Accept: 'application/json' };
  const init = { method, headers };
  if (body !== null && body !== undefined) {
    headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const res = await fetchWithTimeout(url.toString(), init, { timeoutMs, fetchImpl });
  const text = await res.text();
  let parsed;
  try {
    parsed = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`API ${pathname} returned non-JSON (HTTP ${res.status})`);
  }
  if (!res.ok) {
    const detail = parsed?.detail || parsed?.message || res.statusText || 'Request failed';
    throw new Error(`API ${pathname} HTTP ${res.status}: ${detail}`);
  }
  return parsed;
}
