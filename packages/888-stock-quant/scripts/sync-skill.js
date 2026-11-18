#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(pkgRoot, '..', '..');

const sourceSkill = path.join(repoRoot, 'skills', 'stock-quant', 'SKILL.md');
const destSkill = path.join(pkgRoot, 'skills', 'stock-quant', 'SKILL.md');
const rootLicense = path.join(repoRoot, 'LICENSE');
const destLicense = path.join(pkgRoot, 'LICENSE');

if (!fs.existsSync(sourceSkill)) {
  console.error(`Source SKILL.md not found at ${sourceSkill}`);
  process.exit(1);
}

fs.mkdirSync(path.dirname(destSkill), { recursive: true });
fs.copyFileSync(sourceSkill, destSkill);
console.log(`Synced SKILL.md -> ${path.relative(repoRoot, destSkill)}`);

if (fs.existsSync(rootLicense)) {
  fs.copyFileSync(rootLicense, destLicense);
  console.log(`Synced LICENSE -> ${path.relative(repoRoot, destLicense)}`);
}
