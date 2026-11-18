#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Building precompiled Tailwind CSS -> api/static/css/app.css ..."
npx --yes tailwindcss@3.4.17 \
  -c ./tailwind.config.js \
  -i ./api/static/css/input.css \
  -o ./api/static/css/app.css \
  --minify

ls -lh ./api/static/css/app.css
