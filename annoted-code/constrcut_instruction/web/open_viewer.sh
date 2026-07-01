#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8765}"

echo "Building viewer bundle..."
python3 build_share_bundle.py --skip-zip

echo
echo "Viewer ready:"
echo "  file://$(pwd)/share_bundle/index.html"
echo
echo "Starting HTTP server on http://127.0.0.1:${PORT}/share_bundle/index.html"
echo "Press Ctrl+C to stop."
python3 -m http.server "$PORT"
