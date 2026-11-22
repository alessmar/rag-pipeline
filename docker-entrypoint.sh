#!/bin/sh
# First start: create the single default company so the app works without editing any file.
set -e
f="${RAG_TENANTS_FILE:-tenants.json}"
if [ ! -f "$f" ]; then
  mkdir -p "$(dirname "$f")"
  printf '{"%s": "default"}\n' "${RAG_DEFAULT_TENANT:?set RAG_DEFAULT_TENANT in .env}" > "$f"
fi
exec "$@"
