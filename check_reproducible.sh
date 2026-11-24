#!/usr/bin/env bash
# From-scratch reproducibility check: snapshot the generated files, delete them, re-run run_all.sh, compare byte for byte.
set -euo pipefail
cd "$(dirname "$0")"
S=.repro_snapshot; rm -rf "$S"; mkdir -p "$S/results" "$S/docs"
cp results/*.csv results/*.md results/*.png "$S/results/"; cp docs/report.md docs/viva_qa.md docs/numbers_provenance.md "$S/docs/"; cp README.md "$S/README.md"
rm -f results/*.csv results/*.md results/*.png results/logs/*
bash run_all.sh
same=0; diff=0
for f in "$S"/results/*; do b=$(basename "$f"); if cmp -s "$f" "results/$b"; then same=$((same+1)); else diff=$((diff+1)); echo "DIFFERENT: results/$b"; fi; done
for f in report.md viva_qa.md numbers_provenance.md; do if cmp -s "$S/docs/$f" "docs/$f"; then same=$((same+1)); else diff=$((diff+1)); echo "DIFFERENT: docs/$f"; fi; done
if cmp -s "$S/README.md" README.md; then same=$((same+1)); else diff=$((diff+1)); echo "DIFFERENT: README.md"; fi
echo "byte-identical files: $same, different files: $diff"; rm -rf "$S"; [ "$diff" -eq 0 ]
