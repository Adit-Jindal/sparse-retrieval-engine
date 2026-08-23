#!/usr/bin/env bash
# Run the same checks CI runs, locally, before you push.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Interface conformance tests =="
pytest tests/test_interface_conformance.py -v

echo
echo "== Metrics unit tests =="
pytest tests/test_metrics.py -v

echo
echo "== Full harness run on the toy set =="
python -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/reference_bm25_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json

echo
echo "All smoke checks passed."

# python -m harness.run_harness \
#   --corpus data/nfcorpus/corpus.jsonl \
#   --queries data/nfcorpus/queries_dev.tsv \
#   --qrels data/nfcorpus/qrels_test.txt \
#   --baseline-run data/toy/reference_bm25_run_dev.trec \
#   --run-out runs/dev_run.trec \
#   --report-out runs/dev_report.json