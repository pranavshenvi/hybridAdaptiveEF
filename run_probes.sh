#!/usr/bin/env bash
# Probe-length sweep (updateAsOf021126.md): is PercEF's edge over Ada-ef its thresholds or its shorter
# probe? Both scores at probe lengths 50, 100, 300 and 1025 (each Ada-ef length with its own group
# table), setting R, on the 17 same-modality datasets. Smallest datasets first, LAION last.
#   tmux new -s probes   then   bash run_probes.sh
#   DATASETS="sift128 glove100" LENGTHS="100,1025" bash run_probes.sh     (a subset)
# Results: results_unified_<ds>_ablation_probes_<ts>; reuses the headline caches (index, ground truth,
# bins, min-ef, Ada-ef's 1025 table); new Ada-ef tables cached as ada2_table_R_L<L>*.json.
# Summary: analysis/summarize_probes.py . (writes probe_summary.csv). Log: probes_<timestamp>.log.
set -u
cd "$(dirname "$0")"
LOG=probes_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

LENGTHS=${LENGTHS:-"50,100,300,1025"}
DATASETS=${DATASETS:-"vibe_yahoo_minilm vibe_landmark_dino vibe_inaturalist_resnet gist960 fashionmnist784
  coco_i2i yambda sift128 deep1b bigann deepimage96 dbpedia1536 cohere1024 msturing glove100 msmarco384 laion_i2i"}

if pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; then
  echo "[$(date)] another job is running; waiting for it to finish"
  while pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; do sleep 60; done
fi

echo "[$(date)] smoke test of --probe-lengths"
if ! python3 benchmark_unified.py --dataset fashionmnist784 --smoke --quick --settings R --lat-rounds 1 \
     --probe-lengths "$LENGTHS" > probes_smoke.log 2>&1; then
  echo "SMOKE TEST FAILED: not starting. Last lines of probes_smoke.log:"; tail -n 30 probes_smoke.log; exit 1
fi
echo "[$(date)] smoke test passed"

for d in $DATASETS; do
  echo "[$(date)] $d: probe lengths $LENGTHS"
  if python3 benchmark_unified.py --dataset "$d" --quick --settings R --probe-lengths "$LENGTHS" 2>&1 | grep -A22 "SCORECARD"; then
    echo "[$(date)] $d done"
  else
    echo "[$(date)] $d FAILED (see the newest results_unified_${d}_ablation_probes_*/benchmark_unified.log)"
  fi
  python3 analysis/summarize_probes.py . | sed -n '/^PercEF vs Ada-ef/,/^$/p'
done
echo "[$(date)] done (log: $LOG). Copy back: results_unified_*_ablation_probes_*  probe_summary.csv  $LOG"
