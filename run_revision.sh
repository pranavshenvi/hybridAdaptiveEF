#!/usr/bin/env bash
# Revision queue (updateAsOf281026.md): everything that has to run on the server after the Ada-ef
# quantile/units fix, plus the experiments the reviewers' list asks for. One job at a time (latency
# is measured), phase by phase, so a partial run is already usable. A failing run is reported and
# skipped; nothing already on disk is deleted.
#
#   Phase 0  rebuild the C++ extension (the fix is in C++) and smoke-test every new flag
#   Phase 1  headline runs again, every dataset, P and R, with PiP added (--pip): every earlier
#            Ada-ef number used the wrong thresholds. Reuses ground truth, bins, min-ef caches;
#            rebuilds only Ada-ef's tables (cached as ada2_table_*).
#   Phase 2  the runs that compared against Ada-ef: probe-length ablation + group table, the
#            robustness sweep (run_sweep.sh), seed repeats (run_seeds.sh), controlled
#            synthetic experiment (run_controlled_experiments.sh, Figure 4)
#   Phase 3  HNSW M sweep (M = 8, 32)
#   Phase 4  native L2 on the L2 datasets (raw vectors; Ada-ef not applicable)
#   Phase 5  other graph indexes: FAISS HNSW, FAISS NSG, DiskANN Vamana (needs faiss-cpu, diskannpy)
#   Phase 6  learned baselines: DARTH + LAET on more datasets; DARTH summaries redone against the
#            phase-1 runs (no retraining needed: DARTH's own runs did not use Ada-ef)
#
#   tmux new -s revision   then   bash run_revision.sh            (all phases)
#   PHASES="1 3" bash run_revision.sh                              (some phases)
# Once, beforehand: pip install --user faiss-cpu diskannpy   (phase 5; a missing one is skipped)
# Log: revision_<timestamp>.log. Copy back at the end: the command printed on the last line.
set -u
cd "$(dirname "$0")"
LOG=revision_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
touch .revision_start                         # copy-back: everything newer than this

PHASES=${PHASES:-"0 1 2 3 4 5 6"}
# Phase 1: non-Gaussian side first (the paper's main claims), then the near-Gaussian side, then the
# two largest corpora last (each takes most of a day)
HEADLINE=${HEADLINE:-"sift128 deep1b deepimage96 yambda vibe_landmark_dino vibe_inaturalist_resnet gist960
  fashionmnist784 lastfm64 glove100 msturing bigann coco_i2i coco_t2i dbpedia1536 vibe_yahoo_minilm
  vibe_imagenet_align msmarco384 cohere1024 laion_i2i"}
ABLATION=${ABLATION:-"sift128 deep1b deepimage96 bigann glove100 msturing"}
M_SWEEP=${M_SWEEP:-"8 32"}
M_DATASETS=${M_DATASETS:-"sift128 deep1b glove100 msturing"}
L2_DATASETS=${L2_DATASETS:-"sift128 gist960 fashionmnist784 bigann deep1b msturing"}
GRAPHS=${GRAPHS:-"faiss_hnsw nsg vamana"}
GRAPH_DATASETS=${GRAPH_DATASETS:-"sift128 deep1b glove100 msturing"}
DARTH_NEW=${DARTH_NEW:-"gist960 fashionmnist784 bigann yambda vibe_inaturalist_resnet coco_i2i vibe_yahoo_minilm"}

has() { case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

run() {   # run <label> <args...>: one benchmark, scorecard to the log, failure reported
  local label=$1; shift
  echo "[$(date)] $label: benchmark_unified.py $*"
  if python3 benchmark_unified.py "$@" 2>&1 | grep -A14 "SCORECARD"; then
    echo "[$(date)] $label done"
  else
    echo "[$(date)] $label FAILED (see the newest results_unified_$2_*/benchmark_unified.log)"
  fi
}

if pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; then
  echo "[$(date)] another job is running; waiting for it to finish"
  while pgrep -f "benchmark_unified.py|hnsw_test" > /dev/null; do sleep 60; done
fi

if has 0; then
  echo "[$(date)] phase 0: rebuilding the extension"
  ( cd chao_hybrid_ada_ef && rm -rf build chao_hybrid_ada_ef_cpp*.so && python3 setup.py build_ext --inplace > ../revision_build.log 2>&1 ) \
    || { echo "BUILD FAILED: see revision_build.log"; exit 1; }
  python3 test_normal_quantile.py || { echo "QUANTILE TEST FAILED: not starting"; exit 1; }
  for a in "--pip" "--metric=l2 --pip" "--M=8"; do
    # shellcheck disable=SC2086
    if ! python3 benchmark_unified.py --dataset fashionmnist784 --smoke --quick --settings R --lat-rounds 1 $a > revision_smoke.log 2>&1; then
      echo "SMOKE TEST FAILED for '$a': not starting. Last lines of revision_smoke.log:"; tail -n 30 revision_smoke.log; exit 1
    fi
  done
  echo "[$(date)] smoke tests passed"
fi

if has 1; then
  for d in $HEADLINE; do run "headline $d" --dataset "$d" --pip; done
fi

if has 2; then
  for d in $ABLATION; do run "ablation $d" --dataset "$d" --ablation --group-table --quick --settings R; done
  bash run_sweep.sh
  bash run_seeds.sh
  bash run_controlled_experiments.sh > "controlled_$(date +%Y%m%d_%H%M%S).log" 2>&1     && echo "[$(date)] controlled experiments done" || echo "[$(date)] controlled experiments FAILED"
fi

if has 3; then
  for m in $M_SWEEP; do
    for d in $M_DATASETS; do run "M=$m $d" --dataset "$d" --M "$m" --quick; done
  done
fi

if has 4; then
  for d in $L2_DATASETS; do run "native L2 $d" --dataset "$d" --metric l2 --quick --pip; done
fi

if has 5; then
  for g in $GRAPHS; do
    mod=$([ "$g" = vamana ] && echo diskannpy || echo faiss)
    if ! python3 -c "import $mod" 2>/dev/null; then
      echo "[$(date)] $g skipped: python module $mod missing (pip install --user ${mod/faiss/faiss-cpu})"; continue
    fi
    for d in $GRAPH_DATASETS; do run "$g $d" --dataset "$d" --graph "$g" --quick; done
  done
fi

if has 6; then
  for dir in results_darth_*/; do                  # summaries of the earlier DARTH runs vs the new runs
    ds=$(python3 -c "import json,sys;print(json.load(open('${dir}meta.json'))['dataset'])" 2>/dev/null \
         || basename "$dir" | sed -E 's/^results_darth_(.+)_[0-9]{8}_[0-9]{6}$/\1/')
    ours=$(ls -d results_unified_"${ds}"_2* 2>/dev/null | sort | tail -1)
    [ -n "$ours" ] && python3 darth/darth_summarize.py "${dir%/}" "$ours" > "${dir}summary_revision.txt" 2>&1 \
      && echo "[$(date)] DARTH summary redone: ${dir} vs $ours"
  done
  NEW="$DARTH_NEW" LAET_ONLY=" " bash run_baselines.sh      # " ": no LAET-only runs (empty = default list)
fi

echo "[$(date)] all requested phases done (log: $LOG). Copy back (no index files, no DARTH train logs):"
echo "  find . -maxdepth 1 \( -name 'results_*' -o -name '*.log' -o -name '*.csv' \) -newer .revision_start -print0 |"
echo "    tar --null -T - --exclude=train_log.csv --exclude='*.index' --exclude='*.faiss' -czf revision_results.tgz"
