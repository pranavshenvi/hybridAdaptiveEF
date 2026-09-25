#!/usr/bin/env bash
# Runs every controlled-factor config back to back, then summarizes.
# Launch detached so an SSH drop doesn't kill it (summary.md §1 item 9):
#   nohup bash run_controlled_experiments.sh > controlled_run.log 2>&1 & disown
#
# Order is deliberate: A_sift_hard00 first (reuses the existing sift128 index,
# so it's a fast end-to-end check), then the rest of A, then the synthetic
# configs (each alpha builds its own index once; C reuses B_synth_a2.0's).
set -u

python3 controlled_datasets.py || exit 1

for cfg in A_sift_hard00 A_sift_hard10 A_sift_hard30 A_sift_hard50 \
           B_synth_a0.0 B_synth_a0.5 B_synth_a1.0 B_synth_a2.0 \
           C_synth_a2.0_hard50; do
  echo "=== $(date '+%F %T') starting $cfg"
  python3 benchmark_controlled.py --config "$cfg" || echo "!!! $cfg failed, continuing"
done

python3 summarize_controlled.py
