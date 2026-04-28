#!/bin/bash
# Submit every CosmoForge benchmark as an independent sbatch job.
# Each benchmark gets its own node allocation, so the jobs run in
# parallel and write into results/ from a single git commit.
#
# Usage:
#   bash submit_all.sh                  # submit all benchmarks
#   bash submit_all.sh mpi numba        # submit a subset
#
# Requires that sbatch is on PATH and _env.sh has been edited for
# the local cluster (account / partition / module loads).

set -euo pipefail

CF_SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${CF_SBATCH_DIR}"

source ./_env.sh

ALL_BENCHMARKS=(
    scaling
    mpi
    pixel_vs_harmonic
    pixel_direct_scaling
    pixel_direct_only
    numba
)

# Default to the full set, otherwise honour the user's selection.
if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
else
    SELECTED=("${ALL_BENCHMARKS[@]}")
fi

# Record the SHA being benchmarked so we can correlate JSON outputs
# with a specific commit even if HEAD moves later.
GIT_SHA="$(git -C "$(realpath ../../../..)" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Submitting CosmoForge benchmarks at git ${GIT_SHA}"
echo "Account:   ${CF_ACCOUNT}"
echo "Partition: ${CF_PARTITION}"
echo "QoS:       ${CF_QOS:-<partition default>}"
echo

mkdir -p logs

for name in "${SELECTED[@]}"; do
    sbatch_file="benchmark_${name}.sbatch"
    if [[ ! -f "${sbatch_file}" ]]; then
        echo "  SKIP ${name}: ${sbatch_file} not found"
        continue
    fi
    args=(
        --account="${CF_ACCOUNT}"
        --partition="${CF_PARTITION}"
    )
    if [[ -n "${CF_QOS}" ]]; then
        args+=(--qos="${CF_QOS}")
    fi
    echo "  sbatch ${sbatch_file}"
    sbatch "${args[@]}" "${sbatch_file}"
done

echo
echo "Submitted. Track progress with: squeue -u \$USER"
echo "Pull results back to laptop with rsync (see README.md)."
