#!/bin/bash
#SBATCH --job-name=cf_bench_scaling
#SBATCH --account=INF26_litebird_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=00:30:00
#SBATCH --output=logs/scaling_%j.out
#SBATCH --error=logs/scaling_%j.err

set -euo pipefail

CF_SBATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
CF_BENCH_DIR="$(dirname "${CF_SBATCH_DIR}")"
CF_REPO_ROOT="$(realpath "${CF_BENCH_DIR}/../../..")"
export CF_REPO_ROOT

source "${CF_SBATCH_DIR}/_env.sh"
load_cluster_modules
print_run_header

unset I_MPI_PMI_LIBRARY
export OMP_NUM_THREADS="${CF_NCORES}"

cd "${CF_BENCH_DIR}"
uv run python -u benchmark_scaling.py
