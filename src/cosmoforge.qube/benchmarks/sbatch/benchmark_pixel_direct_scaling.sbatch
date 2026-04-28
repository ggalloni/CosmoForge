#!/bin/bash
#SBATCH --job-name=cf_bench_pds
#SBATCH --account=INF26_litebird_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=logs/pds_%j.out
#SBATCH --error=logs/pds_%j.err

set -euo pipefail

# SLURM copies the script into /var/spool/slurmd/...; $0 and BASH_SOURCE
# point there, not at the submission directory. Use SLURM_SUBMIT_DIR
# (set by SLURM to the directory the user ran sbatch from) to resolve
# the layout, with a fallback for non-SLURM execution.
CF_SBATCH_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}"
CF_BENCH_DIR="$(cd "${CF_SBATCH_DIR}/.." && pwd)"
CF_REPO_ROOT="$(cd "${CF_BENCH_DIR}/../../.." && pwd)"
export CF_REPO_ROOT

source "${CF_SBATCH_DIR}/_env.sh"
load_cluster_modules
print_run_header

unset I_MPI_PMI_LIBRARY
export OMP_NUM_THREADS="${CF_NCORES}"

cd "${CF_BENCH_DIR}"
uv run python -u benchmark_pixel_direct_scaling.py
