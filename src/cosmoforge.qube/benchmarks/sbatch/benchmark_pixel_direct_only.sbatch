#!/bin/bash
#SBATCH --job-name=cf_bench_pdo
#SBATCH --account=INF26_litebird_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --output=logs/pdo_%j.out
#SBATCH --error=logs/pdo_%j.err

# benchmark_pixel_direct_only.py supports CLI flags that change which cells
# are run and where the JSON is written:
#   --fsky FLOAT        sky fraction (default in script)
#   --nsides A,B,C      comma-separated nsides to sweep
#   --fields T,QU       which field labels to run
#   --target-nbins INT  bandpower count per cell
#   --suffix STR        appended to output JSON name
#
# Any positional args you pass to sbatch are forwarded to the Python script,
# so e.g.:
#   sbatch benchmark_pixel_direct_only.sbatch \
#       --fsky 0.01 --nsides 16,32,64,128,256 --suffix both_fields

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
uv run python -u benchmark_pixel_direct_only.py "$@"
