#!/bin/bash
#SBATCH --job-name=cf_bench_mpi
#SBATCH --account=INF26_litebird_1
#SBATCH --partition=g100_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --time=01:00:00
#SBATCH --output=logs/mpi_%j.out
#SBATCH --error=logs/mpi_%j.err

# MPI strong-scaling sweep at fixed problem size.
# Runs benchmark_mpi.py at N MPI ranks x (NCORES/N) OMP threads
# for N in {1,2,4,8} on the same allocation, so all rank counts see
# identical hardware and identical caches.
#
# The Python script appends/updates entries in benchmark_mpi_results.json
# under keys "1", "2", "4", "8" — one full sweep per sbatch invocation.

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

cd "${CF_BENCH_DIR}"

# Wipe stale results so the sweep starts fresh each submission.
rm -f results/benchmark_mpi_results.json
rm -f results/benchmark_mpi_fisher_*.npy
rm -f results/benchmark_mpi_spectra_*.npy

for n_ranks in 1 2 4 8; do
    threads=$(( CF_NCORES / n_ranks ))
    echo ""
    echo "----- N=$n_ranks MPI ranks  x  $threads OMP threads = ${CF_NCORES} cores -----"
    echo ""
    OMP_NUM_THREADS=$threads \
        mpirun -n $n_ranks \
            -genv I_MPI_PIN_DOMAIN=omp \
            uv run python -u benchmark_mpi.py
done

echo ""
echo "Sweep complete. Final JSON:"
cat results/benchmark_mpi_results.json
