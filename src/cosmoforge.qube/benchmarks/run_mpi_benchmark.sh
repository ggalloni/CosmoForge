#!/bin/bash
# Run MPI scaling benchmark for QU nside=16 lmax=32
# Uses proper core binding: each rank gets NCORES/nranks threads
# Usage: bash run_mpi_benchmark.sh

unset I_MPI_PMI_LIBRARY
NCORES=48
cd "$(dirname "$0")"

# Remove previous results
rm -f benchmark_mpi_results.json

for np in 1 2 4 8 16 32 48; do
    threads=$((NCORES / np))
    echo ""
    echo "=========================================="
    echo "Running with $np MPI ranks x $threads threads = $NCORES cores"
    echo "=========================================="
    OMP_NUM_THREADS=$threads mpirun -n $np -genv I_MPI_PIN_DOMAIN=omp uv run python -u benchmark_mpi.py
done

echo ""
echo "=========================================="
echo "All runs complete. Results:"
echo "=========================================="
cat benchmark_mpi_results.json
