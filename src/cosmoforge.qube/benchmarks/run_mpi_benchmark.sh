#!/bin/bash
# Run MPI scaling benchmark for QU nside=16 lmax=32
# Uses proper core binding: each rank gets NCORES/nranks threads
# Usage: bash run_mpi_benchmark.sh

unset I_MPI_PMI_LIBRARY
NCORES=48
cd "$(dirname "$0")"

# Remove previous results
rm -f benchmark_mpi_results.json

for np in 8 16 32 48; do
    threads=$((NCORES / np))
    echo ""
    echo "=========================================="
    echo "Running with $np MPI ranks x $threads threads = $NCORES cores"
    echo "=========================================="
    env \
        OMP_NUM_THREADS=$threads \
        OPENBLAS_NUM_THREADS=$threads \
        MKL_NUM_THREADS=$threads \
        BLIS_NUM_THREADS=$threads \
        NUMEXPR_NUM_THREADS=$threads \
        VECLIB_MAXIMUM_THREADS=$threads \
        NUMBA_NUM_THREADS=$threads \
        mpirun -n $np \
            -genv I_MPI_PIN_DOMAIN=omp \
            -genv OMP_NUM_THREADS=$threads \
            -genv OPENBLAS_NUM_THREADS=$threads \
            -genv MKL_NUM_THREADS=$threads \
            -genv BLIS_NUM_THREADS=$threads \
            -genv NUMEXPR_NUM_THREADS=$threads \
            -genv VECLIB_MAXIMUM_THREADS=$threads \
            -genv NUMBA_NUM_THREADS=$threads \
            uv run python -u benchmark_mpi.py
done

echo ""
echo "=========================================="
echo "All runs complete. Results:"
echo "=========================================="
cat benchmark_mpi_results.json
