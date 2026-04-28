#!/bin/bash
# Run QML vs PCL methodology comparison.
# Single process, threaded LAPACK. Outputs land next to the script.
# Usage: bash run_qml_vs_pcl.sh

unset I_MPI_PMI_LIBRARY
NCORES=${NCORES:-48}
cd "$(dirname "$0")"

echo "Running qml_vs_pseudocl.py with OMP_NUM_THREADS=$NCORES"
OMP_NUM_THREADS=$NCORES uv run --extra pcl python -u qml_vs_pseudocl.py
