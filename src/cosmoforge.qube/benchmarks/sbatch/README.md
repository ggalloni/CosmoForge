# CosmoForge benchmark sbatch scripts

SLURM batch scripts to run every benchmark on a single allocation per
benchmark, in parallel across cluster nodes, and write results into
`../results/` from a single git commit.

## Layout

```
sbatch/
├── README.md                              # this file
├── _env.sh                                # cluster-specific config (edit once)
├── submit_all.sh                          # dispatch all jobs
├── benchmark_scaling.sbatch               # nside/lmax sweep
├── benchmark_mpi.sbatch                   # MPI strong scaling (N=1,2,4,8)
├── benchmark_pixel_vs_harmonic.sbatch     # basis comparison at fsky=0.1
├── benchmark_pixel_direct_scaling.sbatch  # T+QU at nside 16/32/64
├── benchmark_pixel_direct_only.sbatch     # fsky=0.01
├── benchmark_numba.sbatch                 # JIT kernel timings
└── logs/                                  # per-job stdout/stderr
```

## First-time setup

Edit `_env.sh` and set:

- `CF_ACCOUNT` — your SLURM account (default `INF26_litebird_1`).
- `CF_PARTITION` — partition name (default `g100_usr_prod`).
- `CF_QOS` — leave empty for the partition default.
- `load_cluster_modules` — uncomment the `module load ...` lines you need
  to bring `uv`, `mpirun`, and the project's Python interpreter onto PATH.

The `--account` and `--partition` SBATCH directives at the top of each
`*.sbatch` file are populated for g100. If you change clusters, either
edit them in place or override on the sbatch command line via
`submit_all.sh` (the wrapper passes `--account` / `--partition` /
`--qos` from `_env.sh`, which take precedence over the in-file
directives).

Wall-time budgets are all set to **1h** so the same scripts work on the
debug QoS (`g100_qos_dbg`, 1h cap). Real runtimes from prior submissions
are well under that — MPI sweep ≈ 10 min, basis sweeps ≈ 5--10 min,
Numba ≈ 2 min — so the production partition can absolutely run them as
they are. If you ever need a longer slot for a more demanding sweep,
edit the `#SBATCH --time` line of the affected script.

### `uv` on the cluster

Sbatch jobs do **not** source `~/.bashrc` by default, so the standard
`uv` installer location (`~/.local/bin/uv`) may not be on PATH inside
the job. `_env.sh` prepends `~/.local/bin` to PATH unconditionally; if
that still doesn't pick up `uv` (e.g. you installed it elsewhere), set
`CF_PYTHON` before sbatching to bypass `uv` and invoke the project venv
directly:

```bash
CF_PYTHON="${PWD}/../../.venv/bin/python" sbatch benchmark_numba.sbatch
```

## Submitting

```bash
cd src/cosmoforge.qube/benchmarks/sbatch

# All benchmarks (six independent jobs):
bash submit_all.sh

# Only a subset:
bash submit_all.sh mpi numba

# Single benchmark, raw sbatch:
sbatch benchmark_pixel_vs_harmonic.sbatch

# pixel_direct_only forwards extra args to the Python script:
sbatch benchmark_pixel_direct_only.sbatch \
    --fsky 0.01 --nsides 16,32,64,128,256 --suffix both_fields
```

Each job writes its JSON into `../results/`. The JSON metadata block
captures the git SHA, hostname, SLURM job ID, BLAS/numba versions, and
thread counts at runtime, so the four files produced by one
`submit_all.sh` invocation should agree on the SHA --- this is the
property that makes the results paper-ready.

Track jobs with:

```bash
squeue -u $USER
```

## Pulling results back to the laptop

From the laptop:

```bash
rsync -avz --progress \
    ggalloni@login.g100.cineca.it:/g100_work/INF26_litebird_1/ggalloni/CosmoForge/src/cosmoforge.qube/benchmarks/results/ \
    ./src/cosmoforge.qube/benchmarks/results/
```

(Adjust the user / cluster path to match your account.)

## What each benchmark measures

| Script                              | Configuration                                       | Output                                              |
|-------------------------------------|-----------------------------------------------------|-----------------------------------------------------|
| `benchmark_scaling.py`              | T+QU at nside 8/16/32, harmonic basis, nsims=1000   | `benchmark_scaling_results.json`                    |
| `benchmark_mpi.py`                  | T at nside=32, lmax=64, nsims=10000, N=1,2,4,8 ranks| `benchmark_mpi_results.json` + `_fisher/_spectra_*.npy` |
| `benchmark_pixel_vs_harmonic.py`    | T at lmax 8/16/24/32/48, fsky=0.1, three methods    | `benchmark_pixel_vs_harmonic_results.json`          |
| `benchmark_pixel_direct_scaling.py` | T+QU at nside 16/32/64, fsky=0.1, three methods     | `benchmark_pixel_direct_scaling_partial_cluster.json` |
| `benchmark_pixel_direct_only.py`    | T+QU at fsky=0.01, three methods                    | `benchmark_pixel_direct_only_*_results.json`        |
| `benchmark_numba.py`                | JIT vs steady-state for legendre + signal_matrix    | `benchmark_numba_jit_results.json`                  |

## Threading model

Each `*.sbatch` (except `benchmark_mpi.sbatch`) requests one MPI rank with
all 48 cores as OMP threads. The MPI sweep runs N=1,2,4,8 ranks within a
single 48-core allocation, with `OMP_NUM_THREADS = 48 / N` per rank, so
all cores are utilised at every rank count.

`I_MPI_PMI_LIBRARY` is unset before each run because the system value
conflicts with `mpirun` invocations launched outside an `srun` context;
this matches the pattern in `run_mpi_benchmark.sh`.

## Reproducibility

Every sbatch script calls `print_run_header` which emits the git SHA and
the count of locally-modified files. The benchmark Python scripts also
embed the SHA in their JSON metadata. To produce paper-grade numbers:

1. Commit and push everything before submitting.
2. Run `bash submit_all.sh` --- this captures the HEAD SHA at submission.
3. After all jobs finish, the JSONs should all show `git.dirty == false`
   and identical `git.sha` values.

If a job fails or a JSON ends up dirty, rerun only the affected
benchmarks with `bash submit_all.sh <names>` rather than the full set.
