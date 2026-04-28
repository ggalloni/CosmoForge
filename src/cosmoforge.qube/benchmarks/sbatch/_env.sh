# Cluster-specific environment for CosmoForge benchmark sbatch jobs.
# Sourced at the top of every benchmark sbatch script.
# Edit the values below ONCE for your cluster, then leave this file alone.

# --- SLURM accounting ----------------------------------------------------
# Account / partition / QoS used by every sbatch job.
# The submit_all.sh wrapper passes these on the sbatch command line, but
# individual scripts can also read them from the environment.
#
# Confirm against `sacctmgr show user $USER` and `sinfo` on the cluster.
export CF_ACCOUNT="${CF_ACCOUNT:-INF26_litebird_1}"
export CF_PARTITION="${CF_PARTITION:-g100_usr_prod}"
# QoS is optional — leave empty to use the partition default.
export CF_QOS="${CF_QOS:-}"

# --- Repository root (cluster path) --------------------------------------
# Resolved once at the top of every sbatch script via realpath of the
# script's location, so this is informational only.
export CF_REPO_ROOT_DEFAULT="/g100_work/INF26_litebird_1/ggalloni/CosmoForge"

# --- Cluster modules / Python environment --------------------------------
# Customise to whatever brings `uv`, `mpirun`, and the project's Python
# interpreter onto PATH. On g100 you typically need a python module +
# `source ~/.bashrc` (or equivalent) so that uv is on PATH.
load_cluster_modules() {
    # Make uv reachable in non-interactive sbatch jobs. The standard uv
    # installer puts the binary in ~/.local/bin; sbatch jobs do NOT source
    # ~/.bashrc by default, so PATH from interactive sessions is not
    # inherited. This unconditional prepend is harmless if uv is already
    # on PATH.
    export PATH="${HOME}/.local/bin:${PATH}"

    # If uv is not reachable here, fall back to invoking the project venv
    # directly. Override the alias by setting CF_PYTHON before sbatching:
    #   CF_PYTHON="${CF_REPO_ROOT}/.venv/bin/python" sbatch ...
    if ! command -v uv >/dev/null 2>&1; then
        echo "WARN: uv not on PATH; expecting CF_PYTHON or .venv/bin/python" >&2
    fi

    # Cluster-specific module loads — uncomment as needed:
    # module load python/3.13
    # module load intel-oneapi-mpi
    # module load openblas
}

# --- Threading ------------------------------------------------------------
# Total cores per node on g100 (Intel Xeon Platinum 8260, 2 sockets x 24).
export CF_NCORES="${CF_NCORES:-48}"

# --- Diagnostic header ---------------------------------------------------
# Called by every sbatch script after sourcing this file. Records the
# git SHA, hostname, and date so the resulting JSON can be traced back to
# a specific commit.
print_run_header() {
    echo "==========================================================="
    echo "CosmoForge benchmark: $(basename "$0")"
    echo "Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Hostname:   $(hostname)"
    echo "SLURM Job:  ${SLURM_JOB_ID:-<not in slurm>}"
    echo "Node list:  ${SLURM_NODELIST:-<n/a>}"
    if command -v git >/dev/null 2>&1; then
        echo "Git SHA:    $(git -C "${CF_REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo '<not a git repo>')"
        echo "Git status: $(git -C "${CF_REPO_ROOT}" status --porcelain 2>/dev/null | wc -l) modified files"
    fi
    echo "==========================================================="
}
