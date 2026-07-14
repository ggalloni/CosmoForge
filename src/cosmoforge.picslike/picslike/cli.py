"""Command-line entry point for the PICSLike pixel-space likelihood.

Installs ``picslike-run``, which evaluates the Gaussian pixel-space
log-likelihood over the parameter grid declared in the configuration. It runs
unchanged under MPI, since a console script is an executable on ``PATH``::

    picslike-run config.yaml
    mpirun -n 8 picslike-run config.yaml

Grid points are partitioned across ranks. Unlike QUBE there is no packaged
default configuration, so the path is required.

Persistence is opt-in (ADR-0015): without ``--out`` the grid is evaluated and
the result discarded. The entry point says so rather than exiting quietly.
"""

import argparse

from picslike.picslike import PICSLike


def main() -> None:
    """Entry point for ``picslike-run``."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the PICSLike pixel-space Gaussian likelihood over a parameter grid."
        )
    )
    parser.add_argument("config", help="Path to the YAML configuration.")
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Write the likelihood results here. Unset means they are not written.",
    )
    args = parser.parse_args()

    likelihood = PICSLike(args.config)
    log = likelihood.logger
    log.info(f"Configuration: {args.config}")
    likelihood.run()

    # The result accessors read off `likelihood_result`, which only rank 0
    # holds; on a worker they raise rather than return None.
    if likelihood.rank != 0:
        return

    chi2 = likelihood.get_chi_squared()
    log.info(f"Grid evaluated: {chi2.size} points")
    log.info(f"Minimum chi2: {chi2.min():.4f}")
    for name, value in likelihood.get_best_fit().items():
        log.info(f"  best fit {name}: {value:.6g}")

    if args.out:
        likelihood.save_results(args.out)
        log.info(f"Results written to {args.out}")
    else:
        log.warning(
            "--out is unset: the likelihood results were NOT written to disk. "
            "Pass --out PATH to persist this run."
        )
