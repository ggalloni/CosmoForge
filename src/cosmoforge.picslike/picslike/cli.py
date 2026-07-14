"""Command-line entry point for the PICSLike pixel-space likelihood.

``picslike-run`` evaluates the Gaussian pixel-space log-likelihood over the
parameter grid declared in the configuration, partitioning grid points across
MPI ranks. Unlike QUBE there is no packaged default config, so the path is
required.

Persistence is opt-in (ADR-0015): without ``--out`` the grid is evaluated and
the result discarded, so the entry point warns rather than exit quietly.
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

    log.info(f"Best fit: {likelihood.get_best_fit()}")

    if args.out:
        likelihood.save_results(args.out)
        log.info(f"Results written to {args.out}")
    else:
        log.warning(
            "--out is unset: the likelihood results were NOT written to disk. "
            "Pass --out PATH to persist this run."
        )
