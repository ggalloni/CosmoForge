"""Command-line entry points for the QUBE pipeline.

``qube-fisher-run`` builds the Fisher matrix and stops; ``qube-spectra-run``
(alias ``qube-run``) estimates QML power spectra end to end, building its own
Fisher. Both run unmodified under MPI.

Persistence is opt-in (ADR-0015): a run whose output path is unset computes the
result and discards it, so both entry points warn rather than exit quietly.
"""

import argparse
from pathlib import Path

from qube.fisher import Fisher
from qube.spectra import Spectra

#: Packaged config used when no path is given on the command line. Resolved
#: relative to this file so the entry points work from any working directory.
DEFAULT_CONFIG = Path(__file__).parent / "TEB_defaults.yaml"


def _base_parser(description: str) -> argparse.ArgumentParser:
    """Build the parser shared by both entry points: an optional config path."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help=f"Path to the YAML configuration (default: {DEFAULT_CONFIG.name}).",
    )
    return parser


def fisher_main() -> None:
    """Entry point for ``qube-fisher-run``."""
    args = _base_parser(
        "Compute a QML Fisher matrix from a CosmoForge YAML configuration."
    ).parse_args()

    fisher = Fisher(args.config)
    log = fisher.logger
    log.info(f"Configuration: {args.config}")
    fisher.run()

    if fisher.rank != 0:
        return

    matrix = fisher.get_fisher_matrix()
    log.info(f"Fisher matrix: {matrix.shape[0]} x {matrix.shape[1]}")

    if not fisher.params.outfilefisher:
        log.warning(
            "outfilefisher is unset: the Fisher matrix was NOT written to disk. "
            "Set it in the configuration to persist this run."
        )


def spectra_main() -> None:
    """Entry point for ``qube-spectra-run`` and its ``qube-run`` alias."""
    parser = _base_parser(
        "Estimate QML power spectra from a CosmoForge YAML configuration."
    )
    parser.add_argument(
        "--mode",
        choices=("deconvolved", "decorrelated", "convolved"),
        default="deconvolved",
        help="Normalisation of the estimates (default: deconvolved).",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Write the spectra here. Unset means they are not written.",
    )
    args = parser.parse_args()

    spectra = Spectra(args.config)
    log = spectra.logger
    log.info(f"Configuration: {args.config}")
    spectra.run()

    if spectra.rank != 0:
        return

    if args.out:
        spectra.write_power_spectra(mode=args.mode, filename=args.out)
        log.info(f"Spectra written to {args.out}")
    else:
        log.warning(
            "--out is unset: the spectra were NOT written to disk. "
            "Pass --out PATH to persist this run."
        )
