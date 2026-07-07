"""Predict QUBE pipeline memory consumption from basis dimensions.

Companion to the algebraic memory analysis in Paper I §6 — analogous to
ECLIPSE_Memory.f90 but covering both QUBE pipeline paths.

Two paths are supported:

- **harmonic**: Tegmark-style V projection + SMW kernel. Persistent state
  is dominated by ``L`` (n_pix²), ``V_N_inv`` (n_modes × n_pix), and
  ``V_Ninv_VT`` (n_modes²).
- **pixel-direct**: Gjerløw-style direct pixel-space C^{-1}. No V, no
  SMW. Persistent state is dominated by ``Cov_T`` (Core retains it; Core
  only nullifies ``noise_cov1`` for non-pixel-direct modes) and
  ``basis._N`` (asfortranarray F-order copy).

The no-basis traditional path is not modelled here. Its Fisher->Spectra
handoff (ADR-0016) aliases the live Fisher's C^{-1} and retained N rather
than re-reading them from disk, so Spectra's covariance footprint is one
n_pix² array lighter than the old disk round-trip, not heavier.

For each path the calculator splits stage cost into:

- **Persistent state** at stage exit (audit-backed, ~1% accurate at the
  calibration points below).
- **Transient state** at stage peak (modelled where the algebra is
  known; assumed simultaneously alive at the peak — conservative when
  transients fire sequentially).

``StageBudget.peak_bytes = persistent_bytes + Σ transient_bytes``.

Allocator overhead, BLAS workspace, and the empirical
``fisher.compute.derivative_cache`` transient (queue item E for the
harmonic path; ~``n_params × n_pix²`` for the pixel-direct path) are not
modelled in detail. Expect ~5–10% allocator/BLAS slack at production
scale, and an additive Python/numpy overhead floor (~0.2–0.5 GiB) that
dominates at small ``n_pix``.

Calibration points:

- **Harmonic** at eclipse-QU (n_pix=59136, n_modes=33274, lmax_signal=256,
  lmax=128; mem_eclipse_qu_20114986.out, commit d11ab0b):
  basis_setup persistent 57.21 GiB predicted vs 57.80 GiB measured;
  basis_setup peak 112.6 GiB predicted vs 113.2 GiB measured.
- **Pixel-direct** at QU_nside64 fsky=0.1 (n_pix=9800, lmax=128,
  basis_lmax=256 default → switch implicit; mem_nc_20103507.out, commit
  ccabffd): basis_setup persistent (above covariance) 1.43 GiB predicted
  vs 1.44 GiB measured.
"""

from dataclasses import dataclass, field
from typing import Literal

GIBIBYTE: int = 1024**3
_BYTES_PER_DOUBLE: int = 8


@dataclass(frozen=True)
class BudgetConfig:
    """Inputs to the QUBE harmonic-path budget.

    n_pix: total pixel count (2 × n_pix_observed for QU spin-2;
        n_pix_observed for T spin-0; sum for TQU).
    n_modes: total mode count after V projection. The calculator does not
        derive this from lmax_signal because mode counting depends on spin
        and m=0 exclusions.
    lmax_signal: signal-cov ceiling (Layer A; max ℓ in V), not the
        parameter-grid lmax.
    lmax: inference window upper (Layer B). None or equal to ``lmax_signal``
        disables the switch optimisation. Below ``lmax_signal`` activates
        ``_compute_effective_noise`` and adds ``T`` (separate from
        ``V_Ninv_VT``) plus ``S_fixed`` and the corr intermediate to
        basis_setup.
    release_pixel_projector: True matches PR #16's Fisher path (V freed
        after SMW build). False reproduces the legacy keep-V behaviour.
    """

    n_pix: int
    n_modes: int
    lmax_signal: int
    lmax: int | None = None
    release_pixel_projector: bool = True

    def __post_init__(self) -> None:
        if self.n_pix <= 0:
            raise ValueError(f"n_pix must be positive (got {self.n_pix})")
        if self.n_modes <= 0:
            raise ValueError(f"n_modes must be positive (got {self.n_modes})")
        if self.lmax_signal <= 0:
            raise ValueError(f"lmax_signal must be positive (got {self.lmax_signal})")
        if self.lmax is not None and self.lmax > self.lmax_signal:
            raise ValueError(f"lmax={self.lmax} exceeds lmax_signal={self.lmax_signal}")

    @property
    def has_switch(self) -> bool:
        return self.lmax is not None and self.lmax < self.lmax_signal


@dataclass(frozen=True)
class PixelDirectBudgetConfig:
    """Inputs to the QUBE pixel-direct path budget.

    n_pix: total pixel count (same convention as BudgetConfig).
    lmax_signal: signal-cov ceiling (Layer A). Used for the auto-picker
        informational fields and to decide whether the implicit-switch
        S_fixed transient is allocated by Core (lmax_signal > params.lmax
        → switch implicit).
    n_bins: number of bandpower bins in the analysis. Drives the
        per-parameter ``cinv_times_dcb`` dict size during fisher_run.
    n_params: number of derivative parameters. ``n_bins × n_spectra`` in
        practice (e.g. n_bins=6 × 3 spectra = 18 at eclipse-QU). Used for
        the empirical fisher_run derivative-product transient.
    has_switch: True if Core's ``setup_computation_basis`` enters the
        S_fixed branch before discovering the path is pixel-direct (i.e.
        params.lmax < lmax_signal). At default benchmark configs
        (lmax_signal=4·nside, params.lmax=2·nside) this is True. The
        S_fixed buffer is allocated, populated, then dereferenced — but
        the allocator pool keeps it resident through basis_setup exit.
    """

    n_pix: int
    lmax_signal: int
    n_bins: int
    n_params: int
    has_switch: bool = True

    def __post_init__(self) -> None:
        if self.n_pix <= 0:
            raise ValueError(f"n_pix must be positive (got {self.n_pix})")
        if self.lmax_signal <= 0:
            raise ValueError(f"lmax_signal must be positive (got {self.lmax_signal})")
        if self.n_bins <= 0:
            raise ValueError(f"n_bins must be positive (got {self.n_bins})")
        if self.n_params <= 0:
            raise ValueError(f"n_params must be positive (got {self.n_params})")


@dataclass
class StageBudget:
    """Memory cost for one pipeline stage."""

    name: str
    persistent: dict[str, int] = field(default_factory=dict)
    transient: dict[str, int] = field(default_factory=dict)

    @property
    def persistent_bytes(self) -> int:
        return sum(self.persistent.values())

    @property
    def transient_bytes(self) -> int:
        return sum(self.transient.values())

    @property
    def peak_bytes(self) -> int:
        return self.persistent_bytes + self.transient_bytes


@dataclass
class QUBEBudget:
    """Full pipeline budget across all four QUBE stages."""

    config: BudgetConfig | PixelDirectBudgetConfig
    stages: list[StageBudget]
    path: Literal["harmonic", "pixel_direct"] = "harmonic"

    @property
    def lifetime_peak_bytes(self) -> int:
        return max(s.peak_bytes for s in self.stages)

    def stage(self, name: str) -> StageBudget:
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"unknown stage {name!r}")

    def format_table(self) -> str:
        return _format_table(self)


def predict_qube_budget(config: BudgetConfig) -> QUBEBudget:
    pix_sq = config.n_pix * config.n_pix * _BYTES_PER_DOUBLE
    mode_sq = config.n_modes * config.n_modes * _BYTES_PER_DOUBLE
    mode_pix = config.n_modes * config.n_pix * _BYTES_PER_DOUBLE

    covariance_setup = StageBudget(
        name="covariance_setup",
        persistent={"Cov_T": pix_sq},
        transient={"Cov_T (asfortranarray copy on read)": pix_sq},
    )

    basis_persistent: dict[str, int] = {
        "L (Cholesky factor of N, in-place)": pix_sq,
        "V_N_inv": mode_pix,
        "V_Ninv_VT (M kernel)": mode_sq,
    }
    if config.has_switch:
        # _noise_cov_T diverges from _V_Ninv_VT only on the switch path; the
        # no-switch path aliases the buffer (harmonic.py:322).
        basis_persistent["T (noise-bias kernel, switch path)"] = mode_sq
    if not config.release_pixel_projector:
        basis_persistent["V (pixel projector)"] = mode_pix

    basis_transient: dict[str, int] = {}
    if config.release_pixel_projector:
        basis_transient["V (transient before release)"] = mode_pix
    if config.has_switch:
        basis_transient["S_fixed (switch optimisation)"] = pix_sq
        basis_transient["corr intermediate (V_N_inv @ S_fixed)"] = mode_pix

    basis_setup = StageBudget(
        name="basis_setup",
        persistent=basis_persistent,
        transient=basis_transient,
    )

    fisher_run = StageBudget(
        name="fisher_run",
        persistent=dict(basis_persistent),
    )

    spectra_persistent = dict(basis_persistent)
    spectra_persistent["noise_cov_T (Spectra)"] = mode_sq
    spectra_run = StageBudget(
        name="spectra_run",
        persistent=spectra_persistent,
    )

    return QUBEBudget(
        config=config,
        stages=[covariance_setup, basis_setup, fisher_run, spectra_run],
        path="harmonic",
    )


def predict_pixel_direct_budget(config: PixelDirectBudgetConfig) -> QUBEBudget:
    """Predict QUBE memory budget on the pixel-direct path."""
    pix_sq = config.n_pix * config.n_pix * _BYTES_PER_DOUBLE

    covariance_setup = StageBudget(
        name="covariance_setup",
        persistent={"Cov_T (Core retains; not nullified for pixel-direct)": pix_sq},
        transient={"Cov_T (asfortranarray copy on read)": pix_sq},
    )

    # basis_setup adds two pix_sq terms above covariance_setup. The first
    # is structural (asfortranarray F-order copy at base.py:148). The
    # second only appears when Core's setup_computation_basis enters the
    # S_fixed allocation branch before discovering the path is
    # pixel-direct — the buffer is dereferenced at line 610 of core.py
    # but the allocator pool keeps it resident until basis_setup exit.
    basis_persistent: dict[str, int] = {
        "Cov_T (carried from covariance_setup)": pix_sq,
        "basis._N (asfortranarray F-order copy)": pix_sq,
    }
    if config.has_switch:
        basis_persistent["S_fixed (allocator pool retained)"] = pix_sq
    basis_setup = StageBudget(name="basis_setup", persistent=basis_persistent)

    # fisher_run: C_inv (full pixel-space inverse) plus the cinv_times_dcb
    # dict of n_params dense n_pix² products held through trace_loop.
    fisher_persistent = dict(basis_persistent)
    fisher_persistent["_direct_pix_buffer (lazy on first derivative)"] = pix_sq
    fisher_transient = {
        "C_inv (basis_manager.get_projected_inverse)": pix_sq,
        "cinv_times_dcb (n_params dense pixel matrices)": config.n_params * pix_sq,
    }
    fisher_run = StageBudget(
        name="fisher_run",
        persistent=fisher_persistent,
        transient=fisher_transient,
    )

    # Spectra noise-bias path: noise_cov_w = C_bar_inv @ N_bar @ C_bar_inv
    # is n_pix² in pixel-direct (no compression to compress to). Per-bin
    # derivative products are recomputed from C_inv and dC_b similarly to
    # Fisher — same transient shape but bounded by spectra parameter count.
    spectra_persistent = dict(fisher_persistent)
    spectra_persistent["noise_cov_w (Spectra)"] = pix_sq
    # Spectra recomputes derivatives when cache_derivatives=False — peak
    # transient empirically scales with n_bins (per-bin C^{-1} dC products
    # held during the inner loop). The ceiling here over-predicts measured
    # spectra peaks by ~20% at production scales; that's the acceptable
    # bound for cluster sizing.
    spectra_run = StageBudget(
        name="spectra_run",
        persistent=spectra_persistent,
        transient={
            "C_inv per parameter point": pix_sq,
            "per-bin C^{-1} dC products": config.n_bins * pix_sq,
        },
    )

    return QUBEBudget(
        config=config,
        stages=[covariance_setup, basis_setup, fisher_run, spectra_run],
        path="pixel_direct",
    )


def _format_bytes(b: int) -> str:
    return f"{b / GIBIBYTE:9.2f} GiB"


def _format_table(budget: QUBEBudget) -> str:
    cfg = budget.config
    if isinstance(cfg, BudgetConfig):
        switch_str = f"lmax={cfg.lmax}" if cfg.lmax is not None else "no switch"
        header = (
            f"  n_pix={cfg.n_pix}  n_modes={cfg.n_modes}  lmax_signal={cfg.lmax_signal}"
            f"  {switch_str}  release_V={cfg.release_pixel_projector}"
        )
    else:
        switch_str = "switch implicit" if cfg.has_switch else "no switch"
        header = (
            f"  n_pix={cfg.n_pix}  lmax_signal={cfg.lmax_signal}"
            f"  n_bins={cfg.n_bins}  n_params={cfg.n_params}  {switch_str}"
        )
    lines = [
        f"QUBE memory budget [{budget.path}]",
        header,
        "",
    ]
    name_width = 48
    for stage in budget.stages:
        lines.append(
            f"[{stage.name}]  persistent={_format_bytes(stage.persistent_bytes)}"
            f"  peak={_format_bytes(stage.peak_bytes)}"
        )
        if stage.persistent:
            lines.append("  persistent:")
            for term, sz in stage.persistent.items():
                lines.append(f"    {term:<{name_width}}{_format_bytes(sz)}")
        if stage.transient:
            lines.append("  transient (live at stage peak):")
            for term, sz in stage.transient.items():
                lines.append(f"    {term:<{name_width}}{_format_bytes(sz)}")
        lines.append("")
    lines.append(
        f"Lifetime peak: {_format_bytes(budget.lifetime_peak_bytes)}"
        " (above Python baseline RSS, ~25–30 GiB at typical configs)"
    )
    lines.append(
        "Excludes: BLAS scratch, allocator overhead, derivative_cache transient"
        " (~5 × n_modes² × 8 B at eclipse-QU)."
    )
    return "\n".join(lines)


def _main() -> None:  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict QUBE pipeline memory consumption from basis dimensions",
    )
    parser.add_argument(
        "--path",
        choices=["harmonic", "pixel_direct"],
        default="harmonic",
        help="QUBE pipeline path to model",
    )
    parser.add_argument(
        "--n-pix", type=int, required=True, help="total pixel count (Q+U for spin-2)"
    )
    parser.add_argument(
        "--lmax-signal", type=int, required=True, help="signal-cov ceiling (Layer A)"
    )
    # Harmonic-only
    parser.add_argument(
        "--n-modes", type=int, default=None, help="(harmonic) total mode count after V"
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=None,
        help="(harmonic) inference window upper (Layer B); omit to disable switch",
    )
    parser.add_argument(
        "--keep-pixel-projector",
        action="store_true",
        help="(harmonic) legacy: do not release V after SMW build",
    )
    # Pixel-direct only
    parser.add_argument(
        "--n-bins", type=int, default=None, help="(pixel_direct) number of bandpower bins"
    )
    parser.add_argument(
        "--n-params",
        type=int,
        default=None,
        help="(pixel_direct) total derivative parameter count",
    )
    parser.add_argument(
        "--no-switch",
        action="store_true",
        help="(pixel_direct) Core did not allocate S_fixed transient",
    )
    args = parser.parse_args()

    if args.path == "harmonic":
        if args.n_modes is None:
            parser.error("--n-modes is required for --path harmonic")
        config = BudgetConfig(
            n_pix=args.n_pix,
            n_modes=args.n_modes,
            lmax_signal=args.lmax_signal,
            lmax=args.lmax,
            release_pixel_projector=not args.keep_pixel_projector,
        )
        print(predict_qube_budget(config).format_table())
    else:
        if args.n_bins is None or args.n_params is None:
            parser.error("--n-bins and --n-params are required for --path pixel_direct")
        config = PixelDirectBudgetConfig(
            n_pix=args.n_pix,
            lmax_signal=args.lmax_signal,
            n_bins=args.n_bins,
            n_params=args.n_params,
            has_switch=not args.no_switch,
        )
        print(predict_pixel_direct_budget(config).format_table())


if __name__ == "__main__":
    _main()
