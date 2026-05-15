# ADR-0007: Bins class adapted from xQML; attribution to Vanneste+ 2018

## Status

Accepted — implemented in `cosmocore/bins.py`.

## Context

CosmoForge's bandpower binning is implemented via a `Bins` class
(`fromdeltal` classmethod, `cutfirst` filter, P/Q bin operators,
`bin_spectra` method, attributes `lmins`, `lmaxs`, `nbins`, `lbin`,
`dl`). The structure of this class — particularly the P/Q operator
construction — is adapted directly from the xQML codebase
(Vanneste et al. 2018, arXiv:1807.02484, **GPLv3**).

Two distinct things must be cited:

1. **The binning formalism** (P/Q operators applied to per-ℓ
   bandpowers) is standard from Bond, Jaffe & Knox 1998. It is not
   xQML's invention.
2. **The specific implementation pattern** (`fromdeltal` factory,
   the way P/Q matrices are constructed, the choice of attribute
   names) is xQML's. Reusing it triggers GPLv3 obligations.

GPLv3 is a copyleft license. Carrying xQML-derived code into
CosmoForge means CosmoForge's binning code itself must remain
GPLv3-compatible, and downstream users redistributing CosmoForge
must comply. This is not optional and not a polite request — it is
a license obligation.

## Decision

Attribute the `Bins` implementation to xQML / Vanneste+ 2018 in
both code and paper. The attribution is mandatory; the choice is
about *where* it appears, not *whether*.

- **Module docstring** in `cosmocore/bins.py` cites xQML and
  Vanneste+ 2018 explicitly, including arXiv ID and license.
- **Paper** introduces binning with a citation to Vanneste+ 2018
  (e.g. *"We adopt a binning scheme following Vanneste et al.
  (2018), extended with…"*). Bond, Jaffe & Knox 1998 is also cited
  for the underlying P/Q formalism.
- The implementation lists the **CosmoForge additions on top of
  xQML**: input validation (overlap checks, sorting), `lmin`
  zero-padding in `bin_spectra`, `Dl` weighting, `bin_covariance`,
  type annotations, docstrings. Future contributors should preserve
  the distinction between xQML-derived parts and CosmoForge
  additions, so the lineage stays auditable.

## Consequences

- **License obligation**: CosmoForge's binning code is GPLv3 by
  inheritance. The package as a whole must remain
  GPLv3-compatible. Cannot be relicensed to MIT/BSD/Apache without
  reimplementing the xQML-derived parts from scratch.
- **Reimplementation option (deferred)**: a clean-room reimplementation
  from Bond, Jaffe & Knox 1998 alone — without reference to xQML —
  would relax the GPLv3 constraint to whatever CosmoForge chooses.
  Worth doing only if the GPLv3 constraint becomes a real obstacle
  to adoption. Out of scope here.
- **Citation discipline**: any paper draft mentioning binning must
  cite Vanneste+ 2018. Reviewers and readers tracking provenance
  expect this; omission would also breach the GPLv3 attribution
  notice requirement.
- **Future Bins extensions** (e.g. uneven binning, log-ℓ binning,
  bandpower-window-aware binning) inherit the same attribution
  obligation as long as they build on the existing class.

## Note on initial-binning vs rebinning (Bond/Jaffe/Knox 1998 §IV)

Two distinct estimators are admissible for binned bandpowers and
they do **not** coincide in general:

- **Initial binning** (CosmoForge's path): the derivative matrices
  `dC^b = Σ_{ℓ∈b} P_{b,ℓ} dC^ℓ` are summed *before* Fisher / QML
  evaluation. Fisher is `(n_bins × n_bins)` directly. The estimator
  is `(P F P^T)^{-1} P q` only when the unbinned `F` is diagonal in
  ℓ. For non-diagonal Fisher (the generic case) the algebra reduces
  to the binned trace `Tr[C^{-1} dC^b C^{-1} dC^{b'}]`, which is
  what CosmoForge implements.
- **Rebinning**: estimate per-ℓ spectra first, then collapse with
  `(P F P^T)^{-1} P F q`. Different operand ordering; identical to
  initial binning iff `F` is diagonal.

The two coincide for an azimuthally symmetric mask and isotropic
noise; they diverge whenever the mask/noise induces ℓ-ℓ' coupling.
CosmoForge ships the initial-binning form for both algebraic
simplicity and because the per-bin derivative path is the natural
intermediate for both bases.

## References

- Bond, Jaffe & Knox 1998 — original P/Q bandpower formalism,
  including the §IV distinction between initial-binning and
  rebinning estimators.
- Vanneste et al. 2018 (xQML), arXiv:1807.02484 — implementation
  pattern adopted here.
