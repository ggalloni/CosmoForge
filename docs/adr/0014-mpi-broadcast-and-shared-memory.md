# ADR-0014: MPI broadcast and shared-memory conventions

## Status

Accepted — established by the `MPISharedMemoryMixin` introduced in
PR #14 (2026-04-27); made explicit here to record the correctness
invariants that the code embodies but does not enforce.

## Context

CosmoForge runs on multi-node clusters where the load-bearing arrays
(noise covariance, signal kernels, derivative ingredients, basis
operators) reach tens of gigabytes at ECLIPSE-class resolutions.
Three distinct broadcast/distribution primitives are available and
each has a non-overlapping correctness or performance niche:

1. **`comm.bcast`** (lowercase) — serialisation-based broadcast of
   arbitrary Python objects. Convenient for `FieldCollection`,
   parameter dicts, config objects. Carries a **~2 GB message-size
   limit** on most MPI implementations because the serialisation goes
   through a single buffer.
2. **`comm.Bcast`** (uppercase) — buffer-based broadcast of a typed
   memory region. Bypasses the ~2 GB serialisation cap; produces
   **one independent copy per rank** (O(nranks × size) total memory
   on a node).
3. **`MPI.Win.Allocate_shared`** — intra-node shared memory.
   **One allocation per node**, attached as a view by every other
   rank on that node (O(size) per node, regardless of rank count).
   Inter-node distribution still requires an explicit `Bcast` between
   node-local rank 0s.

Naïvely picking among these is a footgun:

- Use lowercase `bcast` for a 5 GB numpy buffer and it silently
  truncates or raises an opaque MPI error at runtime.
- Use `Allocate_shared` for a buffer that consumers want to write
  into and you get torn writes — no MPI implementation guards against
  this, the shared buffer is just shared memory.
- Use uppercase `Bcast` for a Python object (a `FieldCollection`,
  a `Bins`) and it cannot serialise — you have to hand-roll the
  encode/decode.

Before this ADR, the rules existed in source comments and developer
memory only. New contributors reached for whatever felt easier and
hit each footgun at least once.

## Decision

`MPISharedMemoryMixin` in `cosmocore.mpi_utils` is the canonical
seat for the three primitives. Production code that needs to
distribute data across ranks calls one of its methods (or, for
trivial Python objects, calls `comm.bcast` directly with an explicit
acknowledgement of the size limit). The dispatch rule is:

| Payload | Method | Memory footprint |
|---|---|---|
| Large read-only numpy buffer (any size, on a multi-node run) | `_shared_array(arr)` | O(size) per node |
| Large writable numpy buffer, or read-only buffer ≥ 2 GB on a single-node run | `_bcast_array(arr)` | O(size) per rank |
| Python objects, small numpy arrays where copy cost is negligible | `comm.bcast(obj, root=0)` | O(serialised size) per rank, ≤ 2 GB total |

Concretely:

1. **`_shared_array` is the canonical zero-copy primitive.** It uses
   `MPI.Win.Allocate_shared` to allocate one buffer per node, seeded
   on node-local rank 0 either directly (when global rank 0 is on
   that node) or via an inter-node `Bcast` between node-local rank-0
   ranks. All other ranks on the node attach a numpy view onto the
   shared window. Callers treat the returned array as **read-only**;
   writing to it from a non-owning rank is undefined behaviour
   (silent data races, no MPI-level guard).

2. **Read-only is a discipline-level invariant.** The mixin does not
   set `arr.flags.writeable = False` on the returned view, because
   numpy's writeable flag is per-view and does not protect against
   buffer-level writes from other ranks anyway. Code review and this
   ADR are the enforcement mechanism. A future enforcement step
   (setting the writeable flag at attach time on non-owning ranks)
   is a candidate hardening; flagged as a follow-up, not a Decision
   item here.

3. **`_bcast_array` is the canonical large-writable / `>2 GB`
   primitive.** It uses uppercase `Bcast` over a typed buffer,
   bypassing the serialisation path and the implicit 2 GB cap. Every
   rank ends up with its own independent allocation. Use this when
   consumers actually need to write into the array, or on single-node
   runs where the per-rank duplication cost is acceptable and the
   shared-memory machinery (`Split_type`, inter-node communicator,
   window lifecycle) would be overkill.

4. **`comm.bcast` (lowercase) is reserved for Python objects.**
   `FieldCollection`, parameter dictionaries, `Bins` instances,
   config-derived state, small parameter arrays — anything where the
   serialisation path is the natural fit and the total size is well
   under 2 GB. Library code that uses lowercase `bcast` for a payload
   that *might* approach 2 GB is wrong; it must route through
   `_bcast_array`.

5. **Window lifecycle belongs to the mixin.** `_setup_shared_comm`
   creates the intra-node `Split_type(MPI.COMM_TYPE_SHARED)`
   communicator and the inter-node companion. `_cleanup_shared`
   frees every window and both communicators. `close()` and
   `__exit__()` route to `_cleanup_shared`; production consumers
   use the mixin as a context manager or call `close()` explicitly.
   Call sites do not touch `MPI.Win`, `Split_type`, or window
   freeing directly.

6. **The mixin requires `self.comm` and `self.rank`.** Classes that
   mix it in (`Fisher`, `Spectra`, `PICSLike` via `Core`) set these
   in `__init__`. New consumers must follow the same contract.

## Consequences

- **Memory savings at scale.** A 30 GB noise covariance broadcast
  via `_shared_array` consumes 30 GB per node, not 30 GB × ranks.
  This is the difference between fitting in memory and OOMing on the
  cluster's fat-memory partition.
- **The 2 GB lowercase-`bcast` footgun is contained.** Library code
  using lowercase `bcast` is by convention dealing with small Python
  objects; reviewers catch numpy payloads that drift past the cap
  before they ship.
- **Read-only is enforced by code review, not the runtime.** A
  future-you who needs to mutate a shared buffer should not — write
  into a local copy, or route the new code through `_bcast_array`
  if every rank needs to mutate independently. Memory note
  `MPI broadcast convention` in user memory carries the same rule.
- **Lifecycle bugs surface as leaked windows and never-freed
  communicators.** The context-manager / `close()` discipline keeps
  the lifecycle local to the owning object; the mixin's
  `_cleanup_shared` is idempotent so double-close on the same instance
  is safe.
- **Cross-cutting concern with ADR-0012.** Under the stub MPI branch
  (`HAS_MPI = False`) `_shared_array` falls through to a local heap
  buffer and `_bcast_array` returns its input unchanged. The
  discipline-level invariants in this ADR are vacuous under the stub
  because there is only one rank, but the API surface is identical so
  production code reads the same in both branches.

## Validation

- `cosmocore/tests/test_mpi.py`, `qube/tests/test_fisher_mpi.py`,
  `qube/tests/test_spectra_mpi.py` — exercise the `size == 1` path
  (both real-MPI and stub branches).
- Rank-mock tests in `qube/tests/test_fisher.py` and `test_spectra.py`
  cover multi-rank distribution patterns with mocked communicators
  (gated by `pytest.importorskip("mpi4py.MPI")` so they only run when
  real `mpi4py` is installed — see ADR-0012).
- Cluster runs at g100 / Leonardo are the integration test; the
  shared-memory path is exercised whenever a multi-node submission
  loads a ≥ 2 GB noise covariance.

## References

- PR #14 — introduced `MPISharedMemoryMixin`.
- `cosmocore/mpi_utils.py` — `_shared_array`, `_bcast_array`,
  `_setup_shared_comm`, `_cleanup_shared`.
- ADR-0012 — `mpi4py` stub dispatch; this ADR's primitives degrade to
  single-rank identity under the stub branch.
- `project_g100_socket_aware_mpi.md` (memory) — open follow-up on
  rank-per-socket layout that interacts with the intra-node split.
- Follow-up candidate (not in this ADR): set `writeable=False` on the
  numpy view returned by `_shared_array` on non-owning ranks, as a
  best-effort defensive guard.
