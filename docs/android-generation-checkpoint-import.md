# Verified research checkpoint import for Production Generation

## Purpose

`scripts/import_generation_checkpoint.ps1` imports a verified research
checkpoint into the app-private Production Generation store.  It is intended
to make an already validated checkpoint selectable by the existing Generation
screen; it neither starts training nor makes a research run resumable.

The importer is architecture-generic.  For example, a
V1024/T32/D64/FFN128/L19/H2 checkpoint is a Generation-capable candidate, but
the importer does not special-case D64 or FFN128.  The checkpoint bytes and
their decoded header, rather than a filename or a recipe label, are the
authoritative identity.

## Why research checkpoints are not automatically visible

Research/HPO artifacts are deliberately kept outside the Production
app-private store.  Automatically walking host research directories would
make a partially written, unverified, or tokenizer-mismatched file selectable
by the app.  It would also blur the boundary between private experiment data,
Generation payloads, and Training resume state.  Import is therefore an
explicit, validation-gated copy into a narrow namespace.

## Production checkpoint discovery

Production Generation continues to obtain candidates through
`TrainingCheckpointStore.listNativeCheckpointPaths()` and inspect each one
with `GenerationCheckpointInspector`.  The root is the existing app-private
`standalone_training_checkpoints` directory; no parallel Generation registry
is created.  Existing recursive discovery consequently sees a published
`model.ckpt` below the import namespace just as it sees other native
checkpoints.

The inspector remains the compatibility authority.  An imported file that is
malformed, non-finite, unsupported, or missing its required tokenizer is shown
as incompatible/invalid rather than being promoted to a compatible model.

## Import layout

Only a fully published directory is placed under the Production root:

```text
standalone_training_checkpoints/
  imported/
    <import-id>/
      model.ckpt
      byte-bpe-v1024.model       # required only for byte-BPE V1024
```

The importer first uses an app-private, non-discoverable staging location:

```text
generation-import-staging/
  <import-id>.tmp/
    model.ckpt
    byte-bpe-v1024.model
```

`model.ckpt` is a stable destination name, not an identity claim.  The actual
format/header, checkpoint SHA-256, parameter hash, model shape, step, seed,
and tokenizer binding determine the identity.  An import ID may be derived
deterministically from that information and a short content hash.

## Validation

Before transfer, the host validates the checkpoint fail-closed using the
formats that Production Generation can actually inspect (including
`NPRTCKPTV3` when applicable).  Validation checks the decoded format; V, T,
D, FFN, L, H; step; seed; tokenizer kind/hash; parameter hash; payload size;
and finite parameter state.  A supplied expected SHA-256 or expected identity
field is an additional strict check, not a substitute for inspection.

After staging, the device repeats the relevant checks against its copied
bytes: presence, byte size, checkpoint SHA-256, header, model shape, step,
seed, finite state, tokenizer binding, and parameter hash.  Host success
alone never authorizes publication.  Device-side inspection should reuse the
Generation inspection/compatibility path wherever possible so the import and
selector do not acquire divergent acceptance rules.

## Tokenizer sidecar

A V1024 checkpoint with `tokenizer_kind=byte_bpe` requires the canonical
tokenizer model supplied with `-TokenizerModel`.  Its actual SHA-256 identity
must equal the `tokenizer_hash` encoded in the checkpoint header.  On success,
the sidecar is copied adjacent to `model.ckpt` as `byte-bpe-v1024.model`, which
is where file-backed Production Generation resolves it.  Missing, unreadable,
or mismatched sidecars reject the import or make the candidate incompatible;
there is no fallback to another tokenizer.

Formats such as V256 that do not require a sidecar retain their existing
Generation semantics.  The importer must not add a byte-BPE sidecar
requirement where the existing inspector does not require one.

## Staging, atomic publish, and conflicts

The Production directory is never written directly.  Only after host and
device validation pass does the importer atomically rename the staged
directory to `imported/<import-id>`.  Directory rename is used where the
app-private filesystem supports it; the resulting directory is the sole point
at which Generation discovery can observe the checkpoint.

If the destination already exists and has the same checkpoint SHA-256, the
operation is an idempotent success.  If its content differs, import fails
closed: it never overwrites the existing directory.  On transfer or validation
failure, only the corresponding staging temporary directory may be removed.
Existing Production checkpoints, active-training artifacts, and unrelated
staging directories are preserved.

The per-import device lock is released only after the host observes a clean
instrumentation exit.  If transport, focus, or timeout state is ambiguous, the
lock and staging bytes are intentionally retained for reconciliation; a retry
must first establish that no owned instrumentation is still running.  A smoke
failure after a successful publish does not roll the published checkpoint back;
the next publish is expected to reconcile it through the idempotent same-bytes
path.

## Training isolation

Imported payloads are Generation-visible only.  They receive no
`TrainingCheckpointMetadata`, native-path registration, or fabricated dataset
and optimizer metadata, and must not appear as Training resume candidates.
If a broad Training directory walk would otherwise see the `imported`
namespace, Training resume discovery must exclude that namespace while
Generation keeps its existing `listNativeCheckpointPaths()` discovery
semantics.

## CLI usage

Run the importer with the source artifact and explicit pinned QAIRT identity:

```powershell
.\scripts\import_generation_checkpoint.ps1 `
  -Checkpoint "<path-to-best-step8000.ckpt>" `
  -TokenizerModel "<path-to-byte-bpe-v1024.model>" `
  -ExpectedStep 8000 `
  -ExpectedSeed 1 `
  -ExpectedV 1024 `
  -ExpectedT 32 `
  -ExpectedD 64 `
  -ExpectedFfn 128 `
  -ExpectedL 19 `
  -ExpectedH 2 `
  -QairtSdkRoot "C:\Qualcomm\AIStack\QAIRT\2.48.40.260702" `
  -ExpectedBuildId "2.48.40.260702151143"
```

`-ImportId`, `-Device`, and optional expected checkpoint SHA-256 may be used
when supported by the script.  Expected identity switches are optional; when
provided, every value must exactly match the decoded checkpoint.  Do not infer
identity from the source filename.

For a device-free preflight, add `-HostOnly`.  This runs the same bounded host
format, registry, finiteness, parameter-hash, and V3 tokenizer-binding checks
without QAIRT or ADB and never writes the Production store.  A device-backed
run publishes after the headless bridge verifies the staged bytes; add
`-RunGenerationSmoke` only when the pinned APKs and an eligible physical
device are available.

## Security and safety

The importer uses the repository's fixed QAIRT policy and fails closed on an
unavailable root, Build ID mismatch, unsupported format, identity mismatch,
non-finite payload, or staging/publish error.  It does not automatically find
or select another SDK, cached artifact, tokenizer, checkpoint, or backend.

Checkpoint and tokenizer artifacts may be private.  Do not commit them, raw
reports, raw evidence, device identifiers, or local absolute paths.  Do not
use this workflow to replace a Production checkpoint, clear app data, stop an
active run, or alter Training state.  Any device operation follows the
repository device-tier policy; foreground/UI validation and other Tier 3
operations require explicit authorization.

## Verification

The expected targeted checks cover:

- importer self-tests: a complete bounded checkpoint protocol fixture,
  tokenizer mismatch rejection, expected-identity rejection, and malformed
  input rejection;
- JVM/import tests: atomic directory publish, absent/wrong tokenizer sidecar,
  expected-identity rejection, idempotent duplicate, same-ID/different-content
  conflict, path traversal rejection, and Training-resume isolation;
- the Android headless bridge: recursive Generation discovery of the published
  subdirectory, inspector compatibility, and the same Training-resume
  exclusion on the app-private store;
- Fast verification appropriate to the changed PowerShell/Kotlin paths,
  including `git diff --check` and relevant JVM tests.

For an authorized device smoke, build against the explicitly pinned QAIRT
root, audit the APK, and use the Production Generation selection path.  The
smoke must separately record selected checkpoint identity, tokenizer identity,
HTP/QNN return-code success, required tensor finite status, CPU-fallback
status, and graph failures.  A successful QNN return code does not by itself
prove finite outputs or no fallback.

## Known limitations

- This is an explicit host-driven import; it does not add an Android file
  picker or automatic research-artifact discovery.
- It does not import a checkpoint into Training, enable resume, or recreate
  research schedule/dataset provenance in Production metadata.
- Atomic rename gives an all-or-nothing visibility boundary on the supported
  app-private filesystem; it is not a power-loss durability claim beyond the
  platform filesystem guarantees.
- A Generation smoke establishes compatibility and execution health only.  It
  is not a quality evaluation and does not claim NPU-only execution, no CPU
  usage, or QNN automatic differentiation.
