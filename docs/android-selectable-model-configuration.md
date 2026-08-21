# Android selectable model configuration contract

The standalone Training and Generation screens use one bounded architecture catalog instead of a single D32 preset.

## Supported architecture set

- Vocabulary: V256 legacy byte tokens, or V1024 canonical byte-BPE.
- Context/depth/heads: T32, L19, H2.
- Model dimension: D32, D48, or D64.
- FFN dimension: FFN32, FFN48, or FFN64, independently selectable.
- Training controls remain fixed: batch 8, learning rate 0.003, seed 1, checkpoint interval 250, and the canonical Adam identity.

This is 18 catalog combinations. `ModelArchitecture` owns architecture identity and checked parameter counting. `TrainingModelConfig` adds optimizer and run controls. The versioned `NPRTMODEL1` codec is the durable boundary for preferences and WorkManager; unknown versions, keys, malformed numbers, and invalid combinations fail closed. A queued worker reconstructs its model from that immutable encoding and does not read a later UI selection.

The Model settings sheet is editable only outside active phases. Applying a model persists the next-run choice and re-inspects any selected dataset against the new tokenizer/cache identity. The active run and its progress continue to display the run's immutable configuration.

## Dataset and tokenizer compatibility

V256 requires `NPRTBYTEV1` with T32/V256 and the legacy byte tokenizer. V1024 requires `NPRTBPEV1` with T32/V1024 and the canonical tokenizer SHA-256:

`sha256:9a70e5929e6556a147b0fbc6ada7afefa5e144cdfe2d83bd60e6b31a13252798`

Both wire formats are inspected from actual bytes, including lengths, token ranges, hashes, record ordering, and trailing bytes. The V1024 native loader also requires the separate canonical `byte-bpe-v1024.model`. The Android app does not yet expose a safe app-private import for that artifact, so V1024 Training is shown but Start is explicitly blocked. There is no fallback to V256, another tokenizer, or CPU training.

## Checkpoint and Generation compatibility

V256 byte models use `NPRTCKPTV2`; canonical V1024 byte-BPE models use `NPRTCKPTV3`. Compatibility is determined from the checkpoint header, tokenizer identity, complete parameter/Adam registries, element counts, finite payloads, trailing-byte check, and parameter hash. Filename or metadata alone is not sufficient.

Generation accepts any architecture in the catalog and passes V/T/D/FFN/L/H from the inspected checkpoint through Kotlin and JNI to native validation. For V1024 file-backed Generation, the adjacent canonical tokenizer model must exist and its SHA-256 must match the checkpoint. Unsupported or malformed configurations fail closed; they are never coerced to D32/H2.

The known D64/FFN64/V1024 checkpoint identity has 602,880 parameter elements and is covered by a host-side inspector test and the physical-device health smoke below. Neither result is a Generation quality claim.

## Migration and verification

An install with no model-selection entry keeps D32/FFN32/V256. Exact legacy `compatibilityKey` entries migrate to that configuration; malformed or non-canonical legacy identities fail closed. Existing V2 checkpoint metadata is decoded as D32/V256, while new metadata carries the full versioned configuration. Existing checkpoint files, SAF grants, history, and app data are not deleted.

Targeted verification on 2026-08-21 passed the JVM unit suite, host suite, Fast gate, Kotlin/androidTest compilation, fixed-QAIRT QNN-enabled APK build, and APK ABI/hash/path/2.47 audit. On the connected physical device, the focused Compose test for idle → Model settings → V1024/D64/FFN64 → Apply passed. The production Activity displayed `V1024/T32/D64/FFN64/L19/H2`, `602,880 parameters`, `Head dim 32`, and the explicit V1024 Training block; applying the selection persisted it across Activity recreation. No training run was started.

For the Generation health smoke, a verified D64/FFN64 step-7500 NPRTCKPTV3 checkpoint and its adjacent canonical tokenizer model were staged in app-private storage without replacing existing files. Device hashes matched the host artifacts, and the checkpoint tokenizer hash matched the adjacent model. The production selector classified the checkpoint as compatible and selected the checkpoint-derived V1024/T32/D64/FFN64/L19/H2 identity. One Greedy, 8-byte Generation completed with backend HTP, QNN attempts 36, QNN successes 36, QNN failures 0, CPU fallback `NO`, finite `YES`, and status `SUCCESS`. Generated text is intentionally omitted and no quality conclusion is drawn. Existing D32/FFN32 checkpoints remained selectable; no D48/FFN48 artifact was present for a device-list smoke.
