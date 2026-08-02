# Validation-selected depth quality

BEST_VALIDATION_V1 trains all 320 steps and ranks checkpoints only by the
independent ROTATED_LAST_POSITION_V2 validation loss (loss, then accuracy,
then earlier step). FINAL_STEP remains the default and does not evaluate or
restore validation checkpoints.

The validation cases have no full-case or initial-prefix overlap with TRAIN,
Oracle, or the static initial Free cases. They intentionally share 11 learned
token transitions; the two-token rule is excluded because it has no third
distinct phase. FNV-1a identifiers are corruption/determinism checks, not
cryptographic authenticity claims.

CPU screening rejected the candidate. L19 seeds 1 and 4 selected step 320;
seed 2 selected step 128, but held-out Oracle/Free stayed 2/4 rather than
improving over the final step. L18 seed 2 worsened from 3/4 to 2/4. Therefore
HTP smoke and five-seed formal were NOT_RUN_CPU_GATE_REJECTED under the predeclared gate.
The observed validation regression was not reliably predictive of improved
disjoint held-out generation tests.

The direct-seed table exposes canonical initial parameter/Adam, all-step loss
and accuracy, final parameter, and final-logit hashes. Configuration, runner,
APK, path, and device identity remain private provenance; the exact strict
comparison covers all harvested report fields, including generation fields.

Legacy device regression is independently classified from generation quality.
The L6/H8 and L19 seed-1 reports each match all 2248 canonical anchor fields.
L6/H8 retains Oracle/Free 4/4. L19 retains its canonical Oracle/Free 2/4;
its terminal FAILED is published as FINITE_QUALITY_SHORTFALL, not as a
numeric or device regression.

Live progress updates, foreground-service continuity, and the ongoing true-to-
false completion transition passed. Notification tap return and auto-cancel
failed and remain explicitly deferred Android UI issues. They do not block the
direct-seed equivalence result or the CPU-gated validation decision.

Raw checkpoints, parameters, optimizer state, tensor dumps, APK data, device
identifiers, paths, and logs are excluded.
