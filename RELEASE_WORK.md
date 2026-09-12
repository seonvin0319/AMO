# Release implementation record

Understood as: create compact release code for ASPC, wPC(RC), TD3+BC(RC), A2PR, original ReBRAC, IQL, TD3+AMO and IQL+AMO, with one algorithm module per backend and minimal shared configs grounded in the experiment code and recorded runs.

The user supplied AMO branches to resolve exact TD3/IQL settings. Source authority is now TD3 eee3d486, IQL 7068c765, and original ReBRAC 694385b7. See SOURCE_PROVENANCE.md for all references.

Audit findings applied: preserve IQL log-beta float64 and Adam (0,0.999), custom virtual-Adam sqrt derivative, warm-up boundary and post-update beta0 actor/learning rate. Its V-based critic is independent of both actor temperatures. ReBRAC actor updates start at the first critic step; original Q normalization and evaluation cadence are preserved.

Validation independence: backend agreement alone cannot establish correctness because both ports share equations. Pinned source trainers provide an external implementation oracle. Shared batches/weights test local update equivalence; no independent rollout performance result is inferred. Source fixtures retain numerical equations and only adjust imports for the isolated tests.

Consistency audit: removed stale missing-IQL and provisional-ReBRAC statements from README, provenance, validation, configs and modules. Canonical executable defaults are in common/config.py plus the selected YAML. Source provenance links to the exact upstream files. No broader repository consolidation was performed.

Initial validation: 26 small release tests passed. A post-update dtype assertion catches unintended JAX float64 promotion; Adam bias corrections preserve each parameter dtype. The source IQL virtual-Adam derivative is preserved. The cold README review identified CPU/device ambiguity, full-suite dependencies, and the second actor's missing feedback role; documentation states these explicitly.

User-requested validation audit (2026-09-12): the original tests mostly compared the upstream trainer only to PyTorch and compared only losses between backends. Expanded the original-trainer tests to JAX and the backend tests to all training-state tensors. Added full-width original-trainer comparisons with optimizer moments/counters and scale hypergradients. Fixed float32 Adam bias-correction cancellation and used native PyTorch linear/LayerNorm operations. The full-width comparisons still have strict-gate failures, including a measured TD3 ReLU-boundary discrepancy. All failures and raw measurements remain visible in docs/VALIDATION.md; no thresholds were relaxed. No actual D4RL/MuJoCo or amo_log performance reproduction has been completed. Status: release candidate awaiting benchmark reproduction.
