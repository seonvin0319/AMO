# IQL-AMO Q-weight (`iql_amo_qweight`)

New JAX extension: keep IQL’s **data-action expectile V** and **V-based Bellman
target**, but let a bootstrap actor \(π_B\) reweight the Q regression via a
frozen behavior density \(\hatμ\). Optional AMO log-β adaptation for \(β_E\)
(execution) and/or \(β_B\) (bootstrap).

This is **not** TD3-AMO and **not** a verified improvement over IQL. Clipped /
batch-normalized density ratios are **not** exact importance sampling; they do
**not** correct the state visitation distribution.

## Design vs IQL / IQL-AMO

| Piece | IQL | IQL-AMO | `iql_amo_qweight` |
|-------|-----|---------|-------------------|
| V | expectile on \(Q_{\mathrm{tgt}}(s,a_D)-V(s)\) | same | same (fixed expectile; no adaptive τ) |
| Q target | \(r+γV(s')\) | same | same |
| Q loss | unweighted MSE | unweighted MSE | **weighted** MSE with \(w∝π_B/\hatμ\) on data actions |
| Actors | one AWR | \(π_E\) + \(π_B\) | same dual actors |
| \(β_B\) outer | L1 + L2_RMS on policy actions | — | **unweighted TD MSE** of virtual critic |
| \(\hatμ\) | — | — | BC Gaussian, then **frozen** |

\(π_B\) never injects generated / mean actions into V or Q targets.

## Density and weights

Same Gaussian as IQL AWR NLL: mean \(=\tanh(\mathrm{MLP}(s))\), shared
\(\logσ∈[-20,2]\), no tanh Jacobian. Then

\[
\log r_i=\mathrm{clip}\big(\logπ_B(a_i|s_i)-\mathrm{sg}(\log\hatμ(a_i|s_i)),\log w_{\min},\log w_{\max}\big),
\quad
w_i=\frac{e^{\log r_i}}{\mathrm{mean}_j e^{\log r_j}}.
\]

Default (untuned): \(w_{\min}=0.1\), \(w_{\max}=10\). Real Q uses \(\mathrm{sg}(w)\).
Virtual critic path does **not** stop-grad \(w_+\) (batch mean stays in the
\(β_B\) graph).

## Update order

1. Cache \(q_D\), adv, \(y_{\mathrm{inner}}\), \(y_{\mathrm{outer}}\) from pre-update nets  
2. Real V  
3. Real \(π_B\) Adam (cached \(β_B\))  
4. Snapshot post-\(π_B\) and pre-Q critic/Adam  
5. Real Q with current-\(π_B\) weights + target critic  
6. Meta: virtual SGD \(π_B^+\) → \(w_+\) → VirtualAdam critic → outer TD MSE → \(ρ_B\); optional execution outer for \(ρ_E\)  
7. Real \(π_E\) Adam (cached \(β_E\))

Virtual params/opt states are never committed. New \(β_B\) applies on the **next**
real bootstrap step.

## Behavior BC

```bash
python scripts/train_iql_amo_qweight_behavior.py \
  --env hopper-medium-expert-v2 --backend jax --device cuda:0 \
  --steps 50000 --output /raid/ext_csv/AMO_store/iql_amo_qweight_behavior_hme
```

Then pass `--behavior-path .../behavior.npz` to `train.py`. BC cost is logged in
`behavior_meta.json` separately from RL.

## Flags

- `qweight_enabled` — if false, \(w≡1\) (IQL Q)  
- `adapt_beta_E` / `adapt_beta_B` — scale updates  
- `rho_E_lr` / `rho_B_lr` — default `rho_lr`

## Diagnostic contrast (30k, hopper-medium-expert, seed 0)

Shared behavior ckpt; \(β_E=5\) fixed; same inits.

- **A** `qweight_enabled=false`, no adapts  
- **B** qweight on, `beta_initial=5`, no adapts  
- **C** qweight on, `adapt_beta_B=true`, warmup 2k, interval 20, `rho_B_lr=2e-3`

Meta settings above are **initial diagnostic** values, not tuned.
