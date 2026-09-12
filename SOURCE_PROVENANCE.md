# Source and configuration provenance

| Source | Pinned revision | Use |
| --- | --- | --- |
| [AMO TD3 branch](https://github.com/seonvin0319/AMO/tree/eee3d486fac0c9f5ecbbea411424e6691b715265) | `eee3d486fac0c9f5ecbbea411424e6691b715265` | TD3+AMO B_pi / L1+L2_RMS, critic options and launcher profiles |
| [AMO IQL branch](https://github.com/seonvin0319/AMO/tree/7068c765918917fc8ac5724120ffff2583baf693) | `7068c765918917fc8ac5724120ffff2583baf693` | IQL+AMO dual B_pi / L1+L2_RMS trainer, helpers and meta config |
| [Original ReBRAC](https://github.com/tinkoff-ai/ReBRAC/tree/694385b72fcef2ae318e3f0ff41ca36122355343) | `694385b72fcef2ae318e3f0ff41ca36122355343` | D4RL ReBRAC update functions, network definitions and all 15 task configs |
| [ASPC](https://github.com/Colin-Jing/ASPC/tree/3119fbefd99ed73134ba26a4db7caf3315cc2250) | `3119fbefd99ed73134ba26a4db7caf3315cc2250` | ASPC, wPC(RC), TD3+BC(RC), IQL |
| [A2PR](https://github.com/ltlhuuu/A2PR/tree/f92a2a4e638c0a72718fbc8c67e7a66030db5f9c) | `f92a2a4e638c0a72718fbc8c67e7a66030db5f9c` | A2PR, matching amo_log/svcho metadata |

Reference functions/classes in `tests/references/` were extracted by Python AST without editing their update equations. Imports were reduced to numerical-test requirements. The TD3 trainer equations in the uploaded branch are identical to AMO a8c1e48, from which its reference fixture was extracted; the branch adds critic construction flags, evaluation repeats and launchers.

## Configuration evidence

[config_sources.json](docs/config_sources.json) records a pinned source per algorithm/task. Recorded run configs and original upstream configs are labeled separately. Server paths, tracking names, resume flags, diagnostic switches and scores do not become defaults.

- TD3+AMO defaults to the completed ext_csv locomotion seeds 1–3 profile: initial `T_E=T_B=1`, `T_lr=0.001`, three critic hidden layers of width 256 with LayerNorm. The corresponding AntMaze profile also appears in the logs.
- `configs/td3_amo_band95.yaml` selects the first-priority values in `scripts/launch_amo_adaptive_multiscale_loco9_tlr3e4_band95_seeds1to3.py` from the uploaded TD3 branch. It uses `T_lr=0.0003`, environment-specific equal initial scales, evaluation every 20,000 steps and five final rounds. It is a locomotion profile, not an AntMaze selection rule.
- IQL+AMO uses the uploaded `meta_amo_bpi_v1.yaml` and recorded beta-init-1, rho-lr-0.002 profile. Environment-specific expectiles and reward transforms remain in the YAML. A CORL base `beta=3` or `10` does not override beta initial 1 when dual mode is enabled.
- ReBRAC uses its original `configs/rebrac/` files directly. All 15 task configs agree with the earlier copied training hyperparameters; original evaluation seed and cadence are also preserved. No MPI-run config is needed to select this requested baseline.
- wPC selects the noise-0.2 RC benchmark settings; noise-0.1 experiments also exist. IQL uses recorded baseline settings. A2PR retains its original four-hidden-layer critic without LayerNorm.

## Preserved implementation details

TD3 uses softplus scales and Adam `(0.9,0.999)` with exponential scale-learning-rate decay. IQL uses log-beta parameters in float64, Adam `(0,0.999)`, constant rho learning rate, projection to `[0.05,100]`, and a 100,000-step warm-up. They are distinct parameterizations and schedules.

IQL's critic target remains V-based. Its beta0 actor does not supply Bellman actions or feed back into the execution actor: the RMS term regularizes target-Q displacement in a parallel auxiliary policy. Identical batches produce identical Q/V trajectories regardless of actor temperatures, and changing beta0 alone leaves execution learning unchanged. See [IQL_AMO.md](docs/IQL_AMO.md) and its reference comparison.

ReBRAC's zero-based loop updates the actor on the first critic step. Its target actor uses the pre-update actor parameters, and BC penalties sum over action coordinates. Its Q normalization has no epsilon floor, as in the original code.

The IQL differentiable Adam square root uses the source's derivative floor at 1e-12 while preserving its exact forward value. This differs from the exact RMS convention, whose value and selected subgradient are zero at the origin.

## Explicit limits and repairs

- ASPC upstream reads `self.reward_mean` without assigning it. The release uses uncentered rewards, equivalent to `reward_mean=0`; the reference test assigns zero explicitly. This does not establish how unrecorded reward-centering experiments were run.
- A2PR keeps the gradient through VAE reconstruction and its boolean advantage-mask addition as logical OR.
- Initializer distributions are preserved. Shared NumPy initial arrays and noise enable backend comparisons, but do not reproduce the legacy framework random bitstreams or old learning curves.
- The endpoint-gradient B_pi proxy is preserved as implemented. Neither the port nor its numerical tests turn it into a global smoothness certificate or monotone-return guarantee.
- The common runner evaluates the execution actor. The uploaded IQL paired comparison between its two actors is not included. Its final repeated seed is recorded explicitly; repeated evaluations are not additional training seeds.
