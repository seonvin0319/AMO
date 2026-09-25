# TD3 RAPO hopper-walker α_lr=1e-4

Updated 2026-09-25 10:09:37 UTC+09:00.

π_E −B_π, π_B L2 RMS, critic 2-layer no LayerNorm, target π_E, α_E=α_B=5, α_lr=1e-4, seeds 0–3, CPU eval 10×5 at 1M.

Trained 24/24, scored 24/24.

| env | 1e-4 μ (seeds) |
|---|---|
| hopper-medium-v2 | 62.3 (62.6/73.7/57.6/55.4) |
| hopper-medium-replay-v2 | 51.8 (39.1/44.0/50.6/73.5) |
| hopper-medium-expert-v2 | 91.9 (109.4/99.5/109.1/49.6) |
| walker2d-medium-v2 | 10.9 (11.8/11.9/10.6/9.4) |
| walker2d-medium-replay-v2 | 70.0 (21.6/91.0/85.9/81.5) |
| walker2d-medium-expert-v2 | 70.0 (54.4/111.4/109.1/5.0) |
