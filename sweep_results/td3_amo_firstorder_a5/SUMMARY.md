# TD3-AMO first-order · α=5

Updated 2026-09-25 10:09:37 UTC+09:00.

execution_score = first_order (g₀·Δa only, no endpoint-gradient term). π_B L2 RMS, π_E meta L_E, dual actors, critic 3-layer LayerNorm, target π_B, α_E=α_B init 5, reward transform none, seeds 0–3, CPU eval 10×5 at 1M. Final α is the last `metrics.jsonl` α_E / α_B at 1M.

Skipped (already finished on the other host): hopper-medium-replay-v2, walker2d-medium-replay-v2, antmaze-medium-diverse-v2.

Trained 48/48, scored 48/48.

## 1M D4RL normalized score

| env | α_lr | μ (seeds) |
|---|---:|---|
| halfcheetah-medium-v2 | 2e-3 | 58.5 (58.7/57.2/58.6/59.5) |
| halfcheetah-medium-replay-v2 | 2e-3 | 49.3 (49.0/50.0/49.6/48.5) |
| halfcheetah-medium-expert-v2 | 2e-3 | 85.2 (99.7/42.1/103.3/95.6) |
| hopper-medium-v2 | 2e-3 | 102.3 (102.2/102.6/102.5/101.9) |
| hopper-medium-expert-v2 | 2e-3 | 41.5 (31.6/34.0/44.7/55.5) |
| walker2d-medium-v2 | 2e-3 | 79.5 (88.7/94.0/38.5/96.7) |
| walker2d-medium-expert-v2 | 2e-3 | 111.2 (111.2/110.3/112.2/111.1) |
| antmaze-umaze-v2 | 3e-4 | 87.5 (92.0/80.0/92.0/86.0) |
| antmaze-umaze-diverse-v2 | 3e-4 | 85.0 (90.0/82.0/84.0/84.0) |
| antmaze-medium-play-v2 | 3e-4 | 60.5 (52.0/54.0/64.0/72.0) |
| antmaze-large-play-v2 | 3e-4 | 41.5 (62.0/38.0/34.0/32.0) |
| antmaze-large-diverse-v2 | 3e-4 | 39.5 (38.0/40.0/36.0/44.0) |

## Final α_E at 1M

| env | α_lr | μ (seeds) |
|---|---:|---|
| halfcheetah-medium-v2 | 2e-3 | 19.17 (19.44/18.63/19.58/19.04) |
| halfcheetah-medium-replay-v2 | 2e-3 | 18.14 (19.77/16.83/17.75/18.20) |
| halfcheetah-medium-expert-v2 | 2e-3 | 20.31 (20.66/20.17/20.23/20.15) |
| hopper-medium-v2 | 2e-3 | 19.32 (19.19/19.11/19.71/19.27) |
| hopper-medium-expert-v2 | 2e-3 | 20.34 (20.10/20.56/20.28/20.43) |
| walker2d-medium-v2 | 2e-3 | 18.65 (17.51/19.05/19.03/18.99) |
| walker2d-medium-expert-v2 | 2e-3 | 18.95 (18.17/19.19/19.05/19.40) |
| antmaze-umaze-v2 | 3e-4 | 7.77 (7.84/7.78/7.65/7.79) |
| antmaze-umaze-diverse-v2 | 3e-4 | 7.44 (7.35/7.48/7.52/7.43) |
| antmaze-medium-play-v2 | 3e-4 | 7.65 (7.58/7.68/7.72/7.61) |
| antmaze-large-play-v2 | 3e-4 | 7.57 (7.57/7.53/7.58/7.60) |
| antmaze-large-diverse-v2 | 3e-4 | 7.59 (7.60/7.49/7.56/7.69) |

## Final α_B at 1M

| env | α_lr | μ (seeds) |
|---|---:|---|
| halfcheetah-medium-v2 | 2e-3 | 13.50 (13.22/13.60/13.64/13.52) |
| halfcheetah-medium-replay-v2 | 2e-3 | 17.47 (16.76/17.67/17.92/17.53) |
| halfcheetah-medium-expert-v2 | 2e-3 | 14.40 (14.59/14.59/14.20/14.23) |
| hopper-medium-v2 | 2e-3 | 17.02 (17.03/16.79/17.00/17.27) |
| hopper-medium-expert-v2 | 2e-3 | 17.58 (17.92/17.26/17.67/17.49) |
| walker2d-medium-v2 | 2e-3 | 13.25 (13.05/13.51/13.21/13.24) |
| walker2d-medium-expert-v2 | 2e-3 | 12.82 (12.79/13.02/12.87/12.61) |
| antmaze-umaze-v2 | 3e-4 | 6.62 (6.66/6.62/6.58/6.64) |
| antmaze-umaze-diverse-v2 | 3e-4 | 6.32 (6.29/6.30/6.34/6.36) |
| antmaze-medium-play-v2 | 3e-4 | 6.39 (6.40/6.57/6.26/6.33) |
| antmaze-large-play-v2 | 3e-4 | 6.40 (6.49/6.28/6.31/6.52) |
| antmaze-large-diverse-v2 | 3e-4 | 6.48 (6.63/6.36/6.47/6.46) |
