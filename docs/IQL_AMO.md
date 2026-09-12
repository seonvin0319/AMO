# IQL+AMO implementation contract

Source: [AMO 7068c765](https://github.com/seonvin0319/AMO/tree/7068c765918917fc8ac5724120ffff2583baf693), `algorithms/offline/iql_adaptive_beta.py` and `iql_adaptive_beta_utils.py`. The release implements `amo_dual_bpi=True` for the recorded Gaussian-policy runs.

## Networks and inner objective

The two actors share twin Q functions and one expectile value function. Every network has two hidden layers of width 256, with ReLU and no LayerNorm. Actor means use tanh; trainable state-independent Gaussian log standard deviations are clipped to [-20,2]. Evaluation uses the execution actor's mean.

For either actor, use the pre-update advantage `A = stop(min(Q_target_1,Q_target_2)(s,a_D) - V(s))` and minimize `mean(min(exp(beta*A),100) * negative_log_likelihood(a_D|s))`. The implementation caps the exponent before exponentiation, then caps the resulting weight.

The parameters are `beta_E=exp(rho_E)` and `beta_B=exp(rho_B)`. The source calls the second actor `actor_fixed`; in dual mode its temperature is learned. The release stores it under `bootstrap` to keep the state layout consistent, but it never supplies IQL Bellman actions.

Q/V follow standard IQL: expectile regression of target Q minus V; the Q target is `r + gamma*(1-done)*V(s_next)` using V cached before its update. The critic loss averages the two Q mean-squared errors. Target Q is updated after every critic update.

## Outer losses

The execution virtual actor uses one differentiable Adam update with its current optimizer moments. On an independent outer batch, let `d = pi_plus(s)-pi(s)` and `g0,g1` be detached action gradients of the target twin-Q minimum at the two endpoints. Minimize `-mean(g0·d - 0.5*norm(g1-g0)*norm(d)) / stop(max(mean(abs(Q_target(s,pi_plus(s)))),1e-6))`.

The second actor's virtual step is SGD. Evaluate its virtual actions against the target twin-Q minimum. With `S=stop(mean(abs(Q_target(s,pi_B_plus(s)))))+1e-6`:

- `L1_B = -2*stop(beta_B)*mean(Q_target(s,pi_B_plus(s)))/S + MSE(pi_B_plus(s),a_D)`.
- `delta = gamma*(1-done)*(Q_target(s_next,pi_B_plus(s_next))-stop(Q_target(s_next,pi_B(s_next))))`.
- `L2_RMS_B = 2*stop(beta_B)*sqrt(mean((delta/S)^2))`.

The exact RMS has zero value and a zero subgradient when all entries are zero. Target-Q parameters, both B_pi endpoint gradients, Q normalization and the explicit beta_B common factors are detached. Gradients reach each temperature through its own virtual actor update.

This RMS measures Q change under candidate policy actions. Because the actual IQL target depends on V, it is a surrogate regularizer and does not measure displacement of the target used to train Q.

The beta_B actor is a parallel auxiliary branch: changing only its temperature does not change the execution actor or beta_E updates either. The source has no feedback from this branch to shared Q/V or execution learning. An invariance check covers this dependency with identical batches.

## Timing and numerics

1. Cache advantages and next-state V before updating Q/V.
2. Update V, Q, and target Q once.
3. Update the beta_B actor with real Adam and advance its cosine learning-rate scheduler.
4. On a meta event, form the execution virtual Adam step at the current execution learning rate and update rho_E. The actual execution step uses the cached pre-meta beta_E.
5. Form the beta_B virtual SGD step from its already-updated actor, using its next cosine learning rate, and update rho_B. Do not apply an additional real beta_B actor step.
6. Apply exactly one real execution actor update and advance its cosine schedule.

The original applies the real execution update between the two scale updates; the release applies it after both. The B branch has no dependency on the execution actor or its optimizer, so these operations commute. Full parameter comparisons cover this ordering.

Both rho parameters and their Adam moments use float64; model computation and beta multiplication use float32. Rho Adam uses learning rate 0.002, betas (0,0.999), epsilon 1e-8, and no weight decay or learning-rate scheduler. After each rho update, clamp it to [log(0.05),log(100)] without resetting moments. Actor/Q/V Adam uses learning rate 0.0003 and betas (0.9,0.999); actor learning rates follow cosine decay over one million updates.

The default meta condition is `step>100000 and (step-100000)%20==0`. The independent outer batch has 256 transitions. Both beta values start at 1 in the selected profile. [config_sources.json](config_sources.json) records the environment overrides and source runs.
