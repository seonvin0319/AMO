# TD3+AMO equations and update order

Let `sg` mean stop-gradient. MSE averages over both batch and action coordinates. Both actors have dataset action a_D as their reference. T_i=softplus(rho_i)>0 for i in {E,B}. The default actor and scale optimizers are Adam with betas (0.9,0.999) and epsilon 1e-8.

## Inner actor objective

For either actor, using the online first critic:

\[
S(\theta)=\operatorname{sg}\!\left(\max\{\mathbb E|Q_1(s,\pi_\theta(s))|,10^{-6}\}\right),\qquad
\ell(\theta,T)=-\frac{\mathbb E Q_1(s,\pi_\theta(s))}{S(\theta)}+\frac{\operatorname{MSE}(\pi_\theta(s),a_D)}{2T}.
\]

Actual actor updates use Adam with lr=3e-4. The inner batch and independently sampled outer batch are separate draws; accidental overlap of individual transitions is allowed.

## Execution scale

The virtual execution parameters use one differentiable Adam step from the current actor parameters and current Adam moments. Moments are treated as constants entering the virtual step; the new gradient and moment calculations remain differentiable in T_E.

On the outer batch let a_0=pi_E(s), a_+=pi_E^+(s), d=a_+-a_0, and

\[
g_0=\operatorname{sg}(\nabla_a\bar Q_1(s,a_0)),\qquad
g_+=\operatorname{sg}(\nabla_a\bar Q_1(s,a_+)),\qquad
\widehat B_\pi=g_0^\top d-\tfrac12\|g_+-g_0\|_2\|d\|_2.
\]

\[
\mathcal L_{T_E}=-\frac{\mathbb E\widehat B_\pi}
{\operatorname{sg}(\max\{\mathbb E|\bar Q_1(s,a_+)|,10^{-6}\})}.
\]

Target-critic parameters and endpoint gradients are frozen. Only the displacement from the virtual actor update carries the scale hypergradient. This is the source's endpoint-gradient estimate; it is not a certified lower bound without additional assumptions.

## Bootstrap scale

The virtual bootstrap update is SGD, even though the actual actor optimizer is Adam:

\[
\theta_B^+=\theta_B-3\cdot10^{-4}\nabla_{\theta_B}\ell(\theta_B,T_B).
\]

Let bar_q(s,a)=min_j bar_Q_j(s,a), a_+=pi_B^+(s), and

\[
S_B=\operatorname{sg}(\mathbb E|\bar q(s,a_+)|)+10^{-6},\qquad
\Delta y_B=\gamma(1-d_{\rm terminal})\left[\bar q(s',\pi_B^+(s'))-\operatorname{sg}(\bar q(s',\pi_B(s')))\right].
\]

\[
L_{1,B}=-2\operatorname{sg}(T_B)\frac{\mathbb E\bar q(s,a_+)}{S_B}+\operatorname{MSE}(a_+,a_D),
\qquad L_{2,\mathrm{RMS},B}=2\operatorname{sg}(T_B)\sqrt{\mathbb E(\Delta y_B/S_B)^2},
\qquad\mathcal L_{T_B}=L_{1,B}+L_{2,\mathrm{RMS},B}.
\]

Both action branches are deterministic and use the same frozen target critics. The old action is the current online bootstrap actor, not its target copy. The full-batch RMS includes terminal transitions as zeros. There is no additive epsilon inside the RMS; its value and chosen subgradient are zero when every input is zero.

## Per-iteration order

1. Update both critics using r+gamma(1-terminal) min_j bar_Q_j(s',clip(bar_pi_B(s')+noise)). Noise standard deviation is 0.2, clipped to [-0.5,0.5]; actions are clipped to [-1,1].
2. Every two critic iterations, cache T_E for the execution actor's actual update.
3. Every 20 critic iterations, evaluate virtual updates on an independently sampled outer batch and update rho_E, then rho_B. Scale learning rates decay exponentially to 0.01 times their initial value over one million critic iterations.
4. Update bootstrap and execution actors with their real Adam optimizers. Use updated T_B and cached T_E, respectively.
5. Update target bootstrap actor and target critics with tau=0.005. An execution target actor is unnecessary for the released evaluation policy and is omitted.

No sequential-hop bank, scale ordering constraint or extra critic is introduced. [Source revision and numerical comparison](../SOURCE_PROVENANCE.md) define the implementation contract.
