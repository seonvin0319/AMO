"""Regression tests for AMO's fixed and adaptive multiscale modes."""
import contextlib
import copy
import gc
import importlib.util
import io
import os
import weakref

import pytest
import torch
import torch.nn as nn
import torchopt

import amo


class Q(nn.Module):
    def __init__(self, state_dim=3, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 16), nn.Tanh(), nn.Linear(16, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


class SmoothActor(nn.Module):
    def __init__(self, state_dim=3, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 8),
            nn.Tanh(),
            nn.Linear(8, action_dim),
            nn.Tanh(),
        )
        self.max_action = 1.0

    def forward(self, state):
        return self.net(state)


def batch(seed, n=8):
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randn(n, 3, generator=generator),
        torch.randn(n, 2, generator=generator).tanh(),
        torch.randn(n, 1, generator=generator),
        torch.randn(n, 3, generator=generator),
        torch.zeros(n, 1),
    ]


def make_trainer(n=2, tau=0.0):
    actor = amo.Actor(3, 2, 1.0)
    q1, q2, vnet = Q(), Q(), amo.Vnet(3)
    return amo.AMO(
        1.0,
        actor,
        torchopt.adam(lr=3e-4, use_accelerated_op=False),
        q1,
        torch.optim.Adam(q1.parameters(), lr=3e-4),
        q2,
        torch.optim.Adam(q2.parameters(), lr=3e-4),
        vnet,
        torch.optim.Adam(vnet.parameters(), lr=3e-4),
        T=1.2,
        T_freq=1,
        policy_freq=1,
        tau=tau,
        proximal_n_steps=n,
        device="cpu",
    )


def make_adaptive_trainer(
    *,
    seed=0,
    tau=0.0,
    T_E=1.2,
    T_B=None,
    T_lr=2e-4,
    actor_lr=3e-4,
    bootstrap_rms_diagnostic_freq=0,
):
    torch.manual_seed(seed)
    actor = SmoothActor()
    q1, q2, vnet = Q(), Q(), amo.Vnet(3)
    return amo.AMO(
        1.0,
        actor,
        torchopt.adam(lr=actor_lr, use_accelerated_op=False),
        q1,
        torch.optim.Adam(q1.parameters(), lr=3e-4),
        q2,
        torch.optim.Adam(q2.parameters(), lr=3e-4),
        vnet,
        torch.optim.Adam(vnet.parameters(), lr=3e-4),
        T=T_E,
        T_B=T_B,
        T_freq=1,
        T_lr=T_lr,
        policy_freq=1,
        tau=tau,
        adaptive_multiscale=True,
        bootstrap_rms_diagnostic_freq=bootstrap_rms_diagnostic_freq,
        actor_lr=actor_lr,
        device="cpu",
    )


def module_values(modules):
    return [
        parameter.detach().clone()
        for module in modules
        for parameter in module.parameters()
    ]


def assert_module_values(modules, expected):
    actual = [parameter for module in modules for parameter in module.parameters()]
    assert all(torch.equal(value, before) for value, before in zip(actual, expected))


def test_inner_loss_matches_T_notation():
    trainer, data = make_trainer(), batch(0)
    state, action = data[:2]
    T = torch.tensor(1.2)
    pi = trainer.actor(state)
    loss, q_abs_mean = trainer._inner_loss(state, pi, action, T)
    expected = -trainer.critic_1(
        state, pi
    ).mean() / q_abs_mean + torch.nn.functional.mse_loss(pi, action) / (2 * T)
    assert torch.allclose(loss, expected)


def test_outer_updates_T_and_keeps_target_frozen():
    trainer = make_trainer()
    target_before = module_values([trainer.critic_1_target])
    T_before = torch.nn.functional.softplus(trainer.log_T).item()
    logs = trainer.train(batch(1), batch(2))
    assert "amo/outer_loss" in logs and "amo/T_grad" in logs
    assert torch.nn.functional.softplus(trainer.log_T).item() != T_before
    assert_module_values([trainer.critic_1_target], target_before)


def test_fixed_chain_routing_is_unchanged():
    chain = make_trainer(n=3)
    logs = chain.train(batch(3), batch(4))
    assert logs["amo/N"] == 3.0
    assert abs(logs["amo/T"] - 3 * logs["amo/tau"]) < 1e-6
    assert logs["amo/outer_N_scale"] == 3.0
    assert len(make_trainer(n=1).actor_bank) == 1
    assert len(make_trainer(n=3).actor_bank) == 3


def test_adaptive_option_initialization_and_old_method_removed():
    fields = amo.TrainConfig.__dataclass_fields__
    assert fields["adaptive_multiscale"].default is False
    assert fields["T_E"].default == 1.25
    assert fields["T_B"].default is None
    assert "T" not in fields
    assert fields["bootstrap_outer_loss_version"].init is False
    assert fields["bootstrap_outer_loss_version"].default == (
        amo.BOOTSTRAP_OUTER_LOSS_VERSION
    )
    for old_name in (
        "adaptive_" + "bootstrap_scale",
        "bootstrap_" + "stability_budget",
        "dual_" + "proximal",
    ):
        assert old_name not in fields
    try:
        amo.TrainConfig(adaptive_multiscale=True, proximal_n_steps=2)
    except ValueError as exc:
        assert "cannot be combined" in str(exc)
    else:
        raise AssertionError("adaptive multiscale accepted actor-chain N")

    trainer = make_adaptive_trainer()
    T_E = torch.nn.functional.softplus(trainer.log_T).item()
    assert trainer.bootstrap_scale().item() > 0
    assert trainer.bootstrap_scale().item() == T_E
    assert torch.equal(trainer.log_T_B, trainer.log_T)
    assert trainer.T_optimizer is not trainer.T_B_optimizer
    assert trainer.log_T is not trainer.log_T_B
    assert (
        trainer.T_optimizer.param_groups[0]["lr"]
        == trainer.T_B_optimizer.param_groups[0]["lr"]
    )
    assert trainer.actor_bank_states[0] is not trainer.actor_bank_states[1]

    source = open(amo.algorithm.__file__, encoding="utf-8").read()
    for symbol in (
        "bootstrap_" + "stability_budget",
        "R_" + "val",
        "V_" + "plus",
        "V_" + "zero",
        "lambda_" + "B",
        "virtual_" + "critic",
        "batch_meta_" + "train",
        "batch_meta_" + "val",
        "bootstrap_init_" + "ratio",
        "initial_scale_" + "ratio",
        "T_B_init_" + "divisor",
        "bootstrap_scale_" + "factor",
    ):
        assert symbol not in source


def test_te_tb_flags_initialize_scales_independently():
    try:
        amo.TrainConfig(T_B=1.0)
    except ValueError as exc:
        assert "adaptive_multiscale" in str(exc)
    else:
        raise AssertionError("T_B accepted without adaptive_multiscale")
    try:
        amo.TrainConfig(adaptive_multiscale=True, T_B=-1.0)
    except ValueError as exc:
        assert "T_B" in str(exc)
    else:
        raise AssertionError("negative T_B accepted")
    try:
        amo.TrainConfig(T_E=-1.0)
    except ValueError as exc:
        assert "T_E" in str(exc)
    else:
        raise AssertionError("negative T_E accepted")

    trainer = make_adaptive_trainer(T_E=10.0, T_B=1.0)
    t_e = torch.nn.functional.softplus(trainer.log_T).item()
    t_b = trainer.bootstrap_scale().item()
    assert abs(t_e - 10.0) < 1e-5
    assert abs(t_b - 1.0) < 1e-5
    assert not torch.equal(trainer.log_T_B, trainer.log_T)

    copied = make_adaptive_trainer(T_E=5.0, T_B=5.0)
    assert torch.equal(copied.log_T_B, copied.log_T)
    amo.TrainConfig(adaptive_multiscale=True, T_E=10.0, T_B=1.0)


def test_adaptive_false_trace_and_v5_checkpoint_surface_are_unchanged():
    torch.manual_seed(17)
    first = make_trainer(n=4)
    second = make_trainer(n=4)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    torch.manual_seed(91)
    logs_first = first.train(batch(20), batch(21))
    torch.manual_seed(91)
    logs_second = second.train(batch(20), batch(21))
    assert logs_first.keys() == logs_second.keys()
    for key in logs_first:
        left, right = logs_first[key], logs_second[key]
        assert left == right or (
            torch.isnan(torch.tensor(left)) and torch.isnan(torch.tensor(right))
        ), key
    assert_module_values(second.actor_bank, module_values(first.actor_bank))
    assert set(first.state_dict()) == {
        "checkpoint_version",
        "actor",
        "actor_bank",
        "actor_bank_states",
        "actor_target",
        "critic_1",
        "critic_2",
        "critic_1_target",
        "critic_2_target",
        "critic_1_optimizer",
        "critic_2_optimizer",
        "log_T",
        "T_optimizer",
        "T_scheduler",
        "total_it",
    }
    assert first.state_dict()["checkpoint_version"] == 5


def test_equal_initial_scales_and_first_outer_update_is_loss_driven():
    trainer = make_adaptive_trainer(actor_lr=3e-3)
    log_T_before = trainer.log_T.detach().item()
    log_T_B_before = trainer.log_T_B.detach().item()
    assert (
        trainer.bootstrap_scale().item()
        == torch.nn.functional.softplus(trainer.log_T).item()
    )
    logs = trainer.train(batch(22), batch(23))
    assert logs["amo/grad_T_E"] != 0
    assert logs["amo/grad_T_E_BPI"] == logs["amo/grad_T_E"]
    assert logs["amo/grad_T_B_total"] != 0
    assert logs["amo/delta_log_T_E"] == trainer.log_T.detach().item() - log_T_before
    assert (
        abs(
            logs["amo/delta_log_T_B"]
            - (trainer.log_T_B.detach().item() - log_T_B_before)
        )
        < 1e-6
    )
    assert logs["amo/bootstrap_outer_loss_version"] == (
        amo.BOOTSTRAP_OUTER_LOSS_VERSION
    )
    assert "amo/N" not in logs and "amo/outer_N_scale" not in logs
    assert abs(logs["amo/T_B_over_T_E"] - logs["amo/T_B"] / logs["amo/T_E"]) < 1e-7


def test_detached_tq_l1_and_exact_rms_target_formula():
    trainer = make_adaptive_trainer(actor_lr=3e-3)
    outer_batch = batch(31)
    outer_batch[4][::2] = 1.0
    outer_loss, details = trainer._bootstrap_multiscale_objective(
        batch(30), outer_batch, trainer.bootstrap_scale()
    )
    q_virtual = torch.minimum(
        trainer.critic_1_target(outer_batch[0], details["virtual_action"]),
        trainer.critic_2_target(outer_batch[0], details["virtual_action"]),
    )
    expected_l1_q = (
        -2
        * trainer.bootstrap_scale().detach()
        * q_virtual.mean()
        / details["q_scale_B"]
    )
    expected_bc = torch.nn.functional.mse_loss(
        details["virtual_action"], outer_batch[1]
    )
    expected_delta_y = trainer.discount * (1.0 - outer_batch[4]) * details["delta_Q_B"]
    expected_rms = (
        torch.linalg.vector_norm(expected_delta_y / details["q_scale_B"])
        / expected_delta_y.numel() ** 0.5
    )
    expected_l2 = 2 * trainer.bootstrap_scale().detach() * expected_rms
    assert torch.allclose(details["L1_B_Q"], expected_l1_q)
    assert torch.allclose(details["L1_B_BC"], expected_bc)
    assert torch.allclose(details["L1_B"], expected_l1_q + expected_bc)
    assert torch.allclose(details["delta_y_B"], expected_delta_y)
    assert torch.allclose(details["R_B"], expected_rms)
    assert torch.allclose(details["L2_RMS_B"], expected_l2)
    assert torch.allclose(outer_loss, details["L1_B"] + expected_l2)
    assert not details["q_scale_B"].requires_grad
    assert torch.isfinite(expected_l2) and expected_l2 >= 0
    assert details["bootstrap_outer_loss_version"] == (amo.BOOTSTRAP_OUTER_LOSS_VERSION)


def test_exact_rms_zero_terminal_mixed_sign_and_rescaling():
    values = torch.zeros(8, 1, requires_grad=True)
    zero_rms = amo.AMO._exact_rms(values)
    zero_grad = torch.autograd.grad(zero_rms, values)[0]
    assert zero_rms == 0.0
    assert torch.all(torch.isfinite(zero_grad))
    assert torch.equal(zero_grad, torch.zeros_like(zero_grad))

    synthetic = torch.tensor([[3.0], [-4.0]])
    assert torch.allclose(
        amo.AMO._exact_rms(synthetic),
        torch.sqrt(synthetic.square().mean()),
    )
    assert amo.AMO._exact_rms(synthetic) > 0
    assert synthetic.mean().abs() < amo.AMO._exact_rms(synthetic)

    delta_q = torch.randn(8, 1, requires_grad=True)
    all_done = torch.ones_like(delta_q)
    terminal = amo.AMO._bootstrap_target_rms(delta_q, all_done, torch.tensor(2.0), 0.99)
    terminal_grad = torch.autograd.grad(terminal["R_B"], delta_q)[0]
    assert terminal["R_B"] == 0.0
    assert torch.equal(terminal_grad, torch.zeros_like(terminal_grad))
    assert torch.isfinite(terminal_grad).all()

    done = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    delta = torch.tensor([[1.0], [9.0], [-2.0], [3.0]])
    base = amo.AMO._bootstrap_target_rms(delta, done, torch.tensor(4.0), 0.9)
    scaled = amo.AMO._bootstrap_target_rms(7.0 * delta, done, torch.tensor(28.0), 0.9)
    manual = torch.sqrt(((0.9 * (1.0 - done) * delta / 4.0).square()).mean())
    assert torch.allclose(base["R_B"], manual)
    assert torch.allclose(base["R_B"], scaled["R_B"])
    assert base["nonterminal_fraction"] == 0.75


def test_tq_coefficients_are_detached_and_inverse_gradient_equivalence():
    rho = torch.tensor(0.3, requires_grad=True)
    T_B = torch.nn.functional.softplus(rho)
    independent_path = torch.tensor(0.7, requires_grad=True)
    independent_terms = amo.AMO._tq_scaled_bootstrap_terms(
        T_B,
        independent_path,
        independent_path.square(),
        independent_path.abs(),
    )
    for key in ("L1_B_Q", "L2_RMS_B", "L_T_B"):
        explicit_grad = torch.autograd.grad(
            independent_terms[key], rho, retain_graph=True, allow_unused=True
        )[0]
        assert explicit_grad is None

    normalized_q = rho.square() + 0.2
    bc = (rho - 0.4).square() + 0.1
    R_B = rho.exp() * 0.05
    terms = amo.AMO._tq_scaled_bootstrap_terms(T_B, normalized_q, bc, R_B)
    assert torch.allclose(terms["L_T_B"], 2.0 * T_B.detach() * terms["L_T_B_inverse"])
    grad_scaled = torch.autograd.grad(terms["L_T_B"], rho, retain_graph=True)[0]
    grad_inverse = torch.autograd.grad(terms["L_T_B_inverse"], rho)[0]
    assert torch.allclose(grad_scaled, 2.0 * T_B.detach() * grad_inverse)


def test_old_squared_and_new_rms_relation_and_gradient_direction():
    rho = torch.tensor(0.4, requires_grad=True)
    T_B = torch.nn.functional.softplus(rho)
    done = torch.tensor([[0.0], [1.0], [0.0], [0.0]])
    pattern = torch.tensor([[1.0], [-4.0], [-2.0], [3.0]])
    delta_q = rho * pattern
    q_scale = torch.tensor(2.5)
    gamma = 0.99
    old = ((1.0 - done) * delta_q.square()).mean() / q_scale.square()
    target = amo.AMO._bootstrap_target_rms(delta_q, done, q_scale, gamma)
    new = 2.0 * T_B.detach() * target["R_B"]
    assert torch.allclose(new, 2.0 * T_B.detach() * gamma * torch.sqrt(old))
    grad_old = torch.autograd.grad(old, rho, retain_graph=True)[0]
    grad_new = torch.autograd.grad(new, rho)[0]
    assert grad_old * grad_new > 0


def test_zero_virtual_displacement_has_zero_rms_loss_and_gradient():
    same = make_adaptive_trainer(actor_lr=0.0)
    _, same_details = same._bootstrap_multiscale_objective(
        batch(32), batch(33), same.bootstrap_scale()
    )
    grad = torch.autograd.grad(same_details["L2_RMS_B"], same.log_T_B)[0]
    assert torch.equal(
        same_details["delta_Q_B"], torch.zeros_like(same_details["delta_Q_B"])
    )
    assert same_details["R_B"] == 0.0
    assert same_details["L2_RMS_B"] == 0.0
    assert torch.isfinite(grad) and grad == 0.0


def test_gradient_isolation_detach_and_functional_virtual_update():
    trainer = make_adaptive_trainer(actor_lr=3e-3)
    inner, outer = batch(40), batch(41)
    tracked = [
        *trainer.actor_bank,
        trainer.critic_1,
        trainer.critic_2,
        trainer.critic_1_target,
        trainer.critic_2_target,
    ]
    before = module_values(tracked)
    for module in tracked:
        for parameter in module.parameters():
            parameter.grad = None

    T_E = torch.nn.functional.softplus(trainer.log_T)
    pi, pi_new = trainer._virtual_actor_bank_update(inner[0], inner[1], outer[0], T_E)
    execution_loss, _ = trainer._outer_loss(outer[0], pi, pi_new)
    grad_E, grad_B = torch.autograd.grad(
        execution_loss, (trainer.log_T, trainer.log_T_B), allow_unused=True
    )
    assert grad_E is not None and torch.isfinite(grad_E) and grad_E != 0
    assert grad_B is None

    bootstrap_loss, details = trainer._bootstrap_multiscale_objective(
        inner, outer, trainer.bootstrap_scale()
    )
    grad_B, grad_E = torch.autograd.grad(
        bootstrap_loss,
        (trainer.log_T_B, trainer.log_T),
        retain_graph=True,
        allow_unused=True,
    )
    grad_l1 = torch.autograd.grad(details["L1_B"], trainer.log_T_B, retain_graph=True)[
        0
    ]
    grad_l1_q = torch.autograd.grad(
        details["L1_B_Q"], trainer.log_T_B, retain_graph=True
    )[0]
    grad_l1_bc = torch.autograd.grad(
        details["L1_B_BC"], trainer.log_T_B, retain_graph=True
    )[0]
    grad_l2 = torch.autograd.grad(details["L2_RMS_B"], trainer.log_T_B)[0]
    assert grad_B is not None and torch.isfinite(grad_B) and grad_B != 0
    assert grad_E is None
    assert torch.isfinite(grad_l1) and grad_l1 != 0
    assert torch.allclose(grad_l1, grad_l1_q + grad_l1_bc)
    assert torch.isfinite(grad_l2) and grad_l2 != 0
    assert not details["Qbar_B_old"].requires_grad
    assert not details["current_next_action"].requires_grad
    assert details["Qbar_B_new"].requires_grad
    assert details["virtual_next_action"].requires_grad
    assert_module_values(tracked, before)
    for target in (trainer.critic_1_target, trainer.critic_2_target):
        assert all(parameter.grad is None for parameter in target.parameters())


def test_dataset_anchors_and_polyak_role_targets():
    trainer = make_adaptive_trainer(tau=1.0)
    data = batch(50)
    references = []
    original = trainer._inner_loss

    def recording_inner(state, pi, reference, T):
        references.append(reference.detach().clone())
        return original(state, pi, reference, T)

    trainer._inner_loss = recording_inner
    trainer._update_actor_bank(
        data[0],
        data[1],
        torch.nn.functional.softplus(trainer.log_T),
        trainer.bootstrap_scale().detach(),
    )
    assert len(references) == 2
    assert all(torch.equal(reference, data[1]) for reference in references)

    trainer.train(batch(51), batch(52))
    for target, source in zip(
        trainer.actor_target.parameters(), trainer.critic_bootstrap_actor().parameters()
    ):
        assert torch.equal(target, source)
    for target, source in zip(
        trainer.execution_actor_target.parameters(),
        trainer.execution_actor().parameters(),
    ):
        assert torch.equal(target, source)
    assert trainer.execution_actor() is trainer.actor_bank[1]
    assert trainer.critic_bootstrap_actor() is trainer.actor_bank[0]


def test_independent_scale_updates_and_checkpoint_round_trip():
    trainer = make_adaptive_trainer(T_lr=1e-2, actor_lr=3e-3)
    T_E = torch.nn.functional.softplus(trainer.log_T).detach()

    trainer.T_optimizer.zero_grad(set_to_none=True)
    trainer.T_B_optimizer.zero_grad(set_to_none=True)
    trainer.log_T.grad = torch.ones_like(trainer.log_T)
    trainer.log_T_B.grad = -torch.ones_like(trainer.log_T_B)
    trainer.T_optimizer.step()
    trainer.T_B_optimizer.step()
    T_E_after = torch.nn.functional.softplus(trainer.log_T).item()
    T_B_after = trainer.bootstrap_scale().item()
    assert T_E_after != T_B_after
    assert T_B_after > T_E_after

    with torch.no_grad():
        trainer.log_T_B.copy_(amo.softplus_inverse(2 * T_E))
    assert trainer.bootstrap_scale().item() > T_E.item()

    with torch.no_grad():
        trainer.log_T_B.copy_(amo.softplus_inverse(T_E - 0.1))
    T_B_before = trainer.bootstrap_scale().item()
    with torch.no_grad():
        trainer.log_T.copy_(amo.softplus_inverse(2 * T_E))
    assert trainer.bootstrap_scale().item() == T_B_before

    trainer._update_bootstrap_scale(batch(60), batch(61), trainer.bootstrap_scale())
    saved = copy.deepcopy(trainer.state_dict())
    restored = make_adaptive_trainer(T_lr=1e-2, actor_lr=3e-3)
    restored.load_state_dict(saved)
    assert saved["checkpoint_version"] == 10
    assert saved["bootstrap_outer_loss_version"] == (amo.BOOTSTRAP_OUTER_LOSS_VERSION)
    assert not any("initialization_n" in key for key in saved)
    assert torch.equal(restored.log_T, trainer.log_T)
    assert torch.equal(restored.log_T_B, trainer.log_T_B)
    assert restored.bootstrap_outer_update_count == trainer.bootstrap_outer_update_count
    assert (
        restored.T_optimizer.state_dict()["param_groups"]
        == saved["T_optimizer"]["param_groups"]
    )
    assert (
        restored.T_B_optimizer.state_dict()["param_groups"]
        == saved["T_B_optimizer"]["param_groups"]
    )
    for key, value in restored.T_B_optimizer.state_dict()["state"][0].items():
        expected = saved["T_B_optimizer"]["state"][0][key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, expected)
        else:
            assert value == expected
    for restored_value, expected_value in zip(
        restored.execution_actor_target.state_dict().values(),
        saved["execution_actor_target"].values(),
    ):
        assert torch.equal(restored_value, expected_value)

    old_loss = copy.deepcopy(saved)
    old_loss["checkpoint_version"] = 8
    old_loss.pop("bootstrap_outer_loss_version")
    rejected = make_adaptive_trainer(T_lr=1e-2, actor_lr=3e-3)
    before_rejection = module_values(
        [*rejected.actor_bank, rejected.critic_1, rejected.critic_2]
    )
    with pytest.raises(ValueError, match="outer-loss mismatch"):
        rejected.load_state_dict(old_loss)
    assert_module_values(
        [*rejected.actor_bank, rejected.critic_1, rejected.critic_2],
        before_rejection,
    )

    old_scale_state = copy.deepcopy(saved)
    old_scale_state["checkpoint_version"] = 9
    rejected = make_adaptive_trainer(T_lr=1e-2, actor_lr=3e-3)
    with pytest.raises(ValueError, match="checkpoint v10"):
        rejected.load_state_dict(old_scale_state)


def test_bootstrap_scale_can_move_above_execution_scale():
    trainer = make_adaptive_trainer(T_lr=1e-2, actor_lr=3e-3)
    T_E = torch.nn.functional.softplus(trainer.log_T).detach()
    with torch.no_grad():
        trainer.log_T_B.copy_(amo.softplus_inverse(2 * T_E))
    T_B = trainer.bootstrap_scale().item()
    assert T_B > T_E.item()
    logs = trainer._bootstrap_scale_logs()
    assert logs["amo/T_B"] == T_B
    assert logs["amo/T_B_over_T_E"] > 1.0


def test_gradient_sum_direction_reproducibility_nonfinite_and_graph_release():
    first = make_adaptive_trainer(seed=7, T_lr=1e-2, actor_lr=3e-3)
    second = make_adaptive_trainer(seed=7, T_lr=1e-2, actor_lr=3e-3)
    first._update_bootstrap_scale(batch(70), batch(71), first.bootstrap_scale())
    second._update_bootstrap_scale(batch(70), batch(71), second.bootstrap_scale())
    assert torch.equal(first.log_T_B, second.log_T_B)
    assert first._bootstrap_last == second._bootstrap_last
    assert (
        abs(
            first._bootstrap_last["grad_T_B_total"]
            - first._bootstrap_last["grad_T_B_from_L1"]
            - first._bootstrap_last["grad_T_B_from_L2_RMS"]
        )
        < 1e-7
    )
    assert (
        abs(
            first._bootstrap_last["grad_T_B_from_L1"]
            - first._bootstrap_last["grad_T_B_from_L1_Q_chain"]
            - first._bootstrap_last["grad_T_B_from_L1_BC_chain"]
        )
        < 1e-7
    )

    small = first._exact_rms(torch.tensor([[0.1], [-0.1]]))
    large = first._exact_rms(torch.tensor([[0.5], [-0.5]]))
    assert large > small > 0

    tracked = [*first.actor_bank, first.critic_1, first.critic_2]
    before = module_values(tracked)
    refs = []
    for index in range(3):
        loss, details = first._bootstrap_multiscale_objective(
            batch(80 + index), batch(90 + index), first.bootstrap_scale()
        )
        grad = torch.autograd.grad(loss, first.log_T_B)[0]
        refs.append(weakref.ref(loss))
        del loss, details, grad
    gc.collect()
    assert all(reference() is None for reference in refs)
    assert_module_values(tracked, before)

    invalid = batch(100)
    invalid[1].fill_(float("nan"))
    scale_before = first.bootstrap_scale().item()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        first._update_bootstrap_scale(batch(99), invalid, first.bootstrap_scale())
    assert first.bootstrap_scale().item() == scale_before
    assert "non-finite update skipped" in output.getvalue()


def test_execution_bpi_formula_and_gradient_are_unchanged():
    trainer = make_adaptive_trainer(actor_lr=3e-3)
    inner, outer = batch(110), batch(111)
    T_E = torch.nn.functional.softplus(trainer.log_T)
    pi, pi_new = trainer._virtual_actor_bank_update(inner[0], inner[1], outer[0], T_E)
    actual, _ = trainer._outer_loss(outer[0], pi, pi_new)

    target = trainer.critic_1_target
    displacement = pi_new - pi
    a0 = pi.detach().clone().requires_grad_(True)
    g0 = torch.autograd.grad(target(outer[0], a0).sum(), a0)[0].detach()
    a1 = pi_new.detach().clone().requires_grad_(True)
    g1 = torch.autograd.grad(target(outer[0], a1).sum(), a1)[0].detach()
    d_norm = displacement.norm(dim=1)
    penalty = 0.5 * (g1 - g0).norm(dim=1) * d_norm
    b_pi = (g0 * displacement).sum(dim=1) - penalty
    q_scale = target(outer[0], pi_new).abs().mean().detach().clamp_min(1e-6)
    expected = -b_pi.mean() / q_scale
    assert torch.allclose(actual, expected)
    actual_grad = torch.autograd.grad(actual, trainer.log_T, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, trainer.log_T)[0]
    assert torch.equal(actual_grad, expected_grad)


def test_nonlocal_rms_diagnostics_are_read_only_and_logged():
    trainer = make_adaptive_trainer(actor_lr=3e-3, bootstrap_rms_diagnostic_freq=1)
    inner, outer = batch(120), batch(121)
    tracked = [*trainer.actor_bank, trainer.critic_1_target, trainer.critic_2_target]
    values_before = module_values(tracked)
    optimizer_before = copy.deepcopy(trainer.actor_bank_states)
    diagnostics = trainer._bootstrap_rms_nonlocal_diagnostics(inner, outer)
    assert_module_values(tracked, values_before)
    before_leaves, _ = torch.utils._pytree.tree_flatten(optimizer_before)
    after_leaves, _ = torch.utils._pytree.tree_flatten(trainer.actor_bank_states)
    assert len(before_leaves) == len(after_leaves)
    for before, after in zip(before_leaves, after_leaves):
        if isinstance(before, torch.Tensor):
            assert torch.equal(before, after)
        else:
            assert before == after
    for key in (
        "amo/R_B_autograd_grad_rho",
        "amo/R_B_secant_slope_small",
        "amo/R_B_secant_slope_large",
        "amo/R_B_autograd_secant_sign_agreement_small",
        "amo/R_B_autograd_secant_sign_agreement_large",
        "amo/R_B_autograd_over_secant_small",
        "amo/R_B_autograd_over_secant_large",
        "amo/grad_T_B_from_L2_squared_relative_old_diag",
        "amo/grad_T_B_from_L2_RMS_readonly_diag",
        "amo/abs_grad_ratio_RMS_over_squared_old_diag",
        "amo/RMS_grad_larger_than_squared_old_diag",
        "amo/R_B_monotone_small",
        "amo/R_B_monotone_large",
    ):
        assert key in diagnostics
        assert torch.isfinite(torch.tensor(diagnostics[key]))

    logs = trainer.train(batch(122), batch(123))
    required = {
        "amo/T_E",
        "amo/T_B",
        "amo/T_B_over_T_E",
        "amo/L1_B_Q",
        "amo/L1_B_BC",
        "amo/L1_B",
        "amo/q_scale_B",
        "amo/delta_y_B_mean",
        "amo/delta_y_B_mean_abs",
        "amo/delta_y_B_rms",
        "amo/delta_y_B_rms_relative",
        "amo/delta_y_B_rms_nonterminal_conditional",
        "amo/delta_y_B_max_abs",
        "amo/nonterminal_fraction",
        "amo/L2_squared_relative_diag",
        "amo/L2_RMS_B",
        "amo/grad_T_B_from_L1_Q_chain",
        "amo/grad_T_B_from_L1_BC_chain",
        "amo/grad_T_B_from_L1",
        "amo/grad_T_B_from_L2_RMS",
        "amo/grad_T_B_total",
        "amo/grad_T_E_BPI",
        "amo/grad_T_B_L1_L2_same_sign",
        "amo/abs_grad_ratio_L1_L2",
        "amo/delta_log_T_B",
        "amo/delta_log_T_E",
        "amo/target_critic_grad_norm",
        "amo/B_PI_E",
        "amo/L_T_E",
        "amo/L_T_B",
        "amo/bootstrap_outer_loss_version",
        "amo/critic_bootstrap_actor_role",
        "amo/evaluation_actor_role",
    }
    assert required.issubset(logs)
    assert logs["amo/bootstrap_outer_loss_version"] == (
        amo.BOOTSTRAP_OUTER_LOSS_VERSION
    )
    assert "amo/L2_target_B" not in logs
    for key in required - {"amo/bootstrap_outer_loss_version"}:
        assert torch.isfinite(torch.tensor(logs[key])), key


def test_amo_locomotion9_launcher_manifest_definition():
    launcher_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "launch_amo_adaptive_multiscale_locomotion9.py",
    )
    spec = importlib.util.spec_from_file_location("adaptive_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert len(launcher.ENVIRONMENTS) == 9
    assert len(set(launcher.ENVIRONMENTS)) == 9
    for domain in ("halfcheetah", "hopper", "walker2d"):
        assert {
            f"{domain}-medium-v2",
            f"{domain}-medium-replay-v2",
            f"{domain}-medium-expert-v2",
        }.issubset(launcher.ENVIRONMENTS)
    assert not any(
        environment.endswith("-expert-v2")
        and not environment.endswith("-medium-expert-v2")
        for environment in launcher.ENVIRONMENTS
    )
    config = launcher.resolved_config("hopper-medium-v2")
    assert config["algorithm"] == "amo_adaptive_multiscale"
    assert config["adaptive_multiscale"] is True
    assert config["max_timesteps"] == 1_000_000
    assert config["seed"] == 0
    assert config["initial_T_E"] == config["initial_T_B"]
    assert config["initial_T_E"] == "TrainConfig.T_E default"
    assert config["bootstrap_outer_loss_version"] == (amo.BOOTSTRAP_OUTER_LOSS_VERSION)
    assert config["bootstrap_scale_loss"] == "L1_B+L2_RMS_B"
    assert "coefficient" not in config
    assert not any("proximal_n_steps" in key for key in config)
    command = launcher.command("hopper-medium-v2")
    assert any("amo_adaptive_multiscale_h-m_s0" in argument for argument in command)
    assert "--project=AMO-adaptive-multiscale" in command
    assert "--group=amo-locomotion9-seed0" in command
    assert not any(argument.startswith("--T_E=") for argument in command)
    assert not any(argument.startswith("--T_B=") for argument in command)
    assert not any("proximal_n_steps" in argument for argument in command)
    assert not any("dual_" + "proximal" in argument for argument in command)
    previous = dict(launcher.LAUNCH_SCALES)
    try:
        launcher.LAUNCH_SCALES.update({"T_E": 10.0, "T_B": 1.0, "T_lr": 3e-4})
        flagged = launcher.resolved_config("hopper-medium-v2")
        assert flagged["initial_T_E"] == 10.0
        assert flagged["initial_T_B"] == 1.0
        flagged_command = launcher.command("hopper-medium-v2")
        assert "--T_E=10.0" in flagged_command
        assert "--T_B=1.0" in flagged_command
        assert "--T_lr=0.0003" in flagged_command
    finally:
        launcher.LAUNCH_SCALES.update(previous)
    launcher_source = open(launcher_path, encoding="utf-8").read()
    assert "T_B_to_T_E" not in launcher_source


if __name__ == "__main__":
    raise SystemExit("Run this module with pytest")
