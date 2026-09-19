"""Production L2 precision scope and its differentiated computation graph."""

import unittest

import jax
import numpy as np

from common.agent import make_agent
from common.config import load_config


class TestL2Precision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = load_config(
            "td3_amo",
            "walker2d-medium-replay-v2",
            overrides={"hidden_dim": 8, "batch_size": 6},
        )
        cls.agent = make_agent("td3_amo", "jax", 3, 2, c, seed=5, device="cpu")
        rng = np.random.default_rng(91)

        def batch():
            return cls.agent.ops.batch(
                {
                    "observations": rng.normal(size=(6, 3)).astype(np.float32),
                    "next_observations": rng.normal(size=(6, 3)).astype(np.float32),
                    "actions": np.tanh(rng.normal(size=(6, 2))).astype(np.float32),
                    "terminals": np.array([[0], [1], [0], [0], [1], [0]], np.float32),
                }
            )

        cls.inner, cls.outer = batch(), batch()

    def check_term(self, index, caller_precision):
        a = self.agent
        expected_precision = caller_precision if index == 0 else "highest"

        def actual(p):
            return a.bootstrap_terms(p, a.state, self.inner, self.outer)[index]

        def expected(p):
            with jax.default_matmul_precision(expected_precision):
                return a._bootstrap_terms(p, a.state, self.inner, self.outer)[index]

        p = a.state["p"]["scale_B"]
        with jax.default_matmul_precision(caller_precision):
            actual_fn = jax.jit(jax.value_and_grad(actual))
            expected_fn = jax.jit(jax.value_and_grad(expected))
            left, right = actual_fn(p), expected_fn(p)
            for x, y in zip(
                jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
            ):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            hlo = actual_fn.lower(p).compile().as_text()
            self.assertEqual(jax.config.jax_default_matmul_precision, caller_precision)
        # Unused computations from the other term, including their precision
        # annotations, must have been removed from the optimized graph.
        if expected_precision != "highest":
            self.assertNotIn("operand_precision={highest,highest}", hlo)
        else:
            self.assertIn("operand_precision={highest,highest}", hlo)

    def test_l1_value_and_gradient_keep_caller_precision(self):
        for precision in ("default", "high", "highest"):
            with self.subTest(caller_precision=precision):
                self.check_term(0, precision)

    def test_l2_value_and_gradient_use_highest(self):
        for precision in ("default", "high", "highest"):
            with self.subTest(caller_precision=precision):
                self.check_term(1, precision)

    def test_terminal_batch_excludes_policy_loss_from_main_hypergradient(self):
        # With every outer transition terminal, target movement is exactly zero.
        # L1 can still have a nonzero gradient, so this detects leakage into L_B.
        a = self.agent
        outer = dict(self.outer)
        outer["terminals"] = jax.numpy.ones_like(outer["terminals"])
        p = a.state["p"]["scale_B"]
        a.c["bootstrap_loss"] = "l2_rms"
        value, grad = jax.value_and_grad(
            lambda x: a.bootstrap_outer(x, a.state, self.inner, outer)
        )(p)
        self.assertEqual(float(value), 0.0)
        self.assertEqual(float(grad["rho"]), 0.0)
        try:
            a.c["bootstrap_loss"] = "l1_l2_rms"
            legacy, legacy_grad = jax.value_and_grad(
                lambda x: a.bootstrap_outer(x, a.state, self.inner, outer)
            )(p)
            self.assertTrue(np.isfinite(float(legacy)))
            self.assertGreater(abs(float(legacy_grad["rho"])), 1e-10)
        finally:
            a.c["bootstrap_loss"] = "l2_rms"

    def test_production_objective_uses_rms_value_and_hypergradient(self):
        a = self.agent
        self.assertEqual(a.c["bootstrap_loss"], "l2_rms")
        p = a.state["p"]["scale_B"]
        with jax.default_matmul_precision("default"):
            actual = jax.jit(jax.value_and_grad(
                lambda x: a.bootstrap_outer(x, a.state, self.inner, self.outer)
            ))(p)
            with jax.default_matmul_precision("highest"):
                expected = jax.jit(jax.value_and_grad(
                    lambda x: a._bootstrap_terms(x, a.state, self.inner, self.outer)[1]
                ))(p)
        for x, y in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


if __name__ == "__main__":
    unittest.main()
