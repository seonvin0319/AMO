"""IQL+AMO L2 precision scope and its differentiated computation graph."""

import unittest

import jax
import numpy as np

from common.agent import make_agent
from common.config import load_config


class TestIQLAmoL2Precision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = load_config(
            "iql_amo",
            "walker2d-medium-replay-v2",
            overrides={"hidden_dim": 8, "batch_size": 6},
        )
        cls.agent = make_agent("iql_amo", "jax", 3, 2, c, seed=5, device="cpu")
        rng = np.random.default_rng(91)

        def batch():
            return cls.agent.ops.batch(
                {
                    "observations": rng.normal(size=(6, 3)).astype(np.float32),
                    "next_observations": rng.normal(size=(6, 3)).astype(np.float32),
                    "actions": np.tanh(rng.normal(size=(6, 2))).astype(np.float32),
                    "terminals": np.array(
                        [[0], [1], [0], [0], [1], [0]], np.float32
                    ),
                }
            )

        cls.inner, cls.outer = batch(), batch()
        cls.adv = cls.agent.ops.array(rng.normal(size=(6, 1)).astype(np.float32))
        cls.lr = float(c["actor_lr"])

    def check_term(self, index, caller_precision):
        a = self.agent
        expected_precision = caller_precision if index == 0 else "highest"

        def actual(p):
            return a.bootstrap_terms(
                p, a.state, self.inner, self.outer, self.adv, self.lr
            )[index]

        def expected(p):
            with jax.default_matmul_precision(expected_precision):
                return a._bootstrap_terms(
                    p, a.state, self.inner, self.outer, self.adv, self.lr
                )[index]

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


if __name__ == "__main__":
    unittest.main()
