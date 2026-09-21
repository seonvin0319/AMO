"""IQL DDPG+BC AMO: single π_E, α_E meta-loss is L_E only, no π_B."""

import unittest

import numpy as np

from common.agent import make_agent
from common.config import load_config


class TestIqlDdpgbcAmoLe(unittest.TestCase):
    def test_single_actor_le_finite_and_no_bootstrap(self):
        c = load_config(
            "iql_ddpgbc_amo",
            "hopper-medium-v2",
            overrides={
                "hidden_dim": 8,
                "batch_size": 6,
                "meta_warmup_steps": 0,
                "meta_interval": 1,
            },
        )
        self.assertTrue(c["const_std"])
        agent = make_agent("iql_ddpgbc_amo", "jax", 11, 3, c, seed=0, device="cpu")
        self.assertNotIn("bootstrap", agent.state["p"])
        self.assertNotIn("scale_B", agent.state["p"])
        self.assertIn("scale_E", agent.state["p"])
        rng = np.random.default_rng(3)

        def batch():
            return agent.ops.batch(
                {
                    "observations": rng.normal(size=(6, 11)).astype(np.float32),
                    "next_observations": rng.normal(size=(6, 11)).astype(np.float32),
                    "actions": np.tanh(rng.normal(size=(6, 3))).astype(np.float32),
                    "terminals": np.zeros((6, 1), np.float32),
                    "rewards": rng.normal(size=(6, 1)).astype(np.float32),
                }
            )

        inner, outer = batch(), batch()
        l_e = float(
            np.asarray(
                agent.execution_outer(
                    agent.state["p"]["scale_E"], agent.state, inner, outer
                )
            )
        )
        self.assertTrue(np.isfinite(l_e))
        metrics = agent.update(batch(), batch())
        self.assertIn("L_alpha_E", metrics)
        self.assertNotIn("L_alpha_B", metrics)
        self.assertTrue(np.isfinite(metrics["L_alpha_E"]))
        self.assertTrue(np.isfinite(metrics["actor_loss"]))
        self.assertGreater(float(np.asarray(metrics["alpha_E"])), 0.0)


if __name__ == "__main__":
    unittest.main()
