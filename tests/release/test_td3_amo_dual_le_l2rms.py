"""Dual-actor π_E meta-loss L_E (BPI) + L2_RMS; π_B stays bootstrap L2_RMS."""

import unittest

import numpy as np

from common.agent import make_agent
from common.config import load_config


class TestDualLeL2Rms(unittest.TestCase):
    def test_dual_actor_le_plus_l2rms_keeps_bootstrap(self):
        c = load_config(
            "td3_amo",
            "hopper-medium-v2",
            overrides={
                "hidden_dim": 8,
                "batch_size": 6,
                "execution_only": False,
                "execution_meta_loss": "le_l2_rms",
                "bootstrap_loss": "l2_rms",
                "meta_interval": 2,
            },
        )
        agent = make_agent("td3_amo", "jax", 11, 3, c, seed=0, device="cpu")
        self.assertIn("bootstrap", agent.state["p"])
        self.assertIn("scale_B", agent.state["p"])
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
        scale = agent.state["p"]["scale_E"]
        l_e = float(np.asarray(agent.execution_outer(scale, agent.state, inner, outer)))
        l2 = float(
            np.asarray(
                agent.bootstrap_terms(
                    scale, agent.state, inner, outer, actor_name="actor"
                )[1]
            )
        )
        self.assertTrue(np.isfinite(l_e) and np.isfinite(l2))
        self.assertNotEqual(l2, 0.0)
        agent.update(batch(), batch())
        metrics = agent.update(batch(), batch())
        self.assertIn("L_alpha_E", metrics)
        self.assertIn("L_alpha_B", metrics)
        self.assertTrue(np.isfinite(metrics["L_alpha_E"]))
        self.assertTrue(np.isfinite(metrics["L_alpha_B"]))


if __name__ == "__main__":
    unittest.main()
