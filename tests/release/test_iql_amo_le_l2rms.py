"""IQL-AMO β_E meta-loss can be L_E (BPI) + L2_RMS on the execution actor."""

import unittest

import numpy as np

from common.agent import make_agent
from common.config import load_config


class TestIqlLeL2Rms(unittest.TestCase):
    def test_execution_meta_pieces_are_finite(self):
        c = load_config(
            "iql_amo",
            "hopper-medium-v2",
            path="/home/svcho/amo/configs/iql_amo_lel2_b5_r1e3.yaml",
            overrides={"hidden_dim": 8, "batch_size": 6, "execution_meta_loss": "le_l2_rms"},
        )
        self.assertEqual(c["execution_meta_loss"], "le_l2_rms")
        agent = make_agent("iql_amo", "jax", 11, 3, c, seed=0, device="cpu")
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
        adv = agent.ops.array(rng.normal(size=(6, 1)).astype(np.float32))
        lr = float(c["actor_lr"])
        scale = agent.state["p"]["scale_E"]
        l_e = float(
            np.asarray(agent.execution_outer(scale, agent.state, inner, outer, adv, lr))
        )
        l2 = float(
            np.asarray(
                agent.bootstrap_terms(
                    scale, agent.state, inner, outer, adv, lr, actor_name="actor"
                )[1]
            )
        )
        self.assertTrue(np.isfinite(l_e) and np.isfinite(l2))
        self.assertTrue(np.isfinite(l_e + l2))


if __name__ == "__main__":
    unittest.main()
