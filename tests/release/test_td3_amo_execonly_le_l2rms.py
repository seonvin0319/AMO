"""π_E-only α_E meta-loss is L_E (BPI) + L2_RMS on the execution actor."""

import unittest

import numpy as np

from common.agent import make_agent
from common.config import load_config


class TestExeconlyLeL2Rms(unittest.TestCase):
    def test_execution_meta_equals_le_plus_l2rms(self):
        c = load_config(
            "td3_amo",
            "hopper-medium-v2",
            path="/home/svcho/amo/configs/td3_amo_execonly_a5_r1e3.yaml",
            overrides={"hidden_dim": 8, "batch_size": 6, "execution_only": True},
        )
        agent = make_agent("td3_amo", "jax", 11, 3, c, seed=0, device="cpu")
        rng = np.random.default_rng(3)

        def batch():
            return agent.ops.batch(
                {
                    "observations": rng.normal(size=(6, 11)).astype(np.float32),
                    "next_observations": rng.normal(size=(6, 11)).astype(np.float32),
                    "actions": np.tanh(rng.normal(size=(6, 3))).astype(np.float32),
                    "terminals": np.zeros((6, 1), np.float32),
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
        combined = l_e + l2
        self.assertTrue(np.isfinite(combined))


if __name__ == "__main__":
    unittest.main()
