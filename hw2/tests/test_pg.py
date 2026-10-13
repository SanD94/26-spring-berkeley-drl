import unittest

import numpy as np
import torch

from agents.pg_agent import PGAgent
from infrastructure import pytorch_util as ptu
from networks.policies import MLPPolicyPG


class PolicyGradientTest(unittest.TestCase):
    def setUp(self):
        ptu.init_gpu(use_gpu=False)
        torch.manual_seed(7)

    def test_agent_update_uses_discounted_returns_and_batch_normalization(self):
        obs = [
            np.array([[1, -2], [0, 3], [-4, 1]], dtype=np.float32),
            np.array([[2, 1], [-1, -3]], dtype=np.float32),
        ]
        actions = [np.array([0, 1, 1]), np.array([1, 0])]
        rewards = [np.array([2, -1, 4]), np.array([3, 8])]
        terminals = [np.array([0, 0, 1]), np.array([0, 1])]
        expected_returns = {
            0.0: ([2, 2, 2, 3, 3], [2, -1, 4, 3, 8]),
            0.5: ([2.5, 2.5, 2.5, 7, 7], [2.5, 1, 4, 7, 8]),
            1.0: ([5, 5, 5, 11, 11], [5, 3, 4, 11, 8]),
        }
        for gamma, returns in expected_returns.items():
            for rtg in (False, True):
                for normalize in (False, True):
                    with self.subTest(gamma=gamma, rtg=rtg, normalize=normalize):
                        agent = PGAgent(
                            ob_dim=2, ac_dim=2, discrete=True, n_layers=1,
                            layer_size=8, gamma=gamma, learning_rate=0.005,
                            use_baseline=False, use_reward_to_go=rtg,
                            baseline_learning_rate=None, baseline_gradient_steps=None,
                            gae_lambda=None, normalize_advantages=normalize,
                        )
                        advantages = np.array(returns[int(rtg)], dtype=np.float32)
                        if normalize:
                            advantages = (advantages - advantages.mean()) / (
                                advantages.std() + 1e-8
                            )
                        flat_obs = ptu.from_numpy(np.concatenate(obs))
                        flat_actions = torch.tensor(np.concatenate(actions))
                        with torch.no_grad():
                            before = agent.actor(flat_obs).log_prob(flat_actions).numpy()
                        expected_loss = -np.dot(before, advantages) / 5
                        info = agent.update(obs, actions, rewards, terminals)
                        self.assertAlmostEqual(info["Actor Loss"], expected_loss, places=5)
                        with torch.no_grad():
                            after = agent.actor(flat_obs).log_prob(flat_actions).numpy()
                        self.assertGreater(np.dot(after - before, advantages), 0)

    def test_continuous_policy_uses_joint_action_log_probability(self):
        policy = MLPPolicyPG(
            ac_dim=2, ob_dim=2, discrete=False, n_layers=1,
            layer_size=8, learning_rate=0.005,
        )
        obs = np.array([[1, -2], [0, 3], [-4, 1]], dtype=np.float32)
        actions = np.array([[0.5, -1], [2, 0.1], [-0.7, 1.5]], dtype=np.float32)
        advantages = np.array([2, -1, 0.5], dtype=np.float32)
        with torch.no_grad():
            distribution = policy(ptu.from_numpy(obs))
            mean = distribution.mean.numpy()
            variance = distribution.variance.numpy()
        log_probs = -0.5 * (
            np.log(2 * np.pi * variance) + (actions - mean) ** 2 / variance
        ).sum(axis=1)
        expected_loss = -np.dot(log_probs, advantages) / 3
        info = policy.update(obs, actions, advantages)
        self.assertAlmostEqual(info["Actor Loss"], expected_loss, places=5)
        with torch.no_grad():
            after = policy(ptu.from_numpy(obs)).log_prob(ptu.from_numpy(actions)).numpy()
        self.assertGreater(np.dot(after - log_probs, advantages), 0)
        self.assertEqual(policy.get_action(obs[0]).shape, (2,))


if __name__ == "__main__":
    unittest.main()
