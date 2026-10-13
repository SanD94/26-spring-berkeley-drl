import unittest

import numpy as np
import torch

from agents.pg_agent import PGAgent
from infrastructure import pytorch_util as ptu
from networks.critics import ValueCritic
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
                        # Two trajectories of unequal lengths: sum each, then average.
                        expected_loss = -(
                            np.dot(before[:3], advantages[:3])
                            + np.dot(before[3:], advantages[3:])
                        ) / 2
                        info = agent.update(obs, actions, rewards, terminals)
                        self.assertAlmostEqual(info["Actor Loss"], expected_loss, places=5)
                        with torch.no_grad():
                            after = agent.actor(flat_obs).log_prob(flat_actions).numpy()
                        self.assertGreater(np.dot(after - before, advantages), 0)

    def test_baseline_update_uses_pre_update_values_and_multiple_gradient_steps(self):
        obs = [
            np.array([[1, -2], [0, 3]], dtype=np.float32),
            np.array([[-4, 1]], dtype=np.float32),
        ]
        actions = [np.array([0, 1]), np.array([1])]
        rewards = [np.array([2, -1]), np.array([4])]
        terminals = [np.array([0, 1]), np.array([1])]
        targets = np.array([1.5, -1, 4], dtype=np.float32)
        flat_obs = np.concatenate(obs)
        tensor_obs = ptu.from_numpy(flat_obs)
        for steps in (1, 5):
            for normalize in (False, True):
                with self.subTest(steps=steps, normalize=normalize):
                    agent = PGAgent(
                        ob_dim=2, ac_dim=2, discrete=True, n_layers=1,
                        layer_size=8, gamma=0.5, learning_rate=0.005,
                        use_baseline=True, use_reward_to_go=True,
                        baseline_learning_rate=0.01, baseline_gradient_steps=steps,
                        gae_lambda=None, normalize_advantages=normalize,
                    )
                    reference = ValueCritic(2, 1, 8, 0.01)
                    reference.load_state_dict(agent.critic.state_dict())
                    with torch.no_grad():
                        before_values = agent.critic(tensor_obs).numpy()
                        log_probs = agent.actor(tensor_obs).log_prob(
                            torch.tensor(np.concatenate(actions))
                        ).numpy()
                    advantages = targets - before_values
                    if normalize:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                    expected_actor_loss = -np.dot(log_probs, advantages) / 2
                    for _ in range(steps):
                        expected_mse = np.mean((reference(tensor_obs).detach().numpy() - targets) ** 2)
                        reference_info = reference.update(flat_obs, targets)
                        self.assertAlmostEqual(reference_info["Baseline Loss"], expected_mse, places=5)
                    info = agent.update(obs, actions, rewards, terminals)
                    self.assertAlmostEqual(info["Actor Loss"], expected_actor_loss, places=5)
                    self.assertAlmostEqual(info["Baseline Loss"], expected_mse, places=5)
                    with torch.no_grad():
                        after_values = agent.critic(tensor_obs).numpy()
                        np.testing.assert_allclose(after_values, reference(tensor_obs).numpy())
                        self.assertEqual(agent.critic(tensor_obs[:1]).shape, (1,))
                    self.assertLess(np.mean((after_values - targets) ** 2),
                                    np.mean((before_values - targets) ** 2))

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
        expected_loss = -np.dot(log_probs, advantages) / 2
        info = policy.update(obs, actions, advantages, n_trajectories=2)
        self.assertAlmostEqual(info["Actor Loss"], expected_loss, places=5)
        with torch.no_grad():
            after = policy(ptu.from_numpy(obs)).log_prob(ptu.from_numpy(actions)).numpy()
        self.assertGreater(np.dot(after - log_probs, advantages), 0)
        self.assertEqual(policy.get_action(obs[0]).shape, (2,))


if __name__ == "__main__":
    unittest.main()
