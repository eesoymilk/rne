import torch
import gym
import numpy as np
from collections import deque
from typing import Literal

from multi_env import MultiEnv
from model import PolicyNet, ValueNet


def compute_discounted_return(
    rewards, dones, last_values, last_dones, gamma=0.99
):
    returns = np.zeros_like(rewards)
    n_step = len(rewards)

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            returns[t] = rewards[t] + gamma * last_values * (1.0 - last_dones)
        else:
            returns[t] = rewards[t] + gamma * returns[t + 1] * (
                1.0 - dones[t + 1]
            )

    return returns


def compute_gae(
    rewards, values, dones, last_values, last_dones, gamma=0.99, lamb=0.95
):
    # rewards    : (n_step, n_env)
    # values     : (n_step, n_env)
    # dones      : (n_step, n_env)
    # advs       : (n_step, n_env)
    # last_values: (n_env)
    # last_dones : (n_env)
    advs = np.zeros_like(rewards)
    n_step = len(rewards)
    last_gae_lam = 0.0

    for t in reversed(range(n_step)):
        if t == n_step - 1:
            next_nonterminal = 1.0 - last_dones
            next_values = last_values
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        advs[t] = last_gae_lam = (
            delta + gamma * lamb * next_nonterminal * last_gae_lam
        )

    return advs + values


class EnvRunner:
    """Runner for multiple environments"""

    def __init__(
        self,
        env: MultiEnv,
        s_dim: int,
        a_dim: int,
        n_step: int = 5,
        gamma: float = 0.99,
        lamb: float = 0.95,
        device: Literal["cuda", "cpu"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.env = env
        self.n_env = env.n_env
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_step = n_step
        self.gamma = gamma
        self.lamb = lamb
        self.device = device

        # last states: (n_env, s_dim)
        # last dones : (n_env)
        self.states = self.env.reset()
        self.dones = np.ones((self.n_env), dtype=np.bool_)

        # Storages (state, action, value, reward, a_logp, done)
        self.mb_states = np.zeros(
            (self.n_step, self.n_env, self.s_dim), dtype=np.float32
        )
        self.mb_actions = np.zeros(
            (self.n_step, self.n_env, self.a_dim), dtype=np.float32
        )
        self.mb_values = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_rewards = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.n_step, self.n_env), dtype=np.float32)
        self.mb_dones = np.zeros((self.n_step, self.n_env), dtype=np.bool_)

        # Reward & length recorder
        self.total_rewards = np.zeros((self.n_env), dtype=np.float32)
        self.total_len = np.zeros((self.n_env), dtype=np.int32)
        self.reward_buf = deque(maxlen=100)
        self.len_buf = deque(maxlen=100)

    def _run_step(self, step: int, policy_net: PolicyNet, value_net: ValueNet):
        """Run a single step"""
        state_tensor = torch.tensor(
            self.states, dtype=torch.float32, device=self.device
        )
        action, a_logp = policy_net(state_tensor)
        value = value_net(state_tensor)

        actions = action.cpu().numpy()
        a_logps = a_logp.cpu().numpy()
        values = value.cpu().numpy()

        self.mb_states[step, :] = self.states
        self.mb_dones[step, :] = self.dones
        self.mb_actions[step, :] = actions
        self.mb_a_logps[step, :] = a_logps
        self.mb_values[step, :] = values
        self.states, rewards, self.dones, _ = self.env.step(actions)
        self.mb_rewards[step, :] = rewards

    def run(self, policy_net: PolicyNet, value_net: ValueNet):
        """Run n steps to get a batch"""
        # 1. Run n steps
        for step in range(self.n_step):
            self._run_step(step, policy_net, value_net)

        last_values = (
            value_net(torch.from_numpy(self.states).float().to(self.device))
            .cpu()
            .numpy()
        )
        self.record()

        # 2. Compute returns
        mb_returns = compute_gae(
            self.mb_rewards,
            self.mb_values,
            self.mb_dones,
            last_values,
            self.dones,
            self.gamma,
            self.lamb,
        )

        return (
            self.mb_states.reshape(self.n_step * self.n_env, self.s_dim),
            self.mb_actions.reshape(self.n_step * self.n_env, self.a_dim),
            self.mb_a_logps.flatten(),
            self.mb_values.flatten(),
            mb_returns.flatten(),
        )

    # Record return & length
    def record(self):
        for i in range(self.n_step):
            for j in range(self.n_env):
                if self.mb_dones[i, j]:
                    self.reward_buf.append(
                        self.total_rewards[j] + self.mb_rewards[i, j]
                    )
                    self.len_buf.append(self.total_len[j] + 1)
                    self.total_rewards[j] = 0
                    self.total_len[j] = 0
                else:
                    self.total_rewards[j] += self.mb_rewards[i, j]
                    self.total_len[j] += 1

    # Get performance
    def get_performance(self):
        if len(self.reward_buf) == 0:
            mean_return = 0
            std_return = 0
        else:
            mean_return = np.mean(self.reward_buf)
            std_return = np.std(self.reward_buf)

        if len(self.len_buf) == 0:
            mean_len = 0
        else:
            mean_len = np.mean(self.len_buf)

        return mean_return, std_return, mean_len
