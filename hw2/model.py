import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    """Weight initialization"""
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class FixedNormal(torch.distributions.Normal):
    """Normal distribution module with fixed mean and std."""

    # Log-probability
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1)

    # Entropy
    def entropy(self):
        return super().entropy().sum(-1)

    # Mode
    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    """Diagonal Gaussian distribution"""

    def __init__(self, inp_dim: int, out_dim: int, std: float = 0.5):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )
        self.fc_mean = init_(nn.Linear(inp_dim, out_dim))
        self.std = torch.full((out_dim,), std)

    # Forward
    def forward(self, x):
        mean = self.fc_mean(x)
        return FixedNormal(mean, self.std.to(x.device))


class PolicyNet(nn.Module):
    def __init__(
        self,
        s_dim: int,
        a_dim: int,
        std: float = 0.5,
        n_hidden: int = 512,
    ):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
        )
        self.dist = DiagGaussian(n_hidden, a_dim, std)

    def forward(self, state: torch.Tensor, deterministic: bool = False):
        feature: torch.Tensor = self.main(state)
        dist: torch.Tensor = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist.log_probs(action)

    def action_step(self, state, deterministic=True):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action

    # Evaluate log-probs & entropy
    def evaluate(self, state, action):
        feature: torch.Tensor = self.main(state)
        dist: torch.Tensor = self.dist(feature)
        return dist.log_probs(action), dist.entropy()


class ValueNet(nn.Module):
    def __init__(self, s_dim: int, n_hidden: int = 512):
        super(ValueNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden * 2),
            nn.ReLU(),
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    # Forward
    def forward(self, state):
        return self.main(state)[:, 0]
