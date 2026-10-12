"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        predicted_chunk = self.sample_actions(state)
        return (predicted_chunk - action_chunk).square().sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.network(state).reshape(
            state.shape[0], self.chunk_size, self.action_dim
        )


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers: list[nn.Module] = []
        input_dim = state_dim + chunk_size * action_dim + 1
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(
            (state.shape[0], 1, 1), device=action_chunk.device, dtype=action_chunk.dtype
        )
        interpolated_chunk = tau * action_chunk + (1 - tau) * noise
        inputs = torch.cat(
            (state, interpolated_chunk.flatten(1), tau.flatten(1)), dim=1
        )
        velocity = self.network(inputs).reshape_as(action_chunk)
        target_velocity = action_chunk - noise
        return (velocity - target_velocity).square().sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        action_chunk = torch.randn(
            (state.shape[0], self.chunk_size, self.action_dim),
            device=state.device,
            dtype=state.dtype,
        )
        for step in range(num_steps):
            tau = state.new_full((state.shape[0], 1), step / num_steps)
            inputs = torch.cat((state, action_chunk.flatten(1), tau), dim=1)
            velocity = self.network(inputs).reshape_as(action_chunk)
            action_chunk = action_chunk + velocity / num_steps
        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
