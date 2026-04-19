import torch
import torch.nn as nn
from typing import Tuple


class ActorNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 64),
        log_std_min: float = -2.0,
        log_std_max: float = 0.5
    ):
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Capas compartidas
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared_network = nn.Sequential(*layers)

        # Cabeza media
        self.mu_head = nn.Sequential(
            nn.Linear(input_dim, action_dim),
            nn.Tanh()  # acción en [-1, 1]
        )

        # Cabeza log_std
        self.log_std_head = nn.Linear(input_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)
        nn.init.uniform_(self.mu_head[0].weight, -0.003, 0.003)
        nn.init.constant_(self.mu_head[0].bias, 0.0)
        nn.init.uniform_(self.log_std_head.weight, -0.003, 0.003)
        nn.init.constant_(self.log_std_head.bias, 0.0)

    def forward(self, state: torch.Tensor):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        features = self.shared_network(state)
        mu = self.mu_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mu, std

    def get_distribution(self, state: torch.Tensor):
        mu, std = self.forward(state)
        return torch.distributions.Normal(mu, std)

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        dist = self.get_distribution(state)
        log_prob = dist.log_prob(action).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1)           # (batch,)
        return log_prob, entropy


class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128, 64)
    ):
        super(CriticNetwork, self).__init__()

        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0.01)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        return self.network(state)  # (batch, 1)
