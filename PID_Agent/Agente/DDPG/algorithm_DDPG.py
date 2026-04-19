import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
import copy

from .model_DDPG import ActorNetwork, CriticNetwork
from ..memory import AbstractReplayBuffer, SimpleReplayBuffer, Experience
from ..abstract_agent import AbstractActorCriticAgent


class OUNoise:

    def __init__(self, action_dim: int, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class DDPGAgent(AbstractActorCriticAgent):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,          # Para ORCH: n_vars. Para CTRL: n_vars * 3 (kp,ki,kd por var)
        agent_role: str,          # 'ctrl' o 'orch'
        n_vars: int,              # Número de variables manipulables
        hidden_dims: tuple = (128, 128, 64),
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,       # Soft update factor para redes target
        noise_sigma: float = 0.2, # Desviación del ruido OU
        noise_theta: float = 0.15,
        replay_buffer: Optional[AbstractReplayBuffer] = None,
        buffer_size: int = 100000,
        batch_size: int = 64,
        warmup_steps: int = 1000, # Steps antes de empezar a entrenar
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_role=agent_role,
            device=device,
            seed=seed
        )

        self.n_vars = n_vars
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.hidden_dims = hidden_dims

        # Redes Actor y Critic
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dims).to(self.device)

        # Redes Target (copias congeladas que se actualizan suavemente)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Congelar redes target (no se optimizan directamente)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizadores
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer
        if replay_buffer is not None:
            self.memory = replay_buffer
        else:
            self.memory = SimpleReplayBuffer(capacity=buffer_size, device=device)

        # Ruido OU para exploración
        self.noise = OUNoise(action_dim, theta=noise_theta, sigma=noise_sigma)
        self.noise_sigma = noise_sigma  # Para decay opcional

        print(f"DDPG Agent creado | role={agent_role} | state={state_dim} | action={action_dim} | n_vars={n_vars} | device={device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:

        state_tensor = self.preprocess_state(state)

        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()

        if training:
            noise = self.noise.sample()
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)

    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:

        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return {}

        batch = self.memory.sample(self.batch_size)
        states = batch['states']           # (batch, state_dim)
        actions = batch['actions']         # (batch, action_dim) - float
        rewards = batch['rewards']         # (batch,)
        next_states = batch['next_states'] # (batch, state_dim)
        dones = batch['dones']             # (batch,)

        # 1. ACTUALIZAR CRITIC
        critic_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones)

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer_critic.step()

        # 2. ACTUALIZAR ACTOR
        actor_loss = self.compute_actor_loss(states)

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer_actor.step()

        # 3. SOFT UPDATE de redes target
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.training_step += 1

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }

    def compute_actor_loss(self, states: torch.Tensor) -> torch.Tensor:

        actions_pred = self.actor(states)
        q_values = self.critic(states, actions_pred)
        return -q_values.mean()

    def compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * ~dones

        current_q = self.critic(states, actions).squeeze(1)
        return nn.MSELoss()(current_q, target_q)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update: target = tau * source + (1 - tau) * target
        Más estable que hard update (como DQN usa cada N steps).
        """
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def reset_noise(self):
        """Resetear ruido OU al inicio de cada episodio."""
        self.noise.reset()

    # Métodos de compatibilidad con DQNTrainer (epsilon no aplica, retorna 0)
    def get_epsilon(self) -> float:
        return 0.0

    def save(self, filepath: str) -> None:
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
            'tau': self.tau,
            'hidden_dims': self.hidden_dims,
            'n_vars': self.n_vars
        }
        torch.save(checkpoint, filepath)
        print(f"DDPG Agent guardado en: {filepath}")

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"DDPG Agent cargado desde: {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'actor_params': sum(p.numel() for p in self.actor.parameters()),
            'critic_params': sum(p.numel() for p in self.critic.parameters()),
            'device': str(self.device),
            'gamma': self.gamma,
            'tau': self.tau
        }
