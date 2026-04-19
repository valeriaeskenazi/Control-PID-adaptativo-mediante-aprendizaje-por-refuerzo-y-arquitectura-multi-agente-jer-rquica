import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional

from .model_AC import ActorNetwork, CriticNetwork
from ..memory import AbstractReplayBuffer, SimpleReplayBuffer, Experience
from ..abstract_agent import AbstractActorCriticAgent


class ACAgent(AbstractActorCriticAgent):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        agent_role: str,          
        n_vars: int,
        hidden_dims: tuple = (128, 128, 64),
        lr_actor: float = 0.0001,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,   # Coeficiente de entropía para exploración (S&B Ec. 13.7)
        replay_buffer: Optional[AbstractReplayBuffer] = None,
        buffer_size: int = 50000,
        batch_size: int = 64,
        warmup_steps: int = 500,
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
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.hidden_dims = hidden_dims

        # Redes
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)

        # Optimizadores separados (igual que el AC viejo)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer (mejora sobre el AC one-step viejo)
        if replay_buffer is not None:
            self.memory = replay_buffer
        else:
            self.memory = SimpleReplayBuffer(capacity=buffer_size, device=device)

        print(f"AC Estocástico creado | role={agent_role} | state={state_dim} | action={action_dim} | device={device}")

    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        #Muestrea acción de la política estocástica π(a|s) = N(μ(s), σ(s)).
        state_tensor = self.preprocess_state(state)

        if training:
            action_tensor, _ = self.actor.sample_action(state_tensor)
        else:
            with torch.no_grad():
                mu, _ = self.actor(state_tensor)
                action_tensor = mu

        action = action_tensor.squeeze(0).detach().cpu().numpy()
        return action.astype(np.float32)

    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        if len(self.memory) < max(self.batch_size, self.warmup_steps):
            return {}

        batch = self.memory.sample(self.batch_size)
        states = batch['states']            # (batch, state_dim)
        actions = batch['actions'].float()  # (batch, action_dim)
        rewards = batch['rewards']          # (batch,)
        next_states = batch['next_states']  # (batch, state_dim)
        dones = batch['dones']              # (batch,)

        # 1. CALCULAR VENTAJA (TD error) — núcleo de S&B Cap. 13
        current_values = self.critic(states).squeeze(1)       # V(s)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(1) # V(s')
            next_values[dones] = 0.0
            target_values = rewards + self.gamma * next_values
            advantage = target_values - current_values.detach()  # δ = r + γV(s') - V(s)

        # 2. ACTUALIZAR CRITIC — minimizar (V(s) - target)²
        critic_loss = self.compute_critic_loss(states, actions, rewards, next_states, dones)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer_critic.step()

        # 3. ACTUALIZAR ACTOR — maximizar E[log π(a|s) * δ] + H[π]
        actor_loss = self.compute_actor_loss(states, actions, advantage)
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer_actor.step()

        self.training_step += 1

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'advantage_mean': advantage.mean().item(),
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }

    def compute_actor_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor
    ) -> torch.Tensor:
        
        dist = self.actor.get_distribution(states)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # (batch,)
        entropy = dist.entropy().sum(dim=-1).mean()      # H[π]

        policy_loss = -(log_probs * advantages).mean()
        entropy_loss = -self.entropy_coef * entropy      # negativo porque maximizamos entropía

        return policy_loss + entropy_loss

    def compute_critic_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        
        current_values = self.critic(states).squeeze(1)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(1)
            next_values[dones] = 0.0
            target_values = rewards + self.gamma * next_values

        return nn.MSELoss()(current_values, target_values)

    # Compatibilidad con trainer (epsilon no aplica)
    def get_epsilon(self) -> float:
        return 0.0

    def save(self, filepath: str) -> None:
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'hidden_dims': self.hidden_dims,
            'n_vars': self.n_vars
        }
        torch.save(checkpoint, filepath)
        print(f"AC Agent guardado en: {filepath}")

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"AC Agent cargado desde: {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'actor_params': sum(p.numel() for p in self.actor.parameters()),
            'critic_params': sum(p.numel() for p in self.critic.parameters()),
            'device': str(self.device),
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef
        }
