import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, List

from .model_PPO import ActorNetwork, CriticNetwork
from ..abstract_agent import AbstractActorCriticAgent


class RolloutBuffer:
    """
    Buffer on-policy para PPO.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        self.clear()

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float,
        terminated: bool = None   # True solo si es estado terminal real (no truncado por max_steps)
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        # Si no se pasa terminated, se asume igual a done (comportamiento anterior)
        self.terminated.append(terminated if terminated is not None else done)

    def get(self) -> Dict[str, torch.Tensor]:
        """Convierte el buffer a tensores para el update."""
        return {
            'states':      torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions':     torch.FloatTensor(np.array(self.actions)).to(self.device),
            'rewards':     torch.FloatTensor(np.array(self.rewards)).to(self.device),
            'next_states': torch.FloatTensor(np.array(self.next_states)).to(self.device),
            'dones':       torch.BoolTensor(np.array(self.dones)).to(self.device),
            'terminated':  torch.BoolTensor(np.array(self.terminated)).to(self.device),
            'log_probs':   torch.FloatTensor(np.array(self.log_probs)).to(self.device),
            'values':      torch.FloatTensor(np.array(self.values)).to(self.device),
        }

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.terminated = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        return len(self.states)


class PPOAgent(AbstractActorCriticAgent):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        agent_role: str,
        n_vars: int,
        hidden_dims: tuple = (128, 128, 64),
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,      # Factor λ para GAE (Schulman et al. 2016)
        clip_epsilon: float = 0.2,     # ε de clipping (Ec. 7 del paper)
        ppo_epochs: int = 10,          # Epochs de actualización por rollout
        rollout_steps: int = 2048,     # Steps por rollout antes de actualizar
        mini_batch_size: int = 64,     # Tamaño de mini-batch dentro de cada epoch
        entropy_coef: float = 0.01,    # Coeficiente de entropía (Ec. 9 del paper)
        value_coef: float = 0.5,       # Coeficiente del critic loss (Ec. 9 del paper)
        max_grad_norm: float = 0.5,    # Gradient clipping
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
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.rollout_steps = rollout_steps
        self.mini_batch_size = mini_batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.hidden_dims = hidden_dims

        # Redes separadas (igual que AC)
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)

        # Optimizadores separados
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Rollout buffer (on-policy)
        self.buffer = RolloutBuffer(device=device)

        print(f"PPO Agent creado | role={agent_role} | state={state_dim} | action={action_dim} | "
              f"clip_ε={clip_epsilon} | epochs={ppo_epochs} | rollout={rollout_steps} | device={device}")

    def select_action(self, state: np.ndarray, training: bool = True):
        state_tensor = self.preprocess_state(state)

        if training:
            with torch.no_grad():
                dist = self.actor.get_distribution(state_tensor)
                action_tensor = dist.sample()
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
                log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
                value = self.critic(state_tensor).item()

            self._last_log_prob = log_prob
            self._last_value = value
        else:
            with torch.no_grad():
                mu, _ = self.actor(state_tensor)
                action_tensor = mu

            self._last_log_prob = 0.0
            self._last_value = 0.0

        action = action_tensor.squeeze(0).cpu().numpy()
        return action.astype(np.float32)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        terminated: bool = None
    ):

        self.buffer.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=self._last_log_prob,
            value=self._last_value,
            terminated=terminated
        )

    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:

        if len(self.buffer) < self.rollout_steps:
            return {}

        batch = self.buffer.get()
        advantages, returns = self._compute_gae(batch)

        # Normalizar ventajas (práctica estándar en PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = []
        total_critic_loss = []
        total_entropy = []
        total_clip_fraction = []

        # Múltiples epochs sobre el mismo rollout (clave de PPO)
        for _ in range(self.ppo_epochs):
            # Mini-batches aleatorios dentro del rollout
            indices = torch.randperm(len(self.buffer))

            for start in range(0, len(self.buffer), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states   = batch['states'][mb_idx]
                mb_actions  = batch['actions'][mb_idx]
                mb_old_lp   = batch['log_probs'][mb_idx]
                mb_adv      = advantages[mb_idx]
                mb_returns  = returns[mb_idx]

                # Log probs con política ACTUAL (nueva)
                new_log_probs, entropy = self.actor.evaluate_actions(mb_states, mb_actions)

                # Ratio π_new / π_old (en espacio log para estabilidad)
                ratio = torch.exp(new_log_probs - mb_old_lp)

                # Clipped surrogate loss (Ec. 7, Schulman et al. 2017)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss
                values_pred = self.critic(mb_states).squeeze(1)
                critic_loss = nn.MSELoss()(values_pred, mb_returns)

                # Loss combinada (Ec. 9, Schulman et al. 2017)
                # L = L_CLIP - c1*L_VF + c2*H[π]
                entropy_loss = -self.entropy_coef * entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss + entropy_loss

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

                # Clip fraction: fracción de steps donde se activó el clip (métrica de diagnóstico)
                clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()

                total_actor_loss.append(actor_loss.item())
                total_critic_loss.append(critic_loss.item())
                total_entropy.append(entropy.mean().item())
                total_clip_fraction.append(clip_fraction)

        # Descartar buffer (on-policy: no reutilizar experiencias)
        self.buffer.clear()
        self.training_step += 1

        return {
            'actor_loss':     np.mean(total_actor_loss),
            'critic_loss':    np.mean(total_critic_loss),
            'entropy':        np.mean(total_entropy),
            'clip_fraction':  np.mean(total_clip_fraction),  # útil para diagnosticar clip_epsilon
            'training_step':  self.training_step,
        }

    def _compute_gae(self, batch: Dict[str, torch.Tensor]):
        """
        Generalized Advantage Estimation (GAE).

        GAE(λ) balancea bias y varianza en la estimación de ventaja:
            λ=0 → TD(0), bajo varianza, alto bias
            λ=1 → Monte Carlo, alto varianza, bajo bias
            λ=0.95 → balance empírico (Schulman et al.)

        Nota: se usa `terminated` (estado terminal real) y NO `dones` para zerear
        next_values. Los episodios truncados por max_steps tienen next_value != 0.
        """
        rewards     = batch['rewards']
        dones       = batch['dones']
        terminated  = batch['terminated']
        values      = batch['values']
        next_states = batch['next_states']

        # Normalizar rewards para estabilizar el critic (estándar en PPO)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(1)
            # Solo zerear next_value en estados terminales reales, NO en truncados
            next_values[terminated] = 0.0

        advantages = torch.zeros_like(rewards)
        gae = 0.0

        # Recorrer en reversa para calcular GAE
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            # Resetear GAE en límites de episodio (done incluye truncated y terminated)
            gae = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else gae)
            advantages[t] = gae

        returns = advantages + values  # V(s) + A(s,a) = Q(s,a)
        return advantages, returns

    # Compatibilidad con trainer
    def get_epsilon(self) -> float:
        return 0.0

    def compute_actor_loss(self, states, actions, advantages):
        """Implementa método abstracto — la lógica real está en update()."""
        log_probs, _ = self.actor.evaluate_actions(states, actions)
        return -(log_probs * advantages).mean()

    def compute_critic_loss(self, states, actions, rewards, next_states, dones):
        """Implementa método abstracto — la lógica real está en update()."""
        values = self.critic(states).squeeze(1)
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(1)
            next_values[dones] = 0.0
            targets = rewards + self.gamma * next_values
        return nn.MSELoss()(values, targets)

    def save(self, filepath: str) -> None:
        checkpoint = {
            'actor_state_dict':           self.actor.state_dict(),
            'critic_state_dict':          self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict':self.optimizer_critic.state_dict(),
            'training_step':              self.training_step,
            'episode_count':              self.episode_count,
            'gamma':                      self.gamma,
            'clip_epsilon':               self.clip_epsilon,
            'gae_lambda':                 self.gae_lambda,
            'hidden_dims':                self.hidden_dims,
            'n_vars':                     self.n_vars,
        }
        torch.save(checkpoint, filepath)
        print(f"PPO Agent guardado en: {filepath}")

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"PPO Agent cargado desde: {filepath}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'training_step':  self.training_step,
            'episode_count':  self.episode_count,
            'buffer_size':    len(self.buffer),
            'actor_params':   sum(p.numel() for p in self.actor.parameters()),
            'critic_params':  sum(p.numel() for p in self.critic.parameters()),
            'device':         str(self.device),
            'gamma':          self.gamma,
            'clip_epsilon':   self.clip_epsilon,
            'gae_lambda':     self.gae_lambda,
        }