import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional

from .model_DQN import DQN_Network
from ..memory import AbstractReplayBuffer, SimpleReplayBuffer
from ..abstract_agent import AbstractValueBasedAgent


class DQNAgent(AbstractValueBasedAgent):
    def __init__(
        self,
        state_dim: int,          
        action_dim: int,
        n_vars: int,
        agent_role: str,
        hidden_dims: tuple = (128, 128, 64),
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        replay_buffer: Optional[AbstractReplayBuffer] = None,
        memory_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = 'cpu',
        seed: Optional[int] = None
    ):

        # Llamar al constructor padre
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_role=agent_role,
            device=device,
            seed=seed,
            epsilon_start=epsilon_start,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay
        )
        
        # Parámetros específicos de DQN
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_dims = hidden_dims
        self.n_vars = n_vars
        
        # Redes neuronales
        self.q_network = DQN_Network(
            state_dim=state_dim,
            n_actions=action_dim,
            n_vars=n_vars,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.target_network = DQN_Network(
            state_dim=state_dim,
            n_actions=action_dim,
            n_vars=n_vars,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        # Replay buffer
        if replay_buffer is not None:
            self.memory = replay_buffer  
        else:
            # Default: SimpleReplayBuffer
            self.memory = SimpleReplayBuffer(capacity=memory_size, device=device)

        # Copiar pesos a red objetivo
        self.update_target_network()
        
        # Optimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True
    ) -> np.ndarray:
        
        # Preprocesar estado completo
        state_tensor = self.preprocess_state(state)  # (1, state_dim)
        
        # Epsilon-greedy
        if training and np.random.random() < self.get_epsilon():
            actions = np.random.randint(0, self.action_dim, size=self.n_vars)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)  # (1, n_vars, n_actions)
                actions = q_values.argmax(dim=2).squeeze(0).cpu().numpy()  # (n_vars,)
        
        return actions.astype(np.int64)
    
    
    def update(self, batch_data: Dict[str, Any] = None) -> Dict[str, float]:
        # Verificar si hay suficientes experiencias
        if len(self.memory) < self.batch_size:
            return {}
        
        # Muestrear batch del buffer
        batch = self.memory.sample(self.batch_size)
        
        states = batch['states']
        actions = batch['actions'].long()  # (batch, n_vars)
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Usar compute_q_loss nuevo
        loss = self.compute_q_loss(states, actions, rewards, next_states, dones)
        
        # PARA PRIORITY BUFFER: calcular TD errors
        if 'weights' in batch:
            weights = batch['weights']
            
            # Q-values actuales
            q_values = self.q_network(states)  # (batch, n_vars, n_actions)
            actions_unsqueezed = actions.unsqueeze(-1)
            current_q = q_values.gather(2, actions_unsqueezed).squeeze(-1)
            current_q_combined = current_q.mean(dim=1)
            
            # Q-values objetivo
            with torch.no_grad():
                next_q_values = self.target_network(next_states)
                next_q_max = next_q_values.max(dim=2)[0]
                next_q_combined = next_q_max.mean(dim=1)
                target_q = rewards + (self.gamma * next_q_combined * ~dones)
                target_q = target_q.clamp(-50.0, 0.0) 
            
            # TD errors para priority buffer
            td_errors_tensor = current_q_combined - target_q
            loss = (weights * (td_errors_tensor ** 2)).mean()
            
            # Actualizar prioridades
            if hasattr(self.memory, 'update_priorities'):
                td_errors_np = td_errors_tensor.abs().detach().cpu().numpy()
                self.memory.update_priorities(batch['indices'], td_errors_np)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Actualizar epsilon
        self.update_epsilon()
        
        # Actualizar red objetivo periódicamente
        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
        
        # Incrementar contador
        self.training_step += 1
        
        # Devolver métricas
        return {
            'q_loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'training_step': self.training_step,
            'memory_size': len(self.memory)
        }
        
    def compute_q_loss(
        self,
        states: torch.Tensor,      # (batch, state_dim)
        actions: torch.Tensor,     # (batch, n_vars)
        rewards: torch.Tensor,     # (batch,)
        next_states: torch.Tensor, # (batch, state_dim)
        dones: torch.Tensor        # (batch,)
    ) -> torch.Tensor:
       
        # Q-values actuales: (batch, n_vars, n_actions)
        q_values = self.q_network(states)
        
        # Seleccionar Q-values de las acciones tomadas
        # actions: (batch, n_vars) → (batch, n_vars, 1)
        actions_unsqueezed = actions.unsqueeze(-1)
        
        # Gather Q-values: (batch, n_vars, 1) → (batch, n_vars)
        current_q = q_values.gather(2, actions_unsqueezed).squeeze(-1)
        
        # Combinar Q-values de todas las variables (PROMEDIO)
        current_q_combined = current_q.mean(dim=1)  # (batch,)
        
        # Q-values objetivo
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # (batch, n_vars, n_actions)
            
            # Mejor acción por variable
            next_q_max = next_q_values.max(dim=2)[0]  # (batch, n_vars)
            
            # Combinar (PROMEDIO)
            next_q_combined = next_q_max.mean(dim=1)  # (batch,)
            
            # Target Q-value
            target_q = rewards + (self.gamma * next_q_combined * ~dones)
            target_q = target_q.clamp(-50.0, 0.0) # para que no explote
        
        # MSE loss
        loss = nn.MSELoss()(current_q_combined, target_q)
        
        return loss
    
    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    
    def save(self, filepath: str) -> None:
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.get_epsilon(),
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'hidden_dims': self.hidden_dims,
            'n_vars': self.n_vars
        }
        
        torch.save(checkpoint, filepath)
        print(f"Agente guardado en: {filepath}")
    
    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Cargar redes
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restaurar parámetros
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.epsilon_min = checkpoint.get('epsilon_min', self.epsilon_min)
        self.epsilon_decay = checkpoint.get('epsilon_decay', self.epsilon_decay)
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'epsilon': self.get_epsilon(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'memory_size': len(self.memory),
            'memory_capacity': self.memory.capacity,
            'network_params': sum(p.numel() for p in self.q_network.parameters()),
            'device': str(self.device),
            'gamma': self.gamma,
            'batch_size': self.batch_size
        }