from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import torch
import numpy as np


class AbstractPIDAgent(ABC):

    def __init__(
        self,
        state_dim: int,
        action_dim: Union[int, Tuple],
        agent_role: str,  # 'ctrl' o 'orch'
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_role = agent_role
        self.device = torch.device(device)
        
        if seed is not None:
            self.set_seed(seed)
        
        self.training_step = 0
        self.episode_count = 0
    
    @abstractmethod
    def select_action(
        self,                       
        state: np.ndarray,          
        training: bool = True
    ) -> Union[int, np.ndarray]:
        pass
    
    @abstractmethod
    def update(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def save(self, filepath: str) -> None:
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> None:
        pass
    
    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        # Convierte tensor a numpy array
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Remover batch dimension
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        
        return action  
    
    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        # Remover batch dimension si existe
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        
        # Asegurar que sea al menos 1D (no scalar)
        if action.shape == ():
            action = action.reshape(1)

        return action  # Siempre devuelve np.ndarray consistente con PIDComponents_translate

    def preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """Procesa el estado antes de pasarlo a la red neuronal.

        Convierte numpy arrays a torch.FloatTensor y mueve al dispositivo.
        Añade dimensión de batch si es necesario.
        """
        # Manejar numpy arrays, tensores o secuencias
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            # Asegurar dtype y device
            state_tensor = state.float().to(self.device)
        else:
            # Intentar construir tensor desde secuencia/valor
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)

        # Normalizar dimensiones:
        # - Si es un estado 1D -> (state_dim,) -> agregar batch -> (1, state_dim)
        # - Si ya es batch (2D) -> dejar como está
        if state_tensor.dim() == 0:
            # Escalar/valor escalar -> convertir a (1,1)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
        elif state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        return state_tensor

    def get_training_info(self) -> Dict[str, Any]:
        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'device': str(self.device),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'agent_role': self.agent_role
        }


class AbstractPolicyGradientAgent(AbstractPIDAgent):

    @abstractmethod
    def compute_policy_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        advantages: torch.Tensor
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_value_loss(
        self, 
        states: torch.Tensor, 
        returns: torch.Tensor
    ) -> torch.Tensor:
        pass


class AbstractValueBasedAgent(AbstractPIDAgent):
    def __init__(
        self,
        state_dim: int,           
        action_dim: int,          
        agent_role: str,          
        device: str = 'cpu',
        seed: Optional[int] = None,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
       
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            agent_role=agent_role,  
            device=device,
            seed=seed
        )
        
        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
    
    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_epsilon(self) -> float:
        return self.epsilon
    
    def reset_epsilon(self, epsilon_start: float = 1.0) -> None:
        self.epsilon = epsilon_start
    
    @abstractmethod
    def compute_q_loss(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor,
        rewards: torch.Tensor, 
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        pass


class AbstractActorCriticAgent(AbstractPIDAgent):

    @abstractmethod
    def compute_actor_loss(self, states: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def compute_critic_loss(self, states: torch.Tensor, actions: torch.Tensor,
                           rewards: torch.Tensor, next_states: torch.Tensor,
                           dones: torch.Tensor) -> torch.Tensor:
        pass