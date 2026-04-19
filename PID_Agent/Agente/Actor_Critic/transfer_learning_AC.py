"""
Utilidad de Transfer Learning para el agente Actor-Critic.

Estrategia:
    1. Cargar pesos pre-entrenados del AC (reactor CSTR original)
    2. Transferir directamente (mismas dimensiones: state=10, action=6)
    3. Opcionalmente congelar capas tempranas (feature extraction)
    4. Fine-tunar con learning rates reducidos

Justificación:
    - Ambos problemas son CSTR con 2 variables controladas y 2 actuadores
    - El agente aprendió patrones generales de ajuste PID que son transferibles:
        * Cómo responder a errores grandes vs pequeños
        * Cómo balancear Kp, Ki, Kd
        * Cuándo ser agresivo vs conservador
    - Las capas tempranas extraen features del estado (error, integral, derivada)
      que son conceptualmente iguales entre ambos reactores
    - Las capas finales mapean features a acciones PID específicas
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import copy


def load_pretrained_ac(
    agent_class,
    checkpoint_path: str,
    state_dim: int = 10,
    action_dim: int = 6,
    agent_role: str = 'ctrl',
    n_vars: int = 2,
    hidden_dims: tuple = (128, 128, 64),
    device: str = 'cpu',
    **agent_kwargs
):
    """
    Carga un agente AC pre-entrenado desde un checkpoint.
    
    Args:
        agent_class: Clase ACAgent
        checkpoint_path: Ruta al checkpoint (.pt)
        state_dim: Dimensión del estado (debe coincidir con el pre-entrenado)
        action_dim: Dimensión de la acción (debe coincidir con el pre-entrenado)
        agent_role: Rol del agente ('ctrl')
        n_vars: Número de variables
        hidden_dims: Arquitectura de la red (debe coincidir)
        device: Dispositivo ('cpu' o 'cuda')
    
    Returns:
        agent: Agente AC con pesos cargados
    """
    agent = agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        agent_role=agent_role,
        n_vars=n_vars,
        hidden_dims=hidden_dims,
        device=device,
        **agent_kwargs
    )
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Verificar compatibilidad de dimensiones
    actor_sd = checkpoint['actor_state_dict']
    first_layer_key = 'shared_network.0.weight'
    if first_layer_key in actor_sd:
        pretrained_state_dim = actor_sd[first_layer_key].shape[1]
        pretrained_hidden = actor_sd[first_layer_key].shape[0]
        if pretrained_state_dim != state_dim:
            raise ValueError(
                f"Dimensión de estado incompatible: pre-entrenado={pretrained_state_dim}, "
                f"nuevo={state_dim}. Transfer learning directo no es posible."
            )
        print(f"Dimensiones compatibles: state_dim={state_dim}, hidden[0]={pretrained_hidden}")
    
    # Cargar pesos
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # NO cargar optimizadores (queremos lr fresco para fine-tuning)
    agent.training_step = checkpoint.get('training_step', 0)
    agent.episode_count = checkpoint.get('episode_count', 0)
    
    print(f"   Pesos pre-entrenados cargados desde: {checkpoint_path}")
    print(f"   Training steps previos: {agent.training_step}")
    print(f"   Episodios previos: {agent.episode_count}")
    
    return agent


def freeze_layers(
    agent,
    freeze_strategy: str = 'early',
    n_freeze: int = 2
):
    """
    Congela capas del agente para transfer learning.
    
    Estrategias:
        'early': Congela las primeras n_freeze capas de shared_network
                 (feature extraction). Fine-tunea las últimas capas y 
                 las cabezas mu/log_std.
        
        'critic_only': Congela todo el actor, solo fine-tunea el critic.
                       Útil si el actor ya es bueno y solo necesita
                       recalibrar las estimaciones de valor.
        
        'none': No congela nada. Fine-tuning completo.
    
    Args:
        agent: Agente AC
        freeze_strategy: 'early', 'critic_only', o 'none'
        n_freeze: Número de capas a congelar (para 'early')
    
    Returns:
        frozen_params: Lista de nombres de parámetros congelados
    """
    frozen_params = []
    
    if freeze_strategy == 'none':
        print(" Sin capas congeladas — fine-tuning completo")
        return frozen_params
    
    elif freeze_strategy == 'early':
        # Congelar primeras n_freeze capas del shared_network en actor y critic
        # shared_network = [Linear, ReLU, Linear, ReLU, Linear, ReLU]
        # Cada "capa" son 2 módulos (Linear + ReLU)
        modules_to_freeze = n_freeze * 2  # Linear + ReLU por capa
        
        # Actor - shared_network
        for idx, (name, param) in enumerate(agent.actor.shared_network.named_parameters()):
            layer_idx = int(name.split('.')[0])
            if layer_idx < modules_to_freeze:
                param.requires_grad = False
                frozen_params.append(f'actor.shared_network.{name}')
        
        # Critic - network (primeras capas)
        for idx, (name, param) in enumerate(agent.critic.network.named_parameters()):
            layer_idx = int(name.split('.')[0])
            if layer_idx < modules_to_freeze:
                param.requires_grad = False
                frozen_params.append(f'critic.network.{name}')
        
        print(f"Congeladas {n_freeze} capas tempranas ({len(frozen_params)} parámetros)")
        
    elif freeze_strategy == 'critic_only':
        # Congelar todo el actor
        for name, param in agent.actor.named_parameters():
            param.requires_grad = False
            frozen_params.append(f'actor.{name}')
        
        print(f"Actor completamente congelado ({len(frozen_params)} parámetros)")
    
    # Recrear optimizadores solo con parámetros entrenables
    trainable_actor = [p for p in agent.actor.parameters() if p.requires_grad]
    trainable_critic = [p for p in agent.critic.parameters() if p.requires_grad]
    
    if trainable_actor:
        agent.optimizer_actor = torch.optim.Adam(
            trainable_actor, 
            lr=agent.optimizer_actor.param_groups[0]['lr']
        )
    
    if trainable_critic:
        agent.optimizer_critic = torch.optim.Adam(
            trainable_critic, 
            lr=agent.optimizer_critic.param_groups[0]['lr']
        )
    
    # Resumen
    total_actor = sum(p.numel() for p in agent.actor.parameters())
    trainable_actor_n = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    total_critic = sum(p.numel() for p in agent.critic.parameters())
    trainable_critic_n = sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)
    
    print(f"   Actor:  {trainable_actor_n}/{total_actor} parámetros entrenables")
    print(f"   Critic: {trainable_critic_n}/{total_critic} parámetros entrenables")
    
    return frozen_params


def setup_transfer_learning(
    agent_class,
    checkpoint_path: str,
    lr_actor: float = 0.00003,      # 3x menor que el original (0.0001)
    lr_critic: float = 0.0003,      # 3x menor que el original (0.001)
    freeze_strategy: str = 'early',
    n_freeze: int = 2,
    entropy_coef: float = 0.02,     # Más exploración al inicio del fine-tuning
    state_dim: int = 10,
    action_dim: int = 6,
    n_vars: int = 2,
    hidden_dims: tuple = (128, 128, 64),
    buffer_size: int = 50000,
    batch_size: int = 64,
    warmup_steps: int = 200,        # Menos warmup (ya tiene experiencia)
    device: str = 'cpu'
) -> Any:
    """
    Configura un agente AC para transfer learning en un solo paso.
    
    Combina: cargar pesos + congelar capas + ajustar hiperparámetros.
    
    Args:
        agent_class: Clase ACAgent
        checkpoint_path: Ruta al checkpoint del agente pre-entrenado
        lr_actor: Learning rate del actor (recomendado: 3-10x menor que original)
        lr_critic: Learning rate del critic (recomendado: 3-10x menor que original)
        freeze_strategy: 'early', 'critic_only', o 'none'
        n_freeze: Capas a congelar (para 'early')
        entropy_coef: Coeficiente de entropía (mayor = más exploración)
        state_dim: Dimensión del estado
        action_dim: Dimensión de la acción
        n_vars: Número de variables controladas
        hidden_dims: Arquitectura de la red
        buffer_size: Tamaño del replay buffer
        batch_size: Tamaño del batch
        warmup_steps: Steps antes de empezar a entrenar
        device: Dispositivo
    
    Returns:
        agent: Agente listo para fine-tuning
    """
    print("=" * 60)
    print(" CONFIGURANDO TRANSFER LEARNING")
    print("=" * 60)
    
    # 1. Cargar agente pre-entrenado
    agent = load_pretrained_ac(
        agent_class=agent_class,
        checkpoint_path=checkpoint_path,
        state_dim=state_dim,
        action_dim=action_dim,
        n_vars=n_vars,
        hidden_dims=hidden_dims,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        entropy_coef=entropy_coef,
        buffer_size=buffer_size,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        device=device
    )
    
    # 2. Congelar capas
    frozen = freeze_layers(agent, freeze_strategy, n_freeze)
    
    # 3. Limpiar replay buffer (experiencia del reactor anterior no aplica)
    agent.memory.clear()
    print(f"Replay buffer limpiado")
    
    # 4. Resetear contadores de entrenamiento (nuevo problema)
    agent.training_step = 0
    agent.episode_count = 0
    print(f"Contadores reseteados")
    
    print("=" * 60)
    print("Agente listo para fine-tuning")
    print(f"   LR Actor:  {lr_actor} | LR Critic: {lr_critic}")
    print(f"   Entropía:  {entropy_coef}")
    print(f"   Warmup:    {warmup_steps} steps")
    print(f"   Estrategia: {freeze_strategy}" + 
          (f" ({n_freeze} capas)" if freeze_strategy == 'early' else ""))
    print("=" * 60)
    
    return agent


def compare_agents(
    agent_pretrained,
    agent_finetuned,
    layer_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compara pesos entre agente pre-entrenado y fine-tuneado.
    Útil para verificar cuánto cambió cada capa durante el fine-tuning.
    
    Returns:
        Dict con la norma de la diferencia por capa
    """
    diffs = {}
    
    for (name1, p1), (name2, p2) in zip(
        agent_pretrained.actor.named_parameters(),
        agent_finetuned.actor.named_parameters()
    ):
        if layer_names and not any(ln in name1 for ln in layer_names):
            continue
        diff = (p1.data - p2.data).norm().item()
        diffs[f'actor.{name1}'] = diff
    
    for (name1, p1), (name2, p2) in zip(
        agent_pretrained.critic.named_parameters(),
        agent_finetuned.critic.named_parameters()
    ):
        if layer_names and not any(ln in name1 for ln in layer_names):
            continue
        diff = (p1.data - p2.data).norm().item()
        diffs[f'critic.{name1}'] = diff
    
    return diffs


def get_transfer_config(
    checkpoint_path: str,
    reactor_type: str = 'cyclopentanol'
) -> Dict[str, Any]:
    """
    Genera la configuración completa para entrenar con transfer learning
    en el reactor de Ciclopentanol.
    
    Args:
        checkpoint_path: Ruta al checkpoint del CSTR original
        reactor_type: Tipo de reactor destino
    
    Returns:
        Dict con toda la configuración necesaria
    """
    if reactor_type == 'cyclopentanol':
        config = {
            # ─── Ambiente ───
            'env_config': {
                'architecture': 'simple',
                'env_type': 'simulation',
                'n_manipulable_vars': 2,
                'manipulable_ranges': [
                    (50.0, 800.0),      # v (L/h)
                    (-8500.0, 0.0)      # QK (kJ/h)
                ],
                'dt_usuario': 0.01,     # h (dinámica rápida)
                'max_steps': 100,
                'max_time_detector': 5.0,  # 5 horas máximo por step
                'reward_dead_band': 0.02,
                'agent_controller_config': {
                    'agent_type': 'continuous'
                },
                'env_type_config': {
                    'dt_simulation': 0.01,
                    'n_manipulable_vars': 2,
                },
                'pid_limits': [
                    (0.01, 5000.0),     # Kp (rango más amplio para este reactor)
                    (0.0, 50000.0),     # Ki
                    (0.0, 100.0)        # Kd
                ],
                'reward_weights': {
                    'error': 1.0, 'tiempo': 0.001, 'overshoot': 0.3, 'energy': 0.001
                },
                'delta_percent_ctrl': 0.2,
                'stability_config': {
                    'error_increase_tolerance': 2.0,
                    'max_sign_changes_ratio': 0.3,
                    'max_abrupt_change_ratio': 0.05,
                    'abrupt_change_threshold': 0.2,
                },
            },
            
            # ─── Agente (transfer learning) ───
            # IMPORTANTE: hidden_dims DEBE coincidir con el modelo pre-entrenado
            'agent_ctrl_config': {
                'state_dim': 10,        # 5 obs × 2 vars
                'action_dim': 6,        # 3 PID × 2 vars
                'n_vars': 2,
                'hidden_dims': (64, 32),    # ← DEBE coincidir con el pre-entrenado
                'lr_actor': 3e-06,          # 3x menor que original (1e-05)
                'lr_critic': 3e-04,         # 3x menor que original (1e-03)
                'gamma': 0.95,              # Mismo que original
                'entropy_coef': 0.02,       # Más exploración al inicio
                'buffer_size': 50000,
                'batch_size': 64,
                'warmup_steps': 200,        # Menos warmup (ya tiene experiencia)
                'device': 'cpu'
            },
            
            # ─── Transfer Learning ───
            'transfer_config': {
                'checkpoint_path': checkpoint_path,
                'freeze_strategy': 'early',  # Congelar capa temprana
                'n_freeze': 1,               # 1 de 2 capas (red más chica)
            },
            
            # ─── Entrenamiento ───
            'n_episodes': 5000,
            'max_steps_per_episode': 100,
            'eval_frequency': 100,
            'save_frequency': 1000,
            'log_frequency': 100,
            'checkpoint_dir': 'checkpoints_transfer_cyclopentanol',
            'use_wandb': False,
            'early_stopping_patience': 20,
            'early_stopping_min_delta_pct': 0.01,
        }
    else:
        raise ValueError(f"Reactor type '{reactor_type}' no soportado")
    
    return config
