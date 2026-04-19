import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path
import wandb
from datetime import datetime


from Environment import PIDControlEnv_simple , PIDControlEnv_complex
from .algorithm_DQN import DQNAgent
from ..memory import Experience, SimpleReplayBuffer, PriorityReplayBuffer


class DQNTrainer:
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.architecture = config['env_config']['architecture']  # 'simple' o 'jerarquica'
        
        # AMBIENTE
        if self.architecture == 'simple':
            # PIDControlEnv_simple is a module; instantiate the class inside it
            self.env = PIDControlEnv_simple.PIDControlEnv_Simple(config['env_config'])
        
        elif self.architecture == 'jerarquica':
            # PIDControlEnv_complex is a module; instantiate the class inside it
            self.env = PIDControlEnv_complex.PIDControlEnv_Complex(config['env_config'])
        
        # ENTRENAMIENTO
        self.n_episodes = config.get('n_episodes', 1000)
        self.max_steps_per_episode = config['env_config'].get('max_steps', 200)
        self.eval_freq = config.get('eval_frequency', 50)
        self.save_freq = config.get('save_frequency', 100)
        self.log_freq = config.get('log_frequency', 10)
        
        # AGENTES
        if self.architecture == 'simple':
            # Inicializar roles y luego crear agente CTRL desde cero
            self.agent_role = 'ctrl'
            self.agent_orch = None
            self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')
        
        elif self.architecture == 'jerarquica':
            # CTRL: Cargar modelo pre-entrenado
            ctrl_checkpoint = config.get('ctrl_checkpoint_path', None)
            
            if ctrl_checkpoint:
                print(f"Cargando agente CTRL pre-entrenado desde: {ctrl_checkpoint}")
                self.agent_role = 'ctrl'
                ctrl_algo = config['agent_ctrl_config'].get('algorithm', 'dqn')
                if ctrl_algo == 'ppo':
                    from ..PPO.algorithm_PPO import PPOAgent
                    self.agent_ctrl = PPOAgent(
                        state_dim   = config['agent_ctrl_config']['state_dim'],
                        action_dim  = config['agent_ctrl_config']['action_dim'],
                        agent_role  = 'ctrl',
                        n_vars      = config['agent_ctrl_config']['n_vars'],
                        hidden_dims = config['agent_ctrl_config'].get('hidden_dims', (128, 64)),
                        device      = config['agent_ctrl_config'].get('device', 'cpu')
                    )
                else:
                    self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')

                self.agent_ctrl.load(ctrl_checkpoint)
                self.agent_ctrl.epsilon = 0.0
                self.env.agente_ctrl = self.agent_ctrl
                self.env.action_type_ctrl = config['agent_ctrl_config'].get('action_type', 'discrete')
            else:
                raise ValueError(
                    "Arquitectura jerárquica requiere agente control entrenado"
                )
            
            # ORCH
            self.agent_role = 'orch'
            self.agent_orch = self._create_agent(config['agent_orch_config'], 'orch')
        

        # EARLY STOPPING
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        early_stopping_min_delta_pct = config.get('early_stopping_min_delta_pct', 0.01)
        sum_weights = sum(self.env.reward_calculator.weights.values())
        self.early_stopping_min_delta = early_stopping_min_delta_pct * sum_weights * 1.5
        self.best_eval_reward = -float('inf')
        self.evals_without_improvement = 0
        
        # Directorios
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # W&B logging
        self.use_wandb = config.get('use_wandb', False)

        # ESTADÍSTICAS
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energies = []  
        self.episode_max_overshoots = []  
        self.epsilons = []
        self.eval_trajectories = [] 

        ##Historial de parámetros PID
        self.kp_history = []
        self.ki_history = []
        self.kd_history = [] 

        self.best_reward = -float('inf')
    
    def _create_agent(self, agent_config: Dict[str, Any], agent_type: str) -> DQNAgent:
        
        # Crear replay buffer según configuración
        buffer_type = agent_config.get('buffer_type', 'simple')
        buffer_size = agent_config.get('buffer_size', 10000)
        device = agent_config.get('device', 'cpu')
        
        if buffer_type == 'simple':
            replay_buffer = SimpleReplayBuffer(capacity=buffer_size, device=device)
        elif buffer_type == 'priority':
            replay_buffer = PriorityReplayBuffer(
                capacity=buffer_size,
                alpha=agent_config.get('priority_alpha', 0.6),
                beta=agent_config.get('priority_beta', 0.4),
                total_training_steps=self.n_episodes * self.max_steps_per_episode,
                device=device
            )
        
        # Crear agente
        agent = DQNAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            agent_role= self.agent_role,
            n_vars=agent_config['n_vars'],
            hidden_dims=agent_config.get('hidden_dims', (128, 128, 64)),
            lr=agent_config.get('lr', 0.001),
            gamma=agent_config.get('gamma', 0.99),
            epsilon_start=agent_config.get('epsilon_start', 1.0),
            epsilon_min=agent_config.get('epsilon_min', 0.01),
            epsilon_decay=agent_config.get('epsilon_decay', 0.995),
            batch_size=agent_config.get('batch_size', 32),
            target_update_freq=agent_config.get('target_update_freq', 100),
            device=device,
            seed=agent_config.get('seed', None),
            replay_buffer=replay_buffer
        )
        
        return agent
    
    def train(self):

        for episode in range(self.n_episodes):
            episode_reward, episode_length, episode_metrics = self._run_episode(episode, training=True)
            
            # Guardar estadísticas
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_energies.append(episode_metrics.get('energy', 0))  
            self.episode_max_overshoots.append(episode_metrics.get('max_overshoot', 0))  
            if self.architecture == 'simple':
                self.epsilons.append(episode_metrics.get('ctrl_epsilon', 0))
            elif self.architecture == 'jerarquica':
                self.epsilons.append(episode_metrics.get('orch_epsilon', 0))  
            
            # Guardar PID params
            n_vars = len(self.env.pid_controllers)
            # Inicializar listas por variable si no existen
            if not self.kp_history:
                self.kp_history = [[] for _ in range(n_vars)]
                self.ki_history = [[] for _ in range(n_vars)]
                self.kd_history = [[] for _ in range(n_vars)]
            for i in range(n_vars):
                self.kp_history[i].append(episode_metrics.get(f'kp_var{i}', 1.0))
                self.ki_history[i].append(episode_metrics.get(f'ki_var{i}', 0.1))
                self.kd_history[i].append(episode_metrics.get(f'kd_var{i}', 0.01))  

            # Logging
            if episode % self.log_freq == 0:
                self._log_episode(episode, episode_reward, episode_length, episode_metrics)
            
            # Evaluación
            if episode % self.eval_freq == 0 and episode > 0:
                eval_reward, stop = self._evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self._save_checkpoint(episode, best=True)
                if stop:
                    print(f"Early stopping en episodio {episode}")
                    break
            
            # Checkpoint periódico
            if episode % self.save_freq == 0 and episode > 0:
                self._save_checkpoint(episode, best=False)

            # W&B
            if self.use_wandb:
                if self.architecture == 'simple':
                    epsilon_log = episode_metrics.get('ctrl_epsilon', 0)
                    loss_log    = episode_metrics.get('ctrl_loss', 0)
                else:
                    epsilon_log = episode_metrics.get('orch_epsilon', 0)
                    loss_log    = episode_metrics.get('orch_loss', 0)

                log_dict = {
                    'reward'   : episode_reward,
                    'energy'   : episode_metrics.get('energy', 0),
                    'overshoot': episode_metrics.get('max_overshoot', 0),
                    'epsilon'  : epsilon_log,
                    'loss'     : loss_log,
                }
                if self.architecture == 'simple' and self.kp_history:
                    for i in range(len(self.kp_history)):
                        log_dict[f'kp_var{i}'] = self.kp_history[i][-1]
                        log_dict[f'ki_var{i}'] = self.ki_history[i][-1]
                        log_dict[f'kd_var{i}'] = self.kd_history[i][-1]
                wandb.log(log_dict, step=episode)      

        

    def _run_episode(self, episode: int, training: bool = True) -> tuple:

        pv_history_episode = []
        sp_history_episode = []

        # Reset ambiente
        if self.architecture == 'simple':
            state = self.env.reset()[0]
        else:
            obs = self.env.reset()[0]
            state_ctrl = obs['ctrl']
            state_orch = obs['orch']
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Métricas acumuladas
        ctrl_losses = []
        orch_losses = []
        episode_energy = 0  
        episode_max_overshoot = 0  
        
        while not done and episode_length < self.max_steps_per_episode:
            
            # ARQUITECTURA SIMPLE
            if self.architecture == 'simple':
                action = self.agent_ctrl.select_action(state, training=training)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                pv_history_episode.append(self.env.manipulable_pvs.copy())
                sp_history_episode.append(self.env.manipulable_setpoints.copy())
                done = terminated or truncated
                
                # Almacenar experiencia
                if training:
                    experience = Experience(state, action, reward, next_state, done)
                    self.agent_ctrl.memory.add(experience)
                    
                    # Actualizar agente
                    metrics = self.agent_ctrl.update()
                    if metrics:
                        ctrl_losses.append(metrics.get('q_loss', 0))
                
                # ACUMULAR MÉTRICAS DEL EPISODIO
                episode_energy += info.get('energy', 0)
                if 'overshoot_manipulable' in info:
                    current_overshoots = info['overshoot_manipulable']
                    episode_max_overshoot = max(episode_max_overshoot, max(current_overshoots))
                
                state = next_state
            
            # ARQUITECTURA JERÁRQUICA
            else:
                # 1. ORCH decide setpoints
                action_orch = self.agent_orch.select_action(state_orch, training=training)
                
                # 2. CTRL ajusta parámetros PID para alcanzar setpoints
                action_ctrl = self.agent_ctrl.select_action(state_ctrl, training=training)
                
                # 3. Combinar acciones 
                action = {
                    'ctrl': action_ctrl,
                    'orch': action_orch
                }
                
                # Ejecutar en ambiente
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                next_state_ctrl = next_obs['ctrl']
                next_state_orch = next_obs['orch']
                
                # Almacenar experiencias
                if training:
                    # Experiencia ORCH
                    exp_orch = Experience(state_orch, action_orch, reward, next_state_orch, done)
                    self.agent_orch.memory.add(exp_orch)
                    
                    # Actualizar agentes
                    metrics_orch = self.agent_orch.update()
                    if metrics_orch:
                        orch_losses.append(metrics_orch.get('q_loss', 0))
                
                state_ctrl = next_state_ctrl
                state_orch = next_state_orch
            
            episode_reward += reward
            episode_length += 1
        
        # Compilar métricas del episodio        
        episode_metrics = {
            'ctrl_loss': np.mean(ctrl_losses) if ctrl_losses else 0,
            'ctrl_epsilon': self.agent_ctrl.get_epsilon(),
            'energy': episode_energy,  
            'max_overshoot': episode_max_overshoot,
            'pv_history': pv_history_episode,
            'sp_history': sp_history_episode
        }

        if self.architecture == 'simple':
            for i, pid in enumerate(self.env.pid_controllers):
                params = pid.get_params()
                episode_metrics[f'kp_var{i}'] = params[0]
                episode_metrics[f'ki_var{i}'] = params[1]
                episode_metrics[f'kd_var{i}'] = params[2]


        if self.architecture == 'jerarquica':
            episode_metrics.update({
                'orch_loss': np.mean(orch_losses) if orch_losses else 0,
                'orch_epsilon': self.agent_orch.get_epsilon(),
            })
        # Normalizar por longitud del episodio
        episode_reward = episode_reward / episode_length if episode_length > 0 else 0

        return episode_reward, episode_length, episode_metrics
    
    def _evaluate(self, n_eval_episodes: int = 5) -> float:
        eval_rewards = []
        for idx in range(n_eval_episodes):
            episode_reward, _, episode_metrics = self._run_episode(episode=-1, training=False)
            eval_rewards.append(episode_reward)
            if idx == 0:
                self.eval_trajectories.append({
                    'episode': len(self.episode_rewards),
                    'pv_history': episode_metrics['pv_history'],
                    'sp_history': episode_metrics['sp_history']
                })

        # PRIMERO calcular mean_reward
        mean_reward = np.mean(eval_rewards)
        print(f"Evaluación: Reward promedio = {mean_reward:.2f}")

        # DESPUÉS early stopping
        if mean_reward > self.best_eval_reward + self.early_stopping_min_delta:
            self.best_eval_reward = mean_reward
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
            print(f"  Sin mejora: {self.evals_without_improvement}/{self.early_stopping_patience}")

        # DESPUÉS wandb
        if self.use_wandb:
            wandb.log({'eval_reward': mean_reward})

        return mean_reward, self.evals_without_improvement >= self.early_stopping_patience
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float]):
        """Logging de episodio."""
        print(f"\nEpisodio {episode}/{self.n_episodes}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Length: {length}")
        print(f"  CTRL Loss: {metrics['ctrl_loss']:.4f}")
        print(f"  CTRL Epsilon: {metrics['ctrl_epsilon']:.4f}")
        
        if self.architecture == 'jerarquica':
            print(f"  ORCH Loss: {metrics['orch_loss']:.4f}")
            print(f"  ORCH Epsilon: {metrics['orch_epsilon']:.4f}")

    def _save_checkpoint(self, episode: int, best: bool = False):
        """Guardar checkpoint."""
        suffix = 'best' if best else f'ep{episode}'
        
        # Guardar CTRL
        ctrl_path = self.checkpoint_dir / f'agent_ctrl_{suffix}.pt'
        self.agent_ctrl.save(str(ctrl_path))
        
        # Guardar ORCH si existe
        if self.agent_orch is not None:
            orch_path = self.checkpoint_dir / f'agent_orch_{suffix}.pt'
            self.agent_orch.save(str(orch_path))
        
        print(f"Checkpoint guardado: {suffix}")

