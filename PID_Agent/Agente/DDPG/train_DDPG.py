import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import wandb

from Environment import PIDControlEnv_simple, PIDControlEnv_complex
from .algorithm_DDPG import DDPGAgent
from ..memory import Experience, SimpleReplayBuffer
from ..DQN.algorithm_DQN import DQNAgent
from ..memory import SimpleReplayBuffer, PriorityReplayBuffer


class DDPGTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.architecture = config['env_config']['architecture']

        # AMBIENTE (igual que DQNTrainer)
        if self.architecture == 'simple':
            self.env = PIDControlEnv_simple.PIDControlEnv_Simple(config['env_config'])
        elif self.architecture == 'jerarquica':
            self.env = PIDControlEnv_complex.PIDControlEnv_Complex(config['env_config'])

        # AGENTES
        if self.architecture == 'simple':
            self.agent_role = 'ctrl'
            self.agent_orch = None
            self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')

        elif self.architecture == 'jerarquica':
            # CTRL pre-entrenado (puede ser cualquiera aca dejo DQN para ejemplo, pero se puede usar cualquier agente pre-entrenado)
            ctrl_checkpoint = config.get('ctrl_checkpoint_path', None)
            if not ctrl_checkpoint:
                raise ValueError("Arquitectura jerárquica requiere ctrl_checkpoint_path")

            print(f"Cargando agente CTRL pre-entrenado desde: {ctrl_checkpoint}")
            self.agent_role = 'ctrl'

            # Soporte para CTRL DQN pre-entrenado
            ctrl_algo = config['agent_ctrl_config'].get('algorithm', 'dqn')
            if ctrl_algo == 'dqn':

                buffer_type = config['agent_ctrl_config'].get('buffer_type', 'simple')
                buffer_size = config['agent_ctrl_config'].get('buffer_size', 10000)
                device = config['agent_ctrl_config'].get('device', 'cpu')
                if buffer_type == 'priority':
                    rb = PriorityReplayBuffer(capacity=buffer_size, device=device)
                else:
                    rb = SimpleReplayBuffer(capacity=buffer_size, device=device)
                self.agent_ctrl = DQNAgent(
                    state_dim=config['agent_ctrl_config']['state_dim'],
                    action_dim=config['agent_ctrl_config']['action_dim'],
                    agent_role='ctrl',
                    n_vars=config['agent_ctrl_config']['n_vars'],
                    hidden_dims=config['agent_ctrl_config'].get('hidden_dims', (128, 64)),
                    device=device,
                    replay_buffer=rb
                )
            else:
                self.agent_ctrl = self._create_agent(config['agent_ctrl_config'], 'ctrl')

            self.agent_ctrl.load(ctrl_checkpoint)
            self.env.agente_ctrl = self.agent_ctrl
            self.env.action_type_ctrl = config['agent_ctrl_config'].get('action_type', 'discrete')

            # ORCH DDPG (a entrenar)
            self.agent_role = 'orch'
            self.agent_orch = self._create_agent(config['agent_orch_config'], 'orch')

        # ENTRENAMIENTO
        self.n_episodes = config.get('n_episodes', 1000)
        self.max_steps_per_episode = config['env_config'].get('max_steps', 200)
        self.eval_freq = config.get('eval_frequency', 50)
        self.save_freq = config.get('save_frequency', 100)
        self.log_freq = config.get('log_frequency', 10)

        # EARLY STOPPING
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        early_stopping_min_delta_pct = config.get('early_stopping_min_delta_pct', 0.01)
        sum_weights = sum(self.env.reward_calculator.weights.values())
        self.early_stopping_min_delta = early_stopping_min_delta_pct * sum_weights * 1.5
        self.best_eval_reward = -float('inf')
        self.evals_without_improvement = 0

        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # W&B logging
        self.use_wandb = config.get('use_wandb', False)

        # ESTADÍSTICAS 
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_energies = []
        self.episode_max_overshoots = []
        self.actor_losses = []
        self.critic_losses = []
        self.kp_history = []
        self.ki_history = []
        self.kd_history = []
        self.eval_trajectories = []

        self.best_reward = -float('inf')

    def _create_agent(self, agent_config: Dict[str, Any], agent_role: str) -> DDPGAgent:
        buffer_size = agent_config.get('buffer_size', 100000)
        device = agent_config.get('device', 'cpu')
        replay_buffer = SimpleReplayBuffer(capacity=buffer_size, device=device)

        agent = DDPGAgent(
            state_dim=agent_config['state_dim'],
            action_dim=agent_config['action_dim'],
            agent_role=agent_role,
            n_vars=agent_config['n_vars'],
            hidden_dims=agent_config.get('hidden_dims', (128, 128, 64)),
            lr_actor=agent_config.get('lr_actor', 0.0001),
            lr_critic=agent_config.get('lr_critic', 0.001),
            gamma=agent_config.get('gamma', 0.99),
            tau=agent_config.get('tau', 0.005),
            noise_sigma=agent_config.get('noise_sigma', 0.2),
            noise_theta=agent_config.get('noise_theta', 0.15),
            replay_buffer=replay_buffer,
            batch_size=agent_config.get('batch_size', 64),
            warmup_steps=agent_config.get('warmup_steps', 1000),
            device=device,
            seed=agent_config.get('seed', None)
        )
        return agent

    def train(self):
        for episode in range(self.n_episodes):
            episode_reward, episode_length, episode_metrics = self._run_episode(episode, training=True)

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_energies.append(episode_metrics.get('energy', 0))
            self.episode_max_overshoots.append(episode_metrics.get('max_overshoot', 0))
            self.actor_losses.append(episode_metrics.get('actor_loss', 0))
            self.critic_losses.append(episode_metrics.get('critic_loss', 0))

            if self.architecture == 'simple':
                n_vars = len(self.env.pid_controllers)
                if not self.kp_history:
                    self.kp_history = [[] for _ in range(n_vars)]
                    self.ki_history = [[] for _ in range(n_vars)]
                    self.kd_history = [[] for _ in range(n_vars)]
                for i in range(n_vars):
                    self.kp_history[i].append(episode_metrics.get(f'kp_var{i}', 1.0))
                    self.ki_history[i].append(episode_metrics.get(f'ki_var{i}', 0.1))
                    self.kd_history[i].append(episode_metrics.get(f'kd_var{i}', 0.01))

            if episode % self.log_freq == 0:
                self._log_episode(episode, episode_reward, episode_length, episode_metrics)

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

            if self.use_wandb:
                log_dict = {
                    'reward':      episode_reward,
                    'energy':      episode_metrics.get('energy', 0),
                    'overshoot':   episode_metrics.get('max_overshoot', 0),
                    'actor_loss':  episode_metrics.get('actor_loss', 0),
                    'critic_loss': episode_metrics.get('critic_loss', 0),
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

        # Resetear ruido OU al inicio de cada episodio
        if self.architecture == 'simple' and hasattr(self.agent_ctrl, 'reset_noise'):
            self.agent_ctrl.reset_noise()
        elif self.architecture == 'jerarquica' and hasattr(self.agent_orch, 'reset_noise'):
            self.agent_orch.reset_noise()

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
        actor_losses = []
        critic_losses = []
        episode_energy = 0
        episode_max_overshoot = 0

        while not done and episode_length < self.max_steps_per_episode:

            # ARQUITECTURA SIMPLE (CTRL DDPG)
            if self.architecture == 'simple':
                action = self.agent_ctrl.select_action(state, training=training)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                pv_history_episode.append(self.env.manipulable_pvs.copy())
                sp_history_episode.append(self.env.manipulable_setpoints.copy())
                done = terminated or truncated

                if training:
                    experience = Experience(state, action, reward, next_state, done)
                    self.agent_ctrl.memory.add(experience)
                    metrics = self.agent_ctrl.update()
                    if metrics:
                        actor_losses.append(metrics.get('actor_loss', 0))
                        critic_losses.append(metrics.get('critic_loss', 0))

                episode_energy += info.get('energy', 0)
                if 'overshoot_manipulable' in info:
                    current_overshoots = info['overshoot_manipulable']
                    episode_max_overshoot = max(episode_max_overshoot, max(current_overshoots))
                state = next_state

            # ARQUITECTURA JERÁRQUICA (ORCH DDPG + CTRL pre-entrenado)
            else:
                # ORCH decide deltas de SP (acción continua)
                action_orch = self.agent_orch.select_action(state_orch, training=training)

                # CTRL decide ajuste PID (pre-entrenado, sin exploración)
                action_ctrl = self.agent_ctrl.select_action(state_ctrl, training=False)

                action = {'ctrl': action_ctrl, 'orch': action_orch}

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_state_ctrl = next_obs['ctrl']
                next_state_orch = next_obs['orch']

                if training:
                    exp_orch = Experience(state_orch, action_orch, reward, next_state_orch, done)
                    self.agent_orch.memory.add(exp_orch)
                    metrics = self.agent_orch.update()
                    if metrics:
                        actor_losses.append(metrics.get('actor_loss', 0))
                        critic_losses.append(metrics.get('critic_loss', 0))

                state_ctrl = next_state_ctrl
                state_orch = next_state_orch

            episode_reward += reward
            episode_length += 1

        # Compilar métricas
        episode_metrics = {
            'actor_loss': np.mean(actor_losses) if actor_losses else 0,
            'critic_loss': np.mean(critic_losses) if critic_losses else 0,
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
        
        if self.use_wandb:
            wandb.log({'eval_reward': mean_reward})

        mean_reward = np.mean(eval_rewards)
        print(f"Evaluación: Reward promedio = {mean_reward:.2f}")

        if mean_reward > self.best_eval_reward + self.early_stopping_min_delta:
            self.best_eval_reward = mean_reward
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
            print(f"  Sin mejora: {self.evals_without_improvement}/{self.early_stopping_patience}")

        return mean_reward, self.evals_without_improvement >= self.early_stopping_patience
    
    def _log_episode(self, episode: int, reward: float, length: int, metrics: Dict[str, float]):
        print(f"\nEpisodio {episode}/{self.n_episodes}")
        print(f"  Reward: {reward:.2f} | Length: {length}")
        print(f"  Actor Loss: {metrics['actor_loss']:.4f} | Critic Loss: {metrics['critic_loss']:.4f}")
        if self.architecture == 'jerarquica':
            mem = len(self.agent_orch.memory) if self.agent_orch else 0
            print(f"  ORCH Memory: {mem}")

    def _save_checkpoint(self, episode: int, best: bool = False):
        suffix = 'best' if best else f'ep{episode}'
        if self.agent_ctrl is not None:
            ctrl_path = self.checkpoint_dir / f'agent_ctrl_{suffix}.pt'
            self.agent_ctrl.save(str(ctrl_path))
        if self.agent_orch is not None:
            orch_path = self.checkpoint_dir / f'agent_orch_{suffix}.pt'
            self.agent_orch.save(str(orch_path))
        print(f"Checkpoint guardado: {suffix}")
