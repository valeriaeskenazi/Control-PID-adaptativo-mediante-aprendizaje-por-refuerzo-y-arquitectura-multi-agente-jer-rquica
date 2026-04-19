import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Optional


class SimplePlotter:

    def __init__(self, save_dir: Optional[str] = None):
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'primary':   '#2E86AB',
            'secondary': '#F18F01',
            'success':   '#06A77D',
            'danger':    '#C73E1D',
            'gray':      '#5A5A5A'
        }

    # -------------------------------------------------------------------------
    # 1. TRAINING OVERVIEW
    # -------------------------------------------------------------------------

    def plot_training_overview(
        self,
        episode_rewards: List[float],
        episode_energies: List[float],
        episode_max_overshoots: List[float],
        window: int = 20,
        epsilons: Optional[List[float]] = None,
        actor: Optional[List[float]] = None,
        critic: Optional[List[float]] = None,
    ):
        """
        Resumen general del entrenamiento: rewards, energía, overshoot y
        exploración/losses según el algoritmo.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        episodes = np.arange(len(episode_rewards))

        # 1. REWARDS
        ax = axes[0, 0]
        ax.plot(episodes, episode_rewards, alpha=0.3,
                color=self.colors['gray'], label='Raw')
        if len(episode_rewards) >= window:
            ma = self._moving_average(episode_rewards, window)
            ax.plot(episodes[window-1:], ma, color=self.colors['primary'],
                    linewidth=2.5, label=f'MA({window})')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_ylabel('Total Reward', fontsize=11)
        ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # 2. ENERGY
        ax = axes[0, 1]
        ax.plot(episodes, episode_energies, alpha=0.3,
                color=self.colors['gray'], label='Raw')
        if len(episode_energies) >= window:
            ma = self._moving_average(episode_energies, window)
            ax.plot(episodes[window-1:], ma, color=self.colors['secondary'],
                    linewidth=2.5, label=f'MA({window})')
        ax.set_ylabel('Energy', fontsize=11)
        ax.set_title('Control Effort (Energy)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. MAX OVERSHOOT
        ax = axes[1, 0]
        ax.plot(episodes, episode_max_overshoots, alpha=0.3,
                color=self.colors['gray'], label='Raw')
        if len(episode_max_overshoots) >= window:
            ma = self._moving_average(episode_max_overshoots, window)
            ax.plot(episodes[window-1:], ma, color=self.colors['danger'],
                    linewidth=2.5, label=f'MA({window})')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Max Overshoot (%)', fontsize=11)
        ax.set_title('Maximum Overshoot', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 4. EXPLORACIÓN / LOSSES
        ax = axes[1, 1]
        if epsilons is not None:
            ax.plot(episodes, epsilons,
                    color=self.colors['success'], linewidth=2.5)
            ax.set_ylabel('Epsilon (ε)', fontsize=11)
            ax.set_title('Exploration Rate', fontsize=12, fontweight='bold')
            ax.set_ylim([-0.05, 1.05])
        elif actor is not None and critic is not None:
            ax.plot(episodes, actor, alpha=0.4,
                    color=self.colors['primary'], label='Actor Raw')
            ax.plot(episodes, critic, alpha=0.4,
                    color=self.colors['secondary'], label='Critic Raw')
            if len(actor) >= window:
                ax.plot(episodes[window-1:],
                        self._moving_average(actor, window),
                        color=self.colors['primary'], linewidth=2.5,
                        label=f'Actor MA({window})')
                ax.plot(episodes[window-1:],
                        self._moving_average(critic, window),
                        color=self.colors['secondary'], linewidth=2.5,
                        label=f'Critic MA({window})')
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title('Actor & Critic Loss', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)

        ax.set_xlabel('Episode', fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 2. PID EVOLUTION — por variable
    # -------------------------------------------------------------------------

    def plot_pid_evolution(
        self,
        kp_history: List[List[float]],
        ki_history: List[List[float]],
        kd_history: List[List[float]],
        var_names: Optional[List[str]] = None
    ):
        """
        Evolución de parámetros PID durante el entrenamiento, una línea
        por variable manipulable.
        """
        n_vars = len(kp_history)

        if var_names is None:
            var_names = [f'Var {i}' for i in range(n_vars)]

        # Colores distintos por variable
        var_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'
        ]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        episodes = np.arange(len(kp_history[0]))

        param_histories = [kp_history, ki_history, kd_history]
        param_labels    = ['Kp', 'Ki', 'Kd']

        for row, (histories, label) in enumerate(zip(param_histories, param_labels)):
            ax = axes[row]
            for i in range(n_vars):
                color = var_colors[i % len(var_colors)]
                ax.plot(episodes, histories[i],
                        color=color, linewidth=2,
                        label=var_names[i])
            ax.set_ylabel(label, fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title('PID Parameters Evolution', fontsize=12,
                             fontweight='bold')

        axes[-1].set_xlabel('Episode', fontsize=11)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 3. VARIABLE EVOLUTION — snapshots de evaluación
    # -------------------------------------------------------------------------

    def plot_variable_evolution(
        self,
        eval_trajectories: List[Dict],
        var_names: Optional[List[str]] = None
    ):
        """
        Muestra cómo evolucionaron las variables manipulables a lo largo
        del entrenamiento, usando los snapshots guardados en cada evaluación.
        """
        if not eval_trajectories:
            print("No hay trayectorias de evaluación guardadas.")
            return

        # Inferir n_vars del primer snapshot
        n_vars = len(eval_trajectories[0]['pv_history'][0])

        if var_names is None:
            var_names = [f'Var {i}' for i in range(n_vars)]

        n_snapshots = len(eval_trajectories)

        # Colormap: azul claro → azul oscuro según progreso del entrenamiento
        cmap = cm.Blues
        colors = [cmap(0.3 + 0.7 * (i / max(n_snapshots - 1, 1)))
                  for i in range(n_snapshots)]

        fig, axes = plt.subplots(n_vars, 1,
                                 figsize=(12, 4 * n_vars),
                                 sharex=False)
        if n_vars == 1:
            axes = [axes]

        for var_idx in range(n_vars):
            ax = axes[var_idx]

            for snap_idx, snap in enumerate(eval_trajectories):
                # pv_history: [[pv_var0, pv_var1], [pv_var0, pv_var1], ...]
                pv_traj = [step[var_idx] for step in snap['pv_history']]
                sp_traj = [step[var_idx] for step in snap['sp_history']]
                steps   = np.arange(len(pv_traj))

                ep_label = f"Ep {snap['episode']}" if snap_idx in [0, n_snapshots-1] \
                           else None

                ax.plot(steps, pv_traj,
                        color=colors[snap_idx],
                        linewidth=1.5,
                        alpha=0.8,
                        label=ep_label)

                # SP solo en el último snapshot para no saturar el gráfico
                if snap_idx == n_snapshots - 1:
                    ax.plot(steps, sp_traj,
                            color=self.colors['danger'],
                            linestyle='--',
                            linewidth=2,
                            label='SP')

                    # Banda ±2%
                    sp_arr = np.array(sp_traj)
                    ax.fill_between(steps,
                                    sp_arr * 0.98, sp_arr * 1.02,
                                    alpha=0.15,
                                    color=self.colors['success'],
                                    label='±2% band')

            ax.set_ylabel(var_names[var_idx], fontsize=11)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_title(f'{var_names[var_idx]} — Evolution During Training',
                         fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Colorbar como referencia de progreso
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(
                                       vmin=eval_trajectories[0]['episode'],
                                       vmax=eval_trajectories[-1]['episode']
                                   ))
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label='Training Episode',
                     fraction=0.02, pad=0.04)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 4. ACTION DISTRIBUTION
    # -------------------------------------------------------------------------

    def plot_action_distribution(
        self,
        action_counts: Dict[int, int],
        action_labels: Optional[List[str]] = None
    ):
        """Distribución de acciones tomadas durante el entrenamiento."""
        if action_labels is None:
            action_labels = [
                'Kp ↑', 'Ki ↑', 'Kd ↑',
                'Kp ↓', 'Ki ↓', 'Kd ↓',
                'Mantener'
            ]

        actions = sorted(action_counts.keys())
        counts  = [action_counts[a] for a in actions]
        labels  = [action_labels[a] if a < len(action_labels)
                   else f'Action {a}' for a in actions]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(actions)), counts,
                      color=self.colors['primary'], alpha=0.8)

        max_idx = counts.index(max(counts))
        bars[max_idx].set_color(self.colors['success'])

        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Action Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        total = sum(counts)
        for bar, count in zip(bars, counts):
            pct = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.01,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # 5. BEST EPISODE
    # -------------------------------------------------------------------------

    def plot_best_episode(
        self,
        pv_trajectory: List[float],
        sp_trajectory: List[float],
        control_trajectory: Optional[List[float]] = None,
        title: str = "Best Episode"
    ):
        """Gráfico del mejor episodio: PV vs SP y señal de control."""
        n_plots = 2 if control_trajectory else 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

        if n_plots == 1:
            axes = [axes]

        steps = np.arange(len(pv_trajectory))

        ax = axes[0]
        ax.plot(steps, pv_trajectory,
                label='Process Value (PV)',
                color=self.colors['primary'], linewidth=2.5)
        ax.plot(steps, sp_trajectory,
                label='Setpoint (SP)',
                color=self.colors['danger'], linestyle='--', linewidth=2)

        sp_mean = np.mean(sp_trajectory)
        ax.fill_between(steps,
                        sp_mean * 0.98, sp_mean * 1.02,
                        alpha=0.15, color=self.colors['success'],
                        label='±2% tolerance')

        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{title} - Tracking Performance',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        if control_trajectory:
            ax = axes[1]
            ax.plot(steps, control_trajectory,
                    color=self.colors['secondary'], linewidth=2)
            ax.set_xlabel('Step', fontsize=11)
            ax.set_ylabel('Control Output', fontsize=11)
            ax.set_title('Control Signal', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            axes[0].set_xlabel('Step', fontsize=11)

        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # UTILIDADES
    # -------------------------------------------------------------------------

    @staticmethod
    def _moving_average(data: List[float], window: int) -> np.ndarray:
        return np.convolve(data, np.ones(window) / window, mode='valid')


# -----------------------------------------------------------------------------
# FUNCIÓN DE RESUMEN
# -----------------------------------------------------------------------------

def print_summary(
    episode_rewards: List[float],
    episode_energies: List[float],
    episode_max_overshoots: List[float],
    best_episode_idx: int
):
    n_episodes = len(episode_rewards)
    last_10 = slice(-10, None)

    print("RESUMEN DE ENTRENAMIENTO")
    print(f"\n{'Métrica':<30} {'Último':<12} {'Promedio (ult 10)':<20} {'Mejor':<12}")
    print("-" * 74)

    print(f"{'Reward':<30} {episode_rewards[-1]:>11.2f} "
          f"{np.mean(episode_rewards[last_10]):>19.2f} "
          f"{max(episode_rewards):>11.2f}")

    print(f"{'Energy':<30} {episode_energies[-1]:>11.2f} "
          f"{np.mean(episode_energies[last_10]):>19.2f} "
          f"{min(episode_energies):>11.2f}")

    print(f"{'Max Overshoot (%)':<30} {episode_max_overshoots[-1]:>11.2f} "
          f"{np.mean(episode_max_overshoots[last_10]):>19.2f} "
          f"{min(episode_max_overshoots):>11.2f}")

    print(f"\nMejor episodio: #{best_episode_idx} "
          f"(Reward: {episode_rewards[best_episode_idx]:.2f})")

    if abs(episode_rewards[0]) > 1e-8:
        mejora = (episode_rewards[-1] - episode_rewards[0]) / abs(episode_rewards[0]) * 100
        print(f"Mejora total: {mejora:.1f}%")

    print(f"Total episodios: {n_episodes}")