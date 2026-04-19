import numpy as np
from typing import List, Dict, Optional
from .PIDComponentes_StabilityCriteria import StabilityCriteria


class RewardCalculator:    
    def __init__(self, 
                 weights: Optional[Dict] = None,
                 ranges: Optional[List] = None, # Lista de tuplas (min, max) para cada variable controlada, usada para normalizar el error
                 dead_band: float = 0.02, # Porcentaje de error relativo al setpoint para considerar que se llegó al objetivo
                 max_time: float = 1800.0, # Tiempo máximo esperado para alcanzar el setpoint (en segundos), usado para normalizar el tiempo de respuesta
                 stability_config: Optional[Dict] = None): # Dict con parámetros para StabilityCriteria (opcional). Si None, usa valores por defecto
        
        # Pesos por defecto
        if weights is None:
            self.weights = {
                'error': 1.0, # Que tan lejo del setpoint
                'tiempo': 0.01, # Que tan rápido responde el sistema
                'overshoot': 0.5, # Que tanto sobrepasa el setpoint
                'energy': 0.1 # Que tanto esfuerzo se necesita
            }
        else:
            self.weights = weights
        
        self.ranges = ranges or [(0.0, 100.0)]
        self.dead_band = dead_band
        self.max_time = max_time

        # Max error posible por variable (para normalizar)
        self.max_errors = [r[1] - r[0] for r in self.ranges]

        # Componente de estabilidad
        sc = stability_config or {}
        self.stability_checker = StabilityCriteria(
            error_increase_tolerance=sc.get('error_increase_tolerance', 1.5),
            max_sign_changes_ratio=sc.get('max_sign_changes_ratio', 0.2),
            max_abrupt_change_ratio=sc.get('max_abrupt_change_ratio', 0.05),
            abrupt_change_threshold=sc.get('abrupt_change_threshold', 0.3)
        )

    def calculate(self,
                  errors: List[float], # Error absoluto por variable [|e1|, |e2|, ...]
                  tiempos_respuesta: List[float], # Tiempo de respuesta por variable [t1, t2, ...]
                  overshoots: List[float], # Overshoot relativo por variable [(pv1 - sp1)/sp1, (pv2 - sp2)/sp2, ...]
                  energy_step: float, # Energía consumida en el step actual (normalizada)
                  pvs: List[float],
                  setpoints: List[float],
                  terminated: bool, #True si el episodio terminó (éxito o fallo)
                  truncated: bool, # True si se alcanzó max_steps
                  trajs_pv: Optional[List[List[float]]] = None, #Trayectorias de PV del ResponseTimeDetector
                  trajs_control: Optional[List[List[float]]] = None) -> float: #Trayectorias de control del ResponseTimeDetector
        
        # Evaluar estabilidad si se proporcionan trayectorias
        stability = None
        if trajs_pv is not None and trajs_control is not None:
            stability = self.stability_checker.check_all(trajs_pv, trajs_control, setpoints)

        if not terminated and not truncated:
            return self._calculate_step_reward(errors, tiempos_respuesta, overshoots,
                                               energy_step, stability)
        else:
            return self._calculate_episode_reward(errors, tiempos_respuesta, overshoots,
                                                  energy_step, pvs, setpoints,
                                                  terminated, stability)

    
    def _calculate_step_reward(self, 
                               errors: List[float],
                               tiempos: List[float],
                               overshoots: List[float],
                               energy: float,
                               stability: Optional[Dict]) -> float:
        
        n_vars = len(errors)

        # Determinar vars_ok para el ponderado (si no hay stability, todas iguales)
        if stability is not None:
            vars_ok = stability['vars_ok']
            ratio = stability['ratio']
        else:
            vars_ok = [False] * n_vars
            ratio = 0.0

        # Reward por variable
        rewards_por_var = []
        for i in range(n_vars):
            max_e = self.max_errors[i] if i < len(self.max_errors) else 1.0
            max_e = max(max_e, 1e-8)

            # Normalizar a [0, 1]
            # Diferencia clave con script anterior: normalizo cada peso por separado para que cada componente tenga un impacto relativo consistente, independientemente de la escala de las variables
            error_norm     = np.clip(errors[i] / max_e, 0.0, 1.0)
            tiempo_norm    = np.clip(tiempos[i] / self.max_time, 0.0, 1.0) if self.max_time > 0 else 0.0
            overshoot_norm = np.clip(overshoots[i] / 100.0, 0.0, 1.0)
            # energy ya viene normalizada del ambiente

            r_i = -(
                self.weights['error']     * error_norm     +
                self.weights['tiempo']    * tiempo_norm    +
                self.weights['overshoot'] * overshoot_norm +
                self.weights['energy']    * energy / max(n_vars, 1)  # distribuir energía
            )

            # Ponderado: variable que NO cumple pesa más (factor 1.5)
            # variable que SÍ cumple pesa menos (factor 0.5) para incentivar al agente a mejorar las que no cumplen
            peso = 0.5 if vars_ok[i] else 1.5
            rewards_por_var.append(r_i * peso)

        reward_base = sum(rewards_por_var) / n_vars

        # Multiplicador global de estabilidad
        multiplicador = self._stability_multiplier(ratio)
        return reward_base * multiplicador
    
    def _calculate_episode_reward(self, 
                                  errors: List[float],
                                  tiempos: List[float],
                                  overshoots: List[float],
                                  energy: float,
                                  pvs: List[float],
                                  setpoints: List[float],
                                  terminated: bool,
                                  stability: Optional[Dict]) -> float:

        step_reward = self._calculate_step_reward(errors, tiempos, overshoots,
                                                  energy, stability)

        success = all(
            abs(pv - sp) / abs(sp) < self.dead_band if sp != 0 else abs(pv) < self.dead_band
            for pv, sp in zip(pvs, setpoints)
        )

        if terminated and success:
            return step_reward * 0.5 + 1.0  # Éxito: reduce penalización + bonus positivo claro
        elif terminated and not success:
            return step_reward * 2.0   # Fallo: duplica penalización
        else:  # truncated
            return step_reward * 1.2   # Truncado: penalización leve extra
        
    def _stability_multiplier(self, ratio: float) -> float:
        """
        Convierte el ratio de estabilidad en un multiplicador para el reward.
        Interpolación lineal entre [0.5, 1.5].
        Sistema estable (ratio=1) → multiplica x1.5 (menos penalización).
        Sistema inestable (ratio=0) → multiplica x0.5 (más penalización).
        Nota: cuando stability=None (no se pasan trayectorias), ratio=0.0
        y el multiplicador es 0.5 — señal neutra conservadora.
        """
        return 0.5 + ratio  # ratio=0 → 0.5, ratio=0.5 → 1.0, ratio=1.0 → 1.5
    
    def update_weights(self, new_weights: dict):
        """Actualizar pesos de los componentes."""
        self.weights.update(new_weights)
    
    def get_weights(self) -> dict:
        """Obtener pesos actuales."""
        return self.weights.copy()