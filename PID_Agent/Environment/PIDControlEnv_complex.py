import gymnasium as gym
import numpy as np
import random
from abc import ABC
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List, Union

from Aux.PIDComponents_PID import PIDController 
from Aux.PIDComponents_time import ResponseTimeDetector
from Aux.PIDComponentes_translate import ApplyAction
from Aux.PIDComponents_Reward import RewardCalculator
from .Simulation_Env.SimulationEnv import SimulationPIDEnv

class PIDControlEnv_Complex(gym.Env, ABC):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        #CONFIGURACION DEL AMBIENTE

        ##Arquitectura
        self.architecture = config.get('architecture', 'jerarquica')  # 'simple' o 'jerairquica'

        ##Tipo de entorno
        env_type = config.get('env_type', 'simulation')

        ##Variables del proceso
        ###Control
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.manipulable_ranges = config.get('manipulable_ranges', [(0.0, 100.0), (0.0, 100.0)]) #Rangos de las variables manipulables 
        self.manipulable_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges
        ]
       
        ###Target
        self.n_target_vars = config.get('n_target_vars', 1)
        self.target_ranges = config.get('target_ranges', [(0.0, 1.0)])
        self.target_setpoints = config.get('target_setpoints', [0.2])
        self.target_working_ranges = config.get('target_working_ranges', [(0.0, 1.0)])
        self.target_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.target_working_ranges
        ]

        # DESPUÉS inyectar e instanciar proceso
        env_type_config = config.get('env_type_config', {}).copy()
        env_type_config['manipulable_ranges'] = self.manipulable_ranges
        if env_type == 'simulation':
            self.proceso = SimulationPIDEnv(env_type_config)

        #CONFIGURACION DEL AGENTE

        ##Configuracion de los Agentes según arquitectura
        self.agente_orch = config.get('agent_orchestrator_config', {})
        self.agente_ctrl = None
        self.action_type_ctrl = None

        ## Estado interno
        ###errores
        self.error_integral_target = [0.0] * self.n_target_vars
        self.error_derivative_target = [0.0] * self.n_target_vars
        self.error_target = [0.0] * self.n_target_vars
        self.error_prevs_target = [0.0] * self.n_target_vars

        ### tiempo de respuesta y dt (Detectores de tiempo solo para dinamicas controlables (PID))
        self.dt_sim = config.get('dt_usuario', 1.0)
        self.reward_dead_band = config.get('reward_dead_band', 0.02)  

        self.max_time_detector = config.get('max_time_detector', 1800)

        self.response_time_detectors = ResponseTimeDetector(
                proceso=self.proceso,
                env_type=env_type,
                dt=self.dt_sim,
                tolerance=self.reward_dead_band 
            ) 

        ### Dinamica del ambiente (PIDs para variables manipulables)
        self.pid_controllers = [
            PIDController(kp=1.0, ki=0.1, kd=0.01, dt=self.dt_sim,
                        output_limits=(rango[0], rango[1]))
            for rango in self.manipulable_ranges
        ]

        #### Valor dummy iniciales (se calculan en el primer step)
        self.tiempo_respuesta = [0.0] * self.n_manipulable_vars


        #ESPACIO DE OBSERVACIONES
        self.obs_structure = ['pv', 'sp', 'error', 'error_integral', 'error_derivative']
        self.obs_size = len(self.obs_structure) 

        n_obs_ctrl = self.obs_size * self.n_manipulable_vars
        # 5 por cada variable objetivo + 1 SP normalizado por cada variable manipulable
        n_obs_orch = self.obs_size * self.n_target_vars + self.n_manipulable_vars

        self.observation_space = spaces.Dict({
            'orch': spaces.Box(
                low=np.full(n_obs_orch, -np.inf, dtype=np.float32),
                high=np.full(n_obs_orch, np.inf, dtype=np.float32),
                dtype=np.float32
            ),
            'ctrl': spaces.Box(
                low=np.full(n_obs_ctrl, -np.inf, dtype=np.float32),
                high=np.full(n_obs_ctrl, np.inf, dtype=np.float32),
                dtype=np.float32
            )
        })

        # ESPACIO DE ACCIONES
        if self.agente_orch.get('agent_type', 'continuous') == 'continuous':
            action_space_orch = spaces.Box(
                low=np.array([-r[1] for r in self.manipulable_ranges], dtype=np.float32),
                high=np.array([r[1] for r in self.manipulable_ranges], dtype=np.float32),
                dtype=np.float32
            )
        elif self.agente_orch.get('agent_type', 'discrete') == 'discrete':
            action_space_orch = spaces.MultiDiscrete([3] * self.n_manipulable_vars)

        # CTRL (placeholder)
        action_space_ctrl = spaces.MultiDiscrete([7] * self.n_manipulable_vars)

        # Combinar
        self.action_space = spaces.Dict({
            'orch': action_space_orch,
            'ctrl': action_space_ctrl
        })

        ## Mapeo de acciones discretas
        if self.agente_orch.get('agent_type', 'continuous') == 'discrete':
            self.ACTION_MAP_ORCH = {
                0: +1,  # Aumentar SP
                1: -1,  # Disminuir SP
                2: 0    # Mantener
            }       

  

        ## Componente para traducir acciones a parámetros de control
        self.apply_action_orch = ApplyAction(
            delta_percent_orch=config.get('delta_percent_orch', 0.05),
            manipulable_ranges=self.manipulable_ranges
        )

        self.apply_action_ctrl = ApplyAction(
            delta_percent_ctrl=config.get('delta_percent_ctrl', 0.2),
            pid_limits=config.get('pid_limits', None),
            manipulable_ranges=self.manipulable_ranges
        )

        # ENTRENAMIENTO
        self.max_steps = config.get('max_steps', 20)
        self.current_step = 0

        # Frecuencia del ORCH: actúa cada orch_freq steps del CTRL
        self.orch_freq = config.get('orch_freq', 1)
        self._accumulated_reward = 0.0
        self._orch_step_count = 0

        ## Recompensa
        self.reward_calculator = RewardCalculator(
            weights=config.get('reward_weights', None),
            ranges=self.target_ranges,
            dead_band=config.get('reward_dead_band', 0.02),
            max_time=config.get('max_time_detector', 1800.0)
        )
            
    def _get_observation(self):            
        obs_orch = []
        for j in range(self.n_target_vars):
            max_e = self.target_working_ranges[j][1] - self.target_working_ranges[j][0]
            max_e = max(max_e, 1e-8)
            obs_orch.extend([
                np.clip(self.target_pvs[j] / max_e, 0.0, 1.0),
                self.target_setpoints[j],
                np.clip(self.error_target[j] / max_e, -1.0, 1.0),
                np.clip(self.error_integral_target[j] / max_e, -1.0, 1.0),
                np.clip(self.error_derivative_target[j] / max_e, -1.0, 1.0)
            ])
        # SP actuales de las variables manipulables (normalizados a [0,1])
        # Le da al orch contexto de qué está pidiendo actualmente
        current_sps = self.current_SPs_manipulable if hasattr(self, 'current_SPs_manipulable') else self.manipulable_pvs
        for i in range(self.n_manipulable_vars):
            r = self.manipulable_ranges[i]
            sp_norm = np.clip((current_sps[i] - r[0]) / (r[1] - r[0] + 1e-8), 0.0, 1.0)
            obs_orch.append(float(sp_norm))
        
        obs_ctrl = []
        for i in range(self.n_manipulable_vars):
            # PV actual, SP deseado (definido por ORCH), errores
            sp_desired = self.new_SP[i] if hasattr(self, 'new_SP') else self.manipulable_pvs[i]
            error = sp_desired - self.manipulable_pvs[i]
            
            obs_ctrl.extend([
                self.manipulable_pvs[i],
                sp_desired,
                error,
                0.0,  # error_integral (simplificado)
                0.0   # error_derivative (simplificado)
            ])
        
        return {
            'orch': np.array(obs_orch, dtype=np.float32),
            'ctrl': np.array(obs_ctrl, dtype=np.float32)
        }

    def _get_info(self):
        return {
            'target_pvs': self.target_pvs.copy(),
            'manipulable_pvs': self.manipulable_pvs.copy(),
            'energy': self.energy_accumulated,
            'new_SP': self.new_SP.copy() if hasattr(self, 'new_SP') else []
        }

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # VARIABES DEL ENTORNO A RESETEAR
        if hasattr(self.proceso, 'reset'):
            pvs_iniciales = self.proceso.reset()
            self.manipulable_pvs = list(pvs_iniciales) if pvs_iniciales else [
                random.uniform(rango[0], rango[1]) for rango in self.manipulable_ranges
            ]
        else:
            self.manipulable_pvs = [
                random.uniform(rango[0], rango[1]) for rango in self.manipulable_ranges
            ]

        self.current_SPs_manipulable = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges
        ]    

        # Inicializar target_pvs con el valor real del proceso si está disponible
        # Evita que el primer estado del orch tenga un Cb ficticio (random)
        if hasattr(self.proceso, 'external_process') and self.proceso.external_process:
            state = self.proceso.external_process.get_state()
            self.target_pvs = [state[0]]  # Cb real del CSTR
        else:
            self.target_pvs = [
                random.uniform(rango[0], rango[1])
                for rango in self.target_working_ranges
            ]

        self.target_setpoints = [
            random.uniform(rango[0], rango[1])
            for rango in self.target_ranges
        ]

        self.new_SP = self.manipulable_pvs.copy()    

        # ERRORES
        self.error_integral_target = [0.0] * self.n_target_vars
        self.error_derivative_target = [0.0] * self.n_target_vars
        self.error_target = [0.0] * self.n_target_vars
        self.error_prevs_target = [0.0] * self.n_target_vars

        #TIEMPO
        self.tiempo_respuesta = [0.0] * self.n_manipulable_vars

        #VARIABLES DE INFO
        self.trajectory_manipulable = [[] for _ in range(self.n_manipulable_vars)]
        self.energy_accumulated = 0.0
        self.overshoot_manipulable = [0.0] * self.n_manipulable_vars
        self.accumulated_error_manipulable = [0.0] * self.n_manipulable_vars

        # VARIABLES DE ENTRENAMIENTO
        self.current_step = 0
        self._accumulated_reward = 0.0
        self._orch_step_count = 0

        # DINAMICA
        for pid in self.pid_controllers:
            pid.reset()

        # OBSERVACION E INFO
        observation = self._get_observation()
        info = self._get_info() 

        return observation, info

    def step(self, action):        
        # 1. EXTRAER ACCIÓN DE ORCH
        if isinstance(action, dict):
            action_orch = action.get('orch', None)
        else:
            action_orch = action

        # 2. DECIDIR SI EL ORCH ACTÚA ESTE STEP
        orch_acts_this_step = (self._orch_step_count % self.orch_freq == 0)

        if orch_acts_this_step and action_orch is not None:
            self.action_type_orch = self.agente_orch.get('agent_type', 'continuous')
            self.new_SP = self.apply_action_orch.translate(
                action=action_orch,
                agent_type='orch',
                action_type=self.action_type_orch,
                current_values=self.current_SPs_manipulable
            )
            self.current_SPs_manipulable = self.new_SP.copy()
            # Resetear integral al cambiar SP — evita arrastre del error anterior
            self.error_integral_target = [0.0] * self.n_target_vars
        
        # 3. CTRL DECIDE PARÁMETROS PID PARA ALCANZAR ESOS SP
        obs_ctrl = self._get_observation()['ctrl']
        action_ctrl = self.agente_ctrl.select_action(obs_ctrl, training=False)
        
        # 4. TRADUCIR ACCIÓN CTRL A PARÁMETROS PID
        pid_params = self.apply_action_ctrl.translate(
            action=action_ctrl,
            agent_type='ctrl',
            action_type=self.action_type_ctrl,
            current_values=[(c.kp, c.ki, c.kd) for c in self.pid_controllers]
        )
        
        # 5. ACTUALIZAR PARÁMETROS PID
        for i, (kp, ki, kd) in enumerate(pid_params):
            self.pid_controllers[i].kp = kp
            self.pid_controllers[i].ki = ki
            self.pid_controllers[i].kd = kd
        
        # 6. SIMULAR VARIABLES MANIPULABLES (PIDs → nuevos SP)
        energy_step = 0.0
        
        resultado = self.response_time_detectors.estimate(
            pvs_inicial=self.manipulable_pvs,
            sps=self.new_SP,
            pid_controllers=self.pid_controllers,
            max_time=self.max_time_detector,
            reset_pid=False
        )

            
        for i in range(self.n_manipulable_vars):
            self.manipulable_pvs[i]   = resultado['pvs_final'][i]
            self.tiempo_respuesta[i]  = resultado['tiempos'][i]
            self.trajectory_manipulable[i] = resultado['trayectorias_pv'][i]

        # Acumular energía
        for i in range(self.n_manipulable_vars):
            traj_u = resultado['trayectorias_control'][i]
            if traj_u:
                n_pasos = len(traj_u)
                energia_raw = sum(abs(u) for u in traj_u) * self.dt_sim
                energy_step += energia_raw / (n_pasos * max(self.manipulable_ranges[i][1], 1.0))
        
        # 7. ACTUALIZAR VARIABLES OBJETIVO (dinámica del proceso)
        if hasattr(self.proceso, 'external_process') and self.proceso.external_process:
            # Si hay simulador externo (CSTR)
            state = self.proceso.external_process.get_state()
            self.target_pvs = [state[0]]  # Cb (primera variable)
        else:
            # Placeholder
            self.target_pvs = [0.0] * self.n_target_vars
        
        # 8. ACTUALIZAR ERRORES (solo de variables objetivo)
        self._update_errors()
        
        # 9. CALCULAR REWARD (basado en variables objetivo)
        errors = [abs(pv - sp) for pv, sp in zip(self.target_pvs, self.target_setpoints)]
        terminated = self._check_terminated()
        truncated = self._check_truncated()
        
        reward_step = self.reward_calculator.calculate(
            errors=errors,
            tiempos_respuesta=[0.0] * self.n_target_vars,
            overshoots=[0.0] * self.n_target_vars,
            energy_step=energy_step,
            pvs=self.target_pvs,
            setpoints=self.target_setpoints,
            terminated=terminated,
            truncated=truncated
        )

        # Acumular reward entre decisiones del ORCH
        self._accumulated_reward += reward_step

        # Entregar reward acumulado cuando el ORCH actúa (o al final del episodio)
        # Dividir por orch_freq para mantener la escala consistente
        if orch_acts_this_step or terminated or truncated:
            if terminated:
                reward = self._accumulated_reward  # fallo: reward completo sin dividir
            else:
                reward = self._accumulated_reward / self.orch_freq
            self._accumulated_reward = 0.0
        else:
            reward = 0.0
        
        # 10. OBTENER OBSERVACIÓN E INFO
        observation = self._get_observation()
        info = self._get_info()
        info['orch_acted'] = orch_acts_this_step
        info['reward_step'] = reward_step

        # 11. INCREMENTAR CONTADORES
        self.current_step += 1
        self._orch_step_count += 1

        return observation, reward, terminated, truncated, info

    def _update_errors(self):

        self.dt = self.dt_sim

        # Actualizar errores para variables objetivo
        for i in range(self.n_target_vars):
            error = self.target_setpoints[i] - self.target_pvs[i]
            self.error_target[i] = error
            self.error_integral_target[i] += error * self.dt
            # Clip para evitar windup — mantiene el integral en el rango físico
            max_e = self.target_working_ranges[i][1] - self.target_working_ranges[i][0]
            self.error_integral_target[i] = np.clip(
                self.error_integral_target[i], -max_e, max_e
            )
            self.error_derivative_target[i] = (error - self.error_prevs_target[i]) / self.dt if self.dt > 0 else 0.0
            self.error_prevs_target[i] = error

        return {
            'error_target': self.error_target,
            'error_integral_target': self.error_integral_target,
            'error_derivative_target': self.error_derivative_target,
            'error_prevs_target': self.error_prevs_target
        }
    
    def _check_truncated(self) -> bool:
        # Episodio se trunca si alcanza max_steps
        return self.current_step >= self.max_steps
    
    def _check_terminated(self) -> bool:
        failure = any(
            pv < rango[0] or pv > rango[1]
            for pv, rango in zip(self.target_pvs, self.target_ranges)
        )
        return failure