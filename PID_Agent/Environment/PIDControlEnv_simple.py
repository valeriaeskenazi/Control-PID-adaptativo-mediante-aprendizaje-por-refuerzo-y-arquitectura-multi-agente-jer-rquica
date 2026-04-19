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

class PIDControlEnv_Simple(gym.Env, ABC):

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        #CONFIGURACION DEL AMBIENTE

        ##Arquitectura
        self.architecture = config.get('architecture', 'Simple')  # 'Simple' o 'Jerarquica'

       ##Tipo de entorno
        env_type = config.get('env_type', 'simulation')

        ##Variables del proceso — PRIMERO definir ranges
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.manipulable_ranges = config.get('manipulable_ranges', [(0.0, 10.0), (0.0, 100.0)])
        self.manipulable_pvs = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges
        ]
        self.manipulable_setpoints = config.get('manipulable_setpoints')
        if self.manipulable_setpoints is None:
            self.manipulable_setpoints = [
                random.uniform(rango[0], rango[1])
                for rango in self.manipulable_ranges
            ]

        # DESPUÉS inyectar e instanciar proceso
        env_type_config = config.get('env_type_config', {}).copy()
        env_type_config['manipulable_ranges'] = self.manipulable_ranges
        if env_type == 'simulation':
            self.proceso = SimulationPIDEnv(env_type_config)

        #CONFIGURACION DEL AGENTE

        ##Configuracion de los Agentes según arquitectura
        if self.architecture == 'simple':
            self.agente_orch = False
            self.agente_ctrl = config.get('agent_controller_config', {})

        ## Estado interno
        ###errores
        self.error_integral_manipulable = [0.0] * self.n_manipulable_vars
        self.error_derivative_manipulable = [0.0] * self.n_manipulable_vars
        self.error_manipulable = [0.0] * self.n_manipulable_vars
        self.error_prevs_manipulable = [0.0] * self.n_manipulable_vars

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

        #### Valor dummy iniciales (se calculan en el primer step)
        self.tiempo_respuesta = [0.0] * self.n_manipulable_vars

        ### Dinamica del ambiente
        self.pid_controllers = [
            PIDController(kp=1.0, ki=0.1, kd=0.01, dt=self.dt_sim,
                        output_limits=(rango[0], rango[1]))  # agrego límites
            for rango in self.manipulable_ranges
        ] 


        #ESPACIO DE OBSERVACIONES

        self.obs_structure = ['pv', # Dónde estoy
                              'sp', # Dónde quiero estar
                              'error', # Cuánto me falta?
                              'error_integral', # Hay offset acumulado? (offset es una diferencia constante entre pv y sp que no permite llegar a sp)
                              'error_derivative' # Voy muy rápido/lento?
                              ]
        self.obs_size = len(self.obs_structure) 
        n_obs_total = self.obs_size * self.n_manipulable_vars
        
        self.observation_space = spaces.Box(
            low=np.full(n_obs_total, -np.inf, dtype=np.float32),
            high=np.full(n_obs_total, np.inf, dtype=np.float32),
            dtype=np.float32
        )
        
        # ESPACIO DE ACCIONES

        # El espacio de acciones es continuo, ya que da numeros, pero se puede manejar tambien como discreto si se usan indices para seleccionar acciones predefinidas
        if self.agente_ctrl.get('agent_type', 'continuous') == 'continuous':
            self.action_space = spaces.Box(
                low=np.tile(np.array([-100, -10, -1]), self.n_manipulable_vars).astype(np.float32),
                high=np.tile(np.array([100, 10, 1]), self.n_manipulable_vars).astype(np.float32),
                dtype=np.float32
            )
        elif self.agente_ctrl.get('agent_type', 'discrete') == 'discrete':
            self.action_space = spaces.MultiDiscrete([7] * self.n_manipulable_vars)


        ## Mapeo de acciones discretas
        if self.agente_ctrl.get('agent_type', 'continuous') == 'discrete':
            self.ACTION_MAP_CTRL = {
                0: ('Kp', +1),  # Kp ↑
                1: ('Ki', +1),  # Ki ↑
                2: ('Kd', +1),  # Kd ↑
                3: ('Kp', -1),  # Kp ↓
                4: ('Ki', -1),  # Ki ↓
                5: ('Kd', -1),  # Kd ↓
                6: ('mantener', 0)
            }

        ## Componente para traducir acciones a parámetros de control
        self.apply_action = ApplyAction(
            delta_percent_ctrl=config.get('delta_percent_ctrl', 0.2),
            pid_limits=config.get('pid_limits', None),
            manipulable_ranges=self.manipulable_ranges
        )

        # ENTRENAMIENTO
        self.max_steps = config.get('max_steps', 20)
        self.current_step = 0

        ## Recompensa
        self.reward_calculator = RewardCalculator(
            weights=config.get('reward_weights', None),
            ranges=self.manipulable_ranges,
            dead_band=config.get('reward_dead_band', 0.02),
            max_time=config.get('max_time_detector', 1800.0)
        )
            
    def _get_observation(self):
        obs = []
        for i in range(self.n_manipulable_vars):
            obs.extend([
                self.manipulable_pvs[i],
                self.manipulable_setpoints[i],
                self.error_manipulable[i],
                self.error_integral_manipulable[i],
                self.error_derivative_manipulable[i]
            ])
        return np.array(obs, dtype=np.float32)
        
    def _get_info(self):
        
        info = {
            # Trayectorias completas durante el step
            'trajectory_manipulable': self.trajectory_manipulable,  # Lista de listas [[pv1_t0, pv1_t1, ...], [pv2_t0, pv2_t1, ...]]
            
            # Energía acumulada (esfuerzo de control)
            'energy': self.energy_accumulated,  # Suma de |control_output| * dt
            
            # Overshoot (máximo pico sobre SP)
            'overshoot_manipulable': self.overshoot_manipulable,  # Lista [overshoot_var1, overshoot_var2, ...]
            
            # Error acumulado absoluto
            'accumulated_error_manipulable': self.accumulated_error_manipulable,           
            
        }
        
        return info

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        #SIMULADOR Y VARIABES DEL ENTORNO A RESETEAR
        if hasattr(self.proceso, 'reset'):
            pvs_iniciales = self.proceso.reset()
            self.manipulable_pvs = list(pvs_iniciales)[:self.n_manipulable_vars] if pvs_iniciales else [
                random.uniform(rango[0], rango[1]) for rango in self.manipulable_ranges
            ]
        else:
            self.manipulable_pvs = [
                random.uniform(rango[0], rango[1]) for rango in self.manipulable_ranges
            ]
        self.manipulable_setpoints = [
            random.uniform(rango[0], rango[1])
            for rango in self.manipulable_ranges
        ]

        # DINAMICA DEL AMBIENTE
        for pid in self.pid_controllers:
            pid.reset()

        # ERRORES
        self.error_integral_manipulable = [0.0] * self.n_manipulable_vars
        self.error_derivative_manipulable = [0.0] * self.n_manipulable_vars
        self.error_manipulable = [0.0] * self.n_manipulable_vars
        self.error_prevs_manipulable = [0.0] * self.n_manipulable_vars

        #TIEMPO
        self.tiempo_respuesta = [0.0] * self.n_manipulable_vars

        #VARIABLES DE INFO
        self.trajectory_manipulable = [[] for _ in range(self.n_manipulable_vars)]
        self.energy_accumulated = 0.0
        self.overshoot_manipulable = [0.0] * self.n_manipulable_vars
        self.accumulated_error_manipulable = [0.0] * self.n_manipulable_vars

        # VARIABLES DE ENTRENAMIENTO
        self.current_step = 0

        # OBSERVACION E INFO
        observation = self._get_observation()
        info = self._get_info() 


        
        return observation, info

    def step(self, action):

        # 1. TRADUCIR ACCION A PARAMETROS DE CONTROL
        self.action_type = self.agente_ctrl.get('agent_type', 'continuous')
        pid_params = self.apply_action.translate(
            action=action,
            agent_type='ctrl',
            action_type=self.action_type,
            current_values=[(c.kp, c.ki, c.kd) for c in self.pid_controllers]
        )
        # pid_params = [(kp1, ki1, kd1), (kp2, ki2, kd2), ...]
        
        # Actualizar parámetros de cada controlador
        for i, (kp, ki, kd) in enumerate(pid_params):
            self.pid_controllers[i].kp = kp
            self.pid_controllers[i].ki = ki
            self.pid_controllers[i].kd = kd

        
        # 2. SIMULAR CADA VARIABLE (ResponseTimeDetector hace toda la simulación)
        energy_step = 0.0 

        resultado_multi = self.response_time_detectors.estimate(
            pvs_inicial=list(self.manipulable_pvs),
            sps=list(self.manipulable_setpoints),
            pid_controllers=self.pid_controllers,
            max_time=self.max_time_detector
        )
            
        # Extraer resultados
        for i in range(self.n_manipulable_vars):
            self.manipulable_pvs[i]   = resultado_multi['pvs_final'][i]
            self.tiempo_respuesta[i]  = resultado_multi['tiempos'][i]
            self.trajectory_manipulable[i] = resultado_multi['trayectorias_pv'][i]
            
            # Métricas por variable
            resultado_i = {
                'trayectoria_pv':      resultado_multi['trayectorias_pv'][i],
                'trayectoria_control': resultado_multi['trayectorias_control'][i],
                'pv_final':            resultado_multi['pvs_final'][i],
                'converged':           resultado_multi['converged'][i]
            }
            self._calculate_variable_metrics(i, resultado_i)

        # Energía
        energy_step = 0.0
        for i in range(self.n_manipulable_vars):
            traj_u = resultado_multi['trayectorias_control'][i]
            if traj_u:
                n_pasos = len(traj_u)
                energia_raw = sum(abs(u) for u in traj_u) * self.dt_sim
                energy_step += energia_raw / (n_pasos * max(self.manipulable_ranges[i][1], 1.0))

        # 3. ACTUALIZAR ERRORES
        self._update_errors()

        # 4. DETERMINAR TERMINACIÓN
        terminated = self._check_terminated()
        truncated = self._check_truncated()

        # 5. CALCULAR RECOMPENSA (usa self.tiempo_respuesta)
        reward = self._calculate_reward(energy_step, terminated, truncated,
                                trajs_pv=resultado_multi['trayectorias_pv'],
                                trajs_control=resultado_multi['trayectorias_control'])
        
        # 6. OBTENER OBSERVACIÓN E INFO
        observation = self._get_observation()
        info = self._get_info()
        
        # 7. INCREMENTAR STEP
        self.current_step += 1

        return observation, reward, terminated, truncated, info
    
    def _update_errors(self):

        self.dt = self.dt_sim

        # Actualizar errores para variables manipulables
        for i in range(self.n_manipulable_vars):
            error = self.manipulable_setpoints[i] - self.manipulable_pvs[i]
            self.error_manipulable[i] = error
            self.error_integral_manipulable[i] += error * self.dt
            self.error_derivative_manipulable[i] = (error - self.error_prevs_manipulable[i]) / self.dt if self.dt > 0 else 0.0
            self.error_prevs_manipulable[i] = error

        return {
            'error_manipulable': self.error_manipulable,
            'error_integral_manipulable': self.error_integral_manipulable,
            'error_derivative_manipulable': self.error_derivative_manipulable,
            'error_prevs_manipulable': self.error_prevs_manipulable
        }
    
    def _check_truncated(self) -> bool:
        # Episodio se trunca si alcanza max_steps
        return self.current_step >= self.max_steps
    
    def _check_terminated(self) -> bool:
        failure = any(
            pv < rango[0] or pv > rango[1]
            for pv, rango in zip(self.manipulable_pvs, self.manipulable_ranges)
        )
        return failure

    def _calculate_variable_metrics(self, var_idx: int, resultado: dict):
        trayectoria = resultado['trayectoria_pv']
        sp = self.manipulable_setpoints[var_idx]
        
        # 1. Overshoot (máximo pico sobre SP, en porcentaje)
        max_pv = max(trayectoria)
        if max_pv > sp:
            self.overshoot_manipulable[var_idx] = (max_pv - sp) / sp * 100
        else:
            self.overshoot_manipulable[var_idx] = 0.0
        
        # 2. Error acumulado (integral del error absoluto)
        accumulated_error = sum(abs(pv - sp) for pv in trayectoria) * self.dt_sim
        self.accumulated_error_manipulable[var_idx] = accumulated_error
        
        # 3. Energía (esfuerzo de control)
        if 'trayectoria_control' in resultado:
            energy = sum(abs(u) for u in resultado['trayectoria_control']) * self.dt_sim
            self.energy_accumulated += energy    

    def _calculate_reward(self, energy_step, terminated, truncated,
                      trajs_pv=None, trajs_control=None) -> float:
        errors = [abs(pv - sp) for pv, sp in zip(self.manipulable_pvs, self.manipulable_setpoints)]
        return self.reward_calculator.calculate(
            errors=errors,
            tiempos_respuesta=self.tiempo_respuesta,
            overshoots=self.overshoot_manipulable,
            energy_step=energy_step,
            pvs=self.manipulable_pvs,
            setpoints=self.manipulable_setpoints,
            terminated=terminated,
            truncated=truncated,
            trajs_pv=trajs_pv,
            trajs_control=trajs_control
        )  