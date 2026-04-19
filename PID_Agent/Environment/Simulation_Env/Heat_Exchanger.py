import numpy as np
from typing import Tuple, Optional, Dict, List, Union


class HeatExchangerSimulator:
    """
    Simulador de intercambiador de calor de primer orden con tiempo muerto.
    
    Sistema SISO:
        - Variable manipulable (PV): T_out — temperatura de salida (°C)
        - Variable de control (u):   M     — señal de control (mA)
    
    Parámetros identificados experimentalmente (Bequette correlation):
        K  = 11.2455  °C/mA   ganancia estática
        τ  = 0.405    min     constante de tiempo
        θ  = 0.125    min     tiempo muerto (≈ 1 paso con dt=0.125)
    
    ODE equivalente (primer orden):
        τ · dT_out/dt = -T_out + K · u(t - θ)
    
    El tiempo muerto se implementa como buffer circular de 1 paso.
    Compatible con la interfaz de CSTRSimulator para conectarse con
    SimulationPIDEnv y el framework de entrenamiento.
    """

    def __init__(
        self,
        dt: float = 0.125,
        control_limits: Tuple[Tuple[float, float]] = ((4.0, 20.0),)
    ):
        self.dt = dt

        # Parámetros del proceso (identificados experimentalmente)
        self.K   = 11.2455   # Ganancia estática (°C/mA)
        self.tau = 0.405     # Constante de tiempo (min)
        self.theta = 0.125   # Tiempo muerto (min)

        # Límites de la variable manipulable (señal de control en mA)
        # En variables desviación: u_abs ∈ [4, 20] mA, u_ss = 12 mA
        # → u_dev ∈ [-8, +8] mA
        self.u_abs_min = 4.0    # mA absoluto
        self.u_abs_max = 20.0   # mA absoluto
        self.u_ss_abs  = 12.0   # mA absoluto (punto de operación nominal)
        
        self.u_min = control_limits[0][0]  # En variables desviación
        self.u_max = control_limits[0][1]  # En variables desviación

        # Estado estacionario
        # Con u_ss = 12 mA → T_out_ss = K * u_ss = 11.2455 * 12 ≈ 134.9 °C
        # (variables desviación — el estado estacionario es 0 en variables desviación)
        self.T_out_ss = 0.0   # variables desviación
        self.u_ss     = 0.0   # variables desviación

        # Estado actual
        self.T_out = self.T_out_ss
        self.state = np.array([self.T_out])  # Compatibilidad con SimulationPIDEnv

        # Buffer para tiempo muerto (delay de 1 paso)
        self._delay_steps = max(1, round(self.theta / self.dt))
        self._u_buffer = [self.u_ss] * (self._delay_steps + 1)

        # Control actual
        self.u_current = self.u_ss

        # Perturbación persistente en T_in (variables desviación)
        self.d_T_in = 0.0  # Perturbación actual en T_in
        # Parámetros de Gd (del ensayo 2 del informe 2013)
        self.K_d = 1.0          # Ganancia de perturbación (°C/°C)
        self.tau_d = 0.915      # Constante de tiempo de perturbación (min)
        self.theta_d = 0.315    # Tiempo muerto de perturbación (min)
        self._d_delay_steps = max(1, round(self.theta_d / self.dt))
        self._d_buffer = [0.0] * (self._d_delay_steps + 1)
        self.T_out_d = 0.0      # Estado del filtro de perturbación

    # ------------------------------------------------------------------
    # Interfaz requerida por SimulationPIDEnv
    # ------------------------------------------------------------------

    def get_n_variables(self) -> int:
        """1 variable manipulable: T_out."""
        return 1

    def get_initial_pvs(self) -> List[float]:
        """Retorna PV inicial — temperatura de salida en estado estacionario."""
        return [self.T_out_ss]

    def simulate_step_multi(self, control_outputs: list, dt: float) -> list:
        """
        Avanza un paso de simulación.

        Args:
            control_outputs: [u] — señal de control en variables desviación (mA)
            dt: paso de tiempo (min)

        Returns:
            [T_out] — temperatura de salida en variables desviación (°C)
        """
        u = float(control_outputs[0])
        u = np.clip(u, self.u_min, self.u_max)
        self.u_current = u

        # Actualizar buffer de tiempo muerto
        self._u_buffer.append(u)
        if len(self._u_buffer) > self._delay_steps + 1:
            self._u_buffer.pop(0)

        # Acción retardada
        u_delayed = self._u_buffer[0]

        # Perturbación con dinámica Gd: τd·dT_d/dt = -T_d + Kd·d(t-θd)
        self._d_buffer.append(self.d_T_in)
        if len(self._d_buffer) > self._d_delay_steps + 1:
            self._d_buffer.pop(0)
        d_delayed = self._d_buffer[0]
        
        dTd_dt = (-self.T_out_d + self.K_d * d_delayed) / self.tau_d
        self.T_out_d += dTd_dt * dt

        # Integrar ODE del proceso: τ·dT/dt = -T + K·u_delayed
        # T_out total = respuesta al control + respuesta a perturbación
        dTdt = (-self.T_out + self.K * u_delayed) / self.tau
        self.T_out += dTdt * dt
        
        # Temperatura medida = proceso + perturbación
        T_total = self.T_out + self.T_out_d

        # Clip físico (±50°C en variables desviación es razonable)
        T_total = np.clip(T_total, -50.0, 50.0)
        self.state = np.array([T_total])

        # Ruido de sensor
        T_meas = T_total + np.random.uniform(-0.05, 0.05)

        return [float(T_meas)]

    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        randomize: bool = False
    ) -> List[float]:
        """
        Resetea el simulador al estado estacionario (variables desviación = 0).

        Args:
            initial_state: [T_out_0] opcional
            randomize: si True, agrega perturbación aleatoria pequeña

        Returns:
            [T_out_inicial]
        """
        if initial_state is not None:
            self.T_out = float(initial_state[0])
        elif randomize:
            self.T_out = self.T_out_ss + np.random.uniform(-2.0, 2.0)
        else:
            self.T_out = self.T_out_ss

        # Resetear buffer de delay
        self._u_buffer = [self.u_ss] * (self._delay_steps + 1)
        self.u_current = self.u_ss
        
        # Resetear perturbación
        self.d_T_in = 0.0
        self.T_out_d = 0.0
        self._d_buffer = [0.0] * (self._d_delay_steps + 1)
        
        self.state = np.array([self.T_out])

        return self.get_initial_pvs()

    def get_state(self) -> List[float]:
        """
        Estado actual del simulador.

        Returns:
            [T_out_total] en variables desviación (proceso + perturbación)
        """
        return [float(self.T_out + self.T_out_d)]

    def get_measurements(self) -> Dict[str, float]:
        """Mediciones con ruido."""
        T_total = self.T_out + self.T_out_d
        return {
            'T_out': T_total + np.random.uniform(-0.05, 0.05),
            'u':     self.u_current
        }

    def set_disturbance(self, delta_T_in: Optional[float] = None):
        """
        Introduce una perturbación persistente en la temperatura de entrada.
        La perturbación se propaga a T_out a través de la dinámica Gd(s).
        
        Args:
            delta_T_in: cambio en T_in (°C) — perturbación escalón sostenida
        """
        if delta_T_in is not None:
            self.d_T_in = delta_T_in  # Escalón persistente
            print(f"Perturbación aplicada: ΔT_in = {delta_T_in} °C (persistente)")