"""
Simulador de reactor CSTR para producción de Ciclopentanol.
Basado en el trabajo final de Dinámica y Control de Procesos (2015).

Reacciones:
    A → B → C     (k1 = k2, Arrhenius)
    2A → D        (k3, Arrhenius)

Variables de estado: [CA, CB, T]
Variables controladas (PVs): CB (concentración de ciclopentanol) y T (temperatura)
Variables manipulables: v (flujo de entrada, L/h) y QK (calor de camisa, kJ/h)

Parámetros extraídos de las matrices A y B del informe evaluadas en estado estacionario.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Tuple, Optional, Dict, List, Union


class CyclopentanolReactor:
    """
    Simulador de reactor CSTR para producción de Ciclopentanol.
    
    Interfaz compatible con SimulationPIDEnv (misma que CSTRSimulator).
    """
    
    def __init__(
        self,
        dt: float = 0.01,  # Paso de tiempo en horas (la dinámica es rápida)
        control_limits: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (50.0, 800.0),      # v: flujo volumétrico (L/h), de v/V ∈ [5, 80] con V=10
            (-8500.0, 0.0)      # QK: calor de camisa (kJ/h), negativo = enfriamiento
        )
    ):
        # Configuración temporal
        self.dt = dt
        self.v_min, self.v_max = control_limits[0]
        self.QK_min, self.QK_max = control_limits[1]
        
        # ─────────────────────────────────────────────────────────────
        # PARÁMETROS DEL PROCESO (extraídos de matrices A, B del informe)
        # ─────────────────────────────────────────────────────────────
        
        # Geometría y condiciones de operación
        self.VR = 10.0          # Volumen del reactor (L)
        self.T0 = 403.0         # Temperatura de alimentación (K)
        self.CA0_nominal = 5.1  # Concentración de A en alimentación nominal (mol/L)
        self.CA0 = self.CA0_nominal  # Valor actual (puede perturbarse: 4.5-5.7)
        
        # Propiedades del fluido
        self.rho_Cp = 2.812     # ρ·Cp (kJ/(L·K))
        
        # Cinética - Arrhenius: k = k0 * exp(-EA/(R*T))
        # Reacción A → B (k1) y B → C (k2): k1 = k2
        self.EoverR_12 = 9760.0     # EA/(R) para k1 y k2 (K)
        self.k0_12 = 1.306e12       # Factor pre-exponencial k1 = k2 (1/h)
        
        # Reacción 2A → D (k3)
        self.EoverR_3 = 8541.0      # EA/(R) para k3 (K)
        self.k0_3 = 8.84e9          # Factor pre-exponencial k3 (L/(mol·h))
        
        # Calores de reacción (kJ/mol)
        # ΔHR < 0 = exotérmico
        self.dHR_AB = 4.16          # A → B (ligeramente endotérmico)
        self.dHR_BC = -11.0         # B → C (exotérmico)
        self.dHR_AD = -41.78        # 2A → D (exotérmico)
        
        # ─────────────────────────────────────────────────────────────
        # ESTADO ESTACIONARIO (calculado numéricamente)
        # ─────────────────────────────────────────────────────────────
        # Controles nominales en SS (del informe: v/V ≈ 18.83 h⁻¹, QK ≈ -7000 kJ/h)
        self.v_ss = 188.3       # L/h  (v/V = 18.83 h⁻¹)
        self.QK_ss = -7000.0    # kJ/h (dentro del rango operativo)
        
        # Calcular SS numéricamente para consistencia
        self.CA_ss, self.CB_ss, self.T_ss = self._compute_steady_state(
            v=self.v_ss, QK=self.QK_ss,
            x0=[1.2, 0.9, 407.0]
        )
        
        # Estado actual [CA, CB, T]
        self.state = np.array([self.CA_ss, self.CB_ss, self.T_ss])
        
        # Valores actuales de control
        self.v_current = self.v_ss
        self.QK_current = self.QK_ss
        
        print("=" * 60)
        print("  Simulador de Reactor Ciclopentanol creado")
        print(f"   Temperatura alimentación: {self.T0} K")
        print(f"   Concentración alimentación: {self.CA0} mol/L")
        print(f"   Límites v: [{self.v_min}, {self.v_max}] L/h")
        print(f"   Límites QK: [{self.QK_min}, {self.QK_max}] kJ/h")
        print(f"   Estado estacionario: CA={self.CA_ss}, CB={self.CB_ss}, T={self.T_ss}")
        print(f"   Paso de tiempo: {self.dt} h")
        print("=" * 60)
    
    # ─────────────────────────────────────────────────────────────
    # INTERFAZ COMPATIBLE CON SimulationPIDEnv
    # ─────────────────────────────────────────────────────────────
    
    def get_n_variables(self) -> int:
        """Número de variables controladas (CB y T)."""
        return 2
    
    def get_initial_pvs(self) -> List[float]:
        """Valores iniciales de PVs [CB, T]."""
        return [self.CB_ss, self.T_ss]
    
    def simulate_step_multi(self, control_outputs: list, dt: float) -> list:
        """
        Simula un paso con AMBAS variables de control simultáneamente.
        
        Args:
            control_outputs: [v, QK] — flujo y calor de camisa
            dt: paso de tiempo (h)
        
        Returns:
            [CB_meas, T_meas] — mediciones ruidosas de CB y T
        """
        # Clipear controles a límites físicos
        v = np.clip(control_outputs[0], self.v_min, self.v_max)
        QK = np.clip(control_outputs[1], self.QK_min, self.QK_max)
        
        self.v_current = v
        self.QK_current = QK
        
        u = np.array([v, QK])
        
        # Integrar ODEs
        try:
            solution = odeint(
                self._reactor_dynamics, 
                self.state, 
                [0, dt], 
                args=(u,),
                mxstep=5000
            )
            self.state = solution[-1]
        except Exception:
            # Si la integración falla, mantener el estado actual
            pass
        
        # Clip post-integración para estabilidad numérica
        self.state[0] = np.clip(self.state[0], 0.0, 10.0)   # CA
        self.state[1] = np.clip(self.state[1], 0.0, 5.0)     # CB
        self.state[2] = np.clip(self.state[2], 250.0, 600.0)  # T
        
        # Mediciones con ruido (simula sensores reales)
        CB_meas = float(self.state[1]) + np.random.uniform(-0.001, 0.001)
        T_meas = float(self.state[2]) + np.random.uniform(-0.1, 0.1)
        
        return [CB_meas, T_meas]
    
    def _reactor_dynamics(self, x: np.ndarray, t: float, u: np.ndarray) -> np.ndarray:
        """
        Ecuaciones diferenciales del reactor CSTR.
        
        Balances del informe (Sección 1.2):
            dCA/dt = (v/V)(CA0 - CA) - k1·CA - 2·k3·CA²
            dCB/dt = -(v/V)·CB + k1·CA - k2·CB
            dT/dt  = (v/V)(T0 - T) + QK/(ρCp·VR) 
                     - (k1·CA·ΔHR_AB + k2·CB·ΔHR_BC + 2·k3·CA²·ΔHR_AD)/(ρCp)
        
        Nota: QK < 0 = enfriamiento. El término +QK/(ρCpVR) reduce T cuando QK < 0.
        """
        # Desempacar estados
        CA, CB, T = x
        
        # Protección numérica
        CA = max(CA, 0.0)
        CB = max(CB, 0.0)
        T = max(T, 250.0)
        
        # Desempacar controles
        v, QK = u
        
        # Tasa de dilución
        D = v / self.VR  # h⁻¹
        
        # Constantes cinéticas (Arrhenius)
        k1 = self.k0_12 * np.exp(-self.EoverR_12 / T)  # h⁻¹
        k2 = k1  # k1 = k2 por enunciado
        k3 = self.k0_3 * np.exp(-self.EoverR_3 / T)    # L/(mol·h)
        
        # Balances de masa
        dCAdt = D * (self.CA0 - CA) - k1 * CA - 2.0 * k3 * CA**2
        dCBdt = -D * CB + k1 * CA - k2 * CB
        
        # Balance de energía
        reaction_heat = (
            k1 * CA * self.dHR_AB 
            + k2 * CB * self.dHR_BC 
            + 2.0 * k3 * CA**2 * self.dHR_AD
        )
        
        dTdt = (
            D * (self.T0 - T)
            + QK / (self.rho_Cp * self.VR)
            - reaction_heat / self.rho_Cp
        )
        
        return np.array([dCAdt, dCBdt, dTdt])
    
    def reset(
        self,
        initial_state: Optional[np.ndarray] = None,
        randomize: bool = False
    ) -> List[float]:
        """
        Resetear simulador a condiciones iniciales.
        
        Returns:
            List[float] con [CB_inicial, T_inicial]
        """
        if initial_state is not None:
            self.state = np.array(initial_state)
        elif randomize:
            self.state = np.array([
                self.CA_ss + np.random.uniform(-0.1, 0.1),
                self.CB_ss + np.random.uniform(-0.05, 0.05),
                self.T_ss + np.random.uniform(-3.0, 3.0)
            ])
        else:
            self.state = np.array([self.CA_ss, self.CB_ss, self.T_ss])
        
        # Resetear controles
        self.v_current = self.v_ss
        self.QK_current = self.QK_ss
        
        # Resetear perturbación
        self.CA0 = self.CA0_nominal
        
        return self.get_initial_pvs()
    
    def get_state(self) -> List[float]:
        """Estado actual [CB, T, CA] — CB primero para consistencia con variable objetivo."""
        return [
            float(self.state[1]),  # CB (variable objetivo)
            float(self.state[2]),  # T  (controlada)
            float(self.state[0])   # CA (estado interno)
        ]
    
    def get_measurements(self) -> Dict[str, float]:
        """Mediciones con ruido."""
        return {
            'CA': float(self.state[0]) + np.random.uniform(-0.001, 0.001),
            'CB': float(self.state[1]) + np.random.uniform(-0.001, 0.001),
            'T': float(self.state[2]) + np.random.uniform(-0.1, 0.1)
        }
    
    def set_disturbance(self, CA0: Optional[float] = None, T0: Optional[float] = None):
        """
        Introducir perturbaciones.
        
        Args:
            CA0: Nueva concentración de alimentación (rango: 4.5-5.7 mol/L)
            T0: Nueva temperatura de alimentación (K)
        """
        if CA0 is not None:
            self.CA0 = np.clip(CA0, 4.5, 5.7)
            print(f"Perturbación: CA0 = {self.CA0} mol/L")
        if T0 is not None:
            self.T0 = T0
            print(f"Perturbación: T0 = {T0} K")
    
    def _compute_steady_state(self, v: float, QK: float, x0: list) -> tuple:
        """Calcula el estado estacionario numéricamente."""
        from scipy.optimize import fsolve
        
        def ss_eq(x):
            u = np.array([v, QK])
            return self._reactor_dynamics(np.array(x), 0.0, u)
        
        x_ss = fsolve(ss_eq, x0, full_output=False)
        return float(x_ss[0]), float(x_ss[1]), float(x_ss[2])
    
    def verify_steady_state(self, tol: float = 0.5):
        """Verificar que los parámetros producen un estado estacionario válido."""
        # Calcular derivadas en el SS con controles SS
        u_ss = np.array([self.v_ss, self.QK_ss])
        x_ss = np.array([self.CA_ss, self.CB_ss, self.T_ss])
        derivs = self._reactor_dynamics(x_ss, 0.0, u_ss)
        
        print("\n--- Verificación de Estado Estacionario ---")
        print(f"dCA/dt = {derivs[0]:.6f} (debería ser ~0)")
        print(f"dCB/dt = {derivs[1]:.6f} (debería ser ~0)")
        print(f"dT/dt  = {derivs[2]:.4f} (debería ser ~0)")
        
        # Verificar constantes cinéticas en SS
        k1_ss = self.k0_12 * np.exp(-self.EoverR_12 / self.T_ss)
        k3_ss = self.k0_3 * np.exp(-self.EoverR_3 / self.T_ss)
        print(f"\nk1 = k2 = {k1_ss:.2f} h⁻¹ (esperado: ~50.17)")
        print(f"k3 = {k3_ss:.2f} L/(mol·h) (esperado: ~6.69)")
        print(f"v/V = {self.v_ss / self.VR:.2f} h⁻¹ (esperado: 18.83)")
        
        is_valid = all(abs(d) < tol for d in derivs)
        print(f"\n{'Correcto' if is_valid else 'Incorrecto'} Estado estacionario {'válido' if is_valid else 'NO válido'}")
        return is_valid
