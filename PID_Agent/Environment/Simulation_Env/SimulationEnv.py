import numpy as np
from typing import Optional, Dict, Any


class SimulationPIDEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Configuración básica
        self.n_manipulable_vars = config.get('n_manipulable_vars', 2)
        self.manipulable_ranges = config.get('manipulable_ranges', 
                                             [(0.0, 100.0)] * self.n_manipulable_vars)
        self.dt_sim = config.get('dt_simulation', 1.0)
    
        # Estado actual del proceso (se inicializa en reset)
        self.manipulable_pvs = None
        
        # Simulador externo (se conecta con connect_external_process)
        self.external_process = None
    
    
    def simulate_step_multi(self, control_outputs: list, dt: float) -> list:
        # Para simulacion casos multivariables
        new_pvs = self.external_process.simulate_step_multi(control_outputs, dt)
        
        for i, (pv, rango) in enumerate(zip(new_pvs, self.manipulable_ranges)):
            new_pvs[i] = float(np.clip(pv, rango[0], rango[1]))
        
        self.manipulable_pvs = new_pvs
        return new_pvs
    
    def connect_external_process(self, process_simulator) -> None:
        self.external_process = process_simulator
    
    def reset(self, initial_pvs: Optional[list] = None) -> list:
        if initial_pvs is not None:
            self.manipulable_pvs = list(initial_pvs)
        else:
            self.manipulable_pvs = [
                np.random.uniform(rango[0], rango[1])
                for rango in self.manipulable_ranges
            ]
        
        if self.external_process is not None and hasattr(self.external_process, 'reset'):
            pvs_cstr = self.external_process.reset()  # [327.0, 100.0]
            self.manipulable_pvs = list(pvs_cstr)
            return self.manipulable_pvs  # retorna los del CSTR, no random
        
        return self.manipulable_pvs

    def get_state(self) -> list:
        if self.manipulable_pvs is None:
            raise RuntimeError("Ambiente no inicializado. Llama a reset() primero.")
        return self.manipulable_pvs.copy()
    
    def get_target_values(self, manipulable_pvs: list) -> list:
        if self.external_process and hasattr(self.external_process, 'get_measurements'):
            measurements = self.external_process.get_measurements()
            # Extraer variable objetivo (ejemplo: Cb)
            return [measurements.get('Cb', 0.0)]
        else:
            return [0.0] * len(manipulable_pvs)