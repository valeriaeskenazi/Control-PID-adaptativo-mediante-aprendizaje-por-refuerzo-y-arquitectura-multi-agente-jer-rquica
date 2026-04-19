import time
import numpy as np

class ResponseTimeDetector:
    
    def __init__(self, proceso, env_type='simulation', dt=1.0, tolerance=0.05):
        self.proceso = proceso
        self.env_type = env_type
        self.dt = dt
        self.tolerance = tolerance

    def estimate(self, pvs_inicial: list, sps: list, 
                pid_controllers: list, max_time: float = 1800, reset_pid=True):
        if self.env_type == 'simulation':
            return self._estimate_multi(pvs_inicial, sps, pid_controllers, max_time, reset_pid)
        elif self.env_type == 'real':
            return self._estimate_online(pvs_inicial, sps, pid_controllers, max_time, reset_pid)

    def _estimate_multi(self, pvs_inicial: list, sps: list, 
                    pid_controllers: list, max_time: float = 1800, reset_pid=True):
        if reset_pid:
            for pid in pid_controllers:
                pid.reset()
        
        n_vars = len(pvs_inicial)
        resultado = {
            'pvs_final': list(pvs_inicial),
            'tiempos': [0.0] * n_vars,
            'trayectorias_pv': [[pv] for pv in pvs_inicial],
            'trayectorias_control': [[] for _ in range(n_vars)],
            'converged': [False] * n_vars
        }
        
        pvs = list(pvs_inicial)
        t = 0
        dead_bands = [
            max(self.tolerance * abs(sp - pv0), self.tolerance * 0.5)
            for sp, pv0 in zip(sps, pvs_inicial)
        ]
        
        while t < max_time:
            # Chequear si TODAS convergieron
            all_converged = all(
                abs(sps[i] - pvs[i]) <= dead_bands[i]
                for i in range(n_vars)
            )
            if all_converged:
                for i in range(n_vars):
                    resultado['converged'][i] = True
                    resultado['tiempos'][i] = t
                break
            
            # Cada PID calcula su output con el PV actual
            control_outputs = []
            for i in range(n_vars):
                u = pid_controllers[i].compute(setpoint=sps[i], process_value=pvs[i])
                control_outputs.append(u)
                resultado['trayectorias_control'][i].append(u)
            
            # UN SOLO paso del CSTR con ambos outputs
            # Completar control_outputs si el simulador espera más variables que las controladas
            full_outputs = list(self.proceso.external_process.state[3:3+2]) if hasattr(self.proceso, 'external_process') else list(control_outputs)
            full_outputs[:len(control_outputs)] = control_outputs
            pvs = self.proceso.simulate_step_multi(full_outputs, self.dt)

            for i in range(n_vars):
                resultado['trayectorias_pv'][i].append(pvs[i])
            
            t += self.dt
        
        # Timeout o fin
        resultado['pvs_final'] = list(pvs)
        for i in range(n_vars):
            if not resultado['converged'][i]:
                resultado['tiempos'][i] = max_time
        
        return resultado

    def _estimate_online(self, pv_inicial, sp, pid_controller, max_time):
        # Resetear PID
        pid_controller.reset()
        
        # Inicializar resultado
        resultado = {
            'tiempo': 0,
            'trayectoria_pv': [],
            'trayectoria_control': [],
            'pv_final': pv_inicial,
            'converged': False
        }
        
        t = 0
        dead_band = self.tolerance * abs(sp)

        # Leer PV inicial
        pv_actual = self.proceso.read_pv(self.variable_index)
        resultado['trayectoria_pv'].append(pv_actual)
        
        # Medir en tiempo real
        while abs(sp - pv_actual) > dead_band:
            # Calcular control
            control_output = pid_controller.compute(
                setpoint=sp,
                process_value=pv_actual
            )
            resultado['trayectoria_control'].append(control_output)
            
            # Escribir control al proceso real
            self.proceso.write_control(
                control_output=control_output,
                variable_index=self.variable_index
            )
            
            # Esperar tiempo REAL
            time.sleep(self.dt)
            t += self.dt

            # Leer nuevo PV
            pv_actual = self.proceso.read_pv(self.variable_index)
            resultado['trayectoria_pv'].append(pv_actual)
            
            # Timeout
            if t >= max_time:
                resultado['tiempo'] = max_time
                resultado['pv_final'] = pv_actual
                resultado['converged'] = False
                return resultado
        
        # Convergencia exitosa
        resultado['tiempo'] = t
        resultado['pv_final'] = pv_actual
        resultado['converged'] = True
        
        return resultado