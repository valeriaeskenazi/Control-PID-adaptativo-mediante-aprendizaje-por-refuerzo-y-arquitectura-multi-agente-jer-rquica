import numpy as np
from typing import Tuple


class PIDController:
    
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, dt=1.0, 
                 output_limits=(-np.inf, np.inf)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits
        
        # Estado interno
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, setpoint: float, process_value: float) -> float:
        # Calcular error
        error = setpoint - process_value
        
        # Término proporcional
        P = self.kp * error
        
        # Término integral
        self.integral += error * self.dt
        I = self.ki * self.integral
        
        # Término derivativo
        derivative = (error - self.prev_error) / self.dt
        D = self.kd * derivative
        
        # Salida total
        output = P + I + D
        
        # Aplicar límites
        output_clipped = np.clip(output, *self.output_limits)
        
        # Anti-windup: si hay saturación, no acumular integral
        if output != output_clipped and self.ki > 1e-8:
            self.integral -= (output - output_clipped) / self.ki
        
        # Guardar para próxima iteración
        self.prev_error = error
        
        return float(output_clipped)
    
    def reset(self) -> None:
        # Resetear estado interno del controlador
        self.integral = 0.0
        self.prev_error = 0.0
    
    def get_params(self) -> Tuple[float, float, float]:
        # Obtener parámetros actuales del controlador
        return (self.kp, self.ki, self.kd)
    
    def get_state(self) -> dict:
        # Obtener estado interno del controlador
        return {
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd,
            'integral': self.integral,
            'prev_error': self.prev_error
        }