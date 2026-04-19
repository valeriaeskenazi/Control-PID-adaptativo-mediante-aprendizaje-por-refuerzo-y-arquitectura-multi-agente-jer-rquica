import numpy as np

class ApplyAction:
    
    def __init__(self, 
                 delta_percent_ctrl=0.2,
                 delta_percent_orch=0.05,
                 pid_limits=None,
                 manipulable_ranges=None):
        
        self.delta_percent_ctrl = delta_percent_ctrl
        self.delta_percent_orch = delta_percent_orch
        
        # Límites para PID
        if pid_limits is None:
            self.pid_limits = [
                (0.01, 100.0),  # Kp
                (0.0, 10.0),     # Ki
                (0.0, 10.0)      # Kd
            ]
        else:
            self.pid_limits = pid_limits
        
        self.manipulable_ranges = manipulable_ranges
        
        # Mapeos fijos para modo discreto
        self.ACTION_MAP_CTRL = {
            0: ('Kp', +1),  # Kp ↑
            1: ('Ki', +1),  # Ki ↑
            2: ('Kd', +1),  # Kd ↑
            3: ('Kp', -1),  # Kp ↓
            4: ('Ki', -1),  # Ki ↓
            5: ('Kd', -1),  # Kd ↓
            6: ('mantener', 0)
        }
        
        self.ACTION_MAP_ORCH = {
            0: +1,   # Aumentar SP
            1: -1,   # Disminuir SP
            2: 0     # Mantener
        }
    
    def translate(self, action, agent_type, action_type, current_values):
        """
        Traduce la acción del agente a parámetros de control.
        
            - Para ctrl: [(kp1_new, ki1_new, kd1_new), (kp2_new, ki2_new, kd2_new), ...]
            - Para orch: [sp1_new, sp2_new, ...]
        """
        
        # CASO 1: CTRL + CONTINUOUS
        if agent_type == 'ctrl' and action_type == 'continuous':
            n_vars = len(current_values)
            new_params = []
            
            # Reshape flat array a tuplas
            action_reshaped = action.reshape(n_vars, 3)

            for i in range(n_vars):
                delta_kp, delta_ki, delta_kd = action_reshaped[i]

                # Aplicar deltas
                kp_new = current_values[i][0] + delta_kp
                ki_new = current_values[i][1] + delta_ki
                kd_new = current_values[i][2] + delta_kd
                
                # Clipear
                kp_new = np.clip(kp_new, self.pid_limits[0][0], self.pid_limits[0][1])
                ki_new = np.clip(ki_new, self.pid_limits[1][0], self.pid_limits[1][1])
                kd_new = np.clip(kd_new, self.pid_limits[2][0], self.pid_limits[2][1])
                
                new_params.append((kp_new, ki_new, kd_new))
            
            return new_params
        
        # CASO 2: CTRL + DISCRETE
        elif agent_type == 'ctrl' and action_type == 'discrete':
            n_vars = len(current_values)
            new_params = []
            
            for i in range(n_vars):
                action_idx = action[i]
                param_name, direction = self.ACTION_MAP_CTRL[action_idx]
                
                kp, ki, kd = current_values[i]
                
                if param_name != 'mantener':
                    multiplier = 1.0 + (direction * self.delta_percent_ctrl)
                    
                    if param_name == 'Kp':
                        kp *= multiplier
                    elif param_name == 'Ki':
                        ki *= multiplier
                    elif param_name == 'Kd':
                        kd *= multiplier
                
                # Clipear
                kp = np.clip(kp, self.pid_limits[0][0], self.pid_limits[0][1])
                ki = np.clip(ki, self.pid_limits[1][0], self.pid_limits[1][1])
                kd = np.clip(kd, self.pid_limits[2][0], self.pid_limits[2][1])
                
                new_params.append((kp, ki, kd))
            
            return new_params
        
        # CASO 3: ORCH + CONTINUOUS
        elif agent_type == 'orch' and action_type == 'continuous':
            n_vars = len(current_values)
            new_sps = []
            
            for i in range(n_vars):
                # action[i] está en [-1, 1] por el tanh del actor
                # Se escala por el rango de la variable y delta_percent_orch
                # para que el delta tenga sentido físico
                # Ejemplo: action=1.0, rango T=120K, delta_percent=0.05 → delta=+6K
                range_size = self.manipulable_ranges[i][1] - self.manipulable_ranges[i][0]
                delta_sp = action[i] * range_size * self.delta_percent_orch
                sp_new = current_values[i] + delta_sp
                
                # Clipear con manipulable_ranges
                sp_new = np.clip(sp_new, 
                                self.manipulable_ranges[i][0], 
                                self.manipulable_ranges[i][1])
                
                new_sps.append(sp_new)
            
            return new_sps
        
        # CASO 4: ORCH + DISCRETE
        elif agent_type == 'orch' and action_type == 'discrete':
            n_vars = len(current_values)
            new_sps = []
            
            for i in range(n_vars):
                action_idx = action[i]
                direction = self.ACTION_MAP_ORCH[action_idx]
                
                sp = current_values[i]
                
                if direction != 0:  # Aumentar o disminuir
                    multiplier = 1.0 + (direction * self.delta_percent_orch)
                    sp *= multiplier
                # Si direction == 0, sp no cambia (mantener)
                
                # Clipear
                sp = np.clip(sp, 
                            self.manipulable_ranges[i][0], 
                            self.manipulable_ranges[i][1])
                
                new_sps.append(sp)
            
            return new_sps