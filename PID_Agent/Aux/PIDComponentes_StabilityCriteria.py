import numpy as np
from typing import List, Tuple, Dict


class StabilityCriteria:
    """
    Verifica 3 criterios por variable:
        1. Error disminuye o se mantiene pequeño
        2. No hay oscilaciones evidentes (cambios de signo en el error)
        3. No hay cambios abruptos en la señal de control
    """

    def __init__(self,
                 error_increase_tolerance: float = 1.5,
                 max_sign_changes_ratio: float = 0.2,
                 max_abrupt_change_ratio: float = 0.05,
                 abrupt_change_threshold: float = 0.3):

        self.error_increase_tolerance = error_increase_tolerance
        self.max_sign_changes_ratio = max_sign_changes_ratio
        self.max_abrupt_change_ratio = max_abrupt_change_ratio
        self.abrupt_change_threshold = abrupt_change_threshold

    def check_all(self,
                  trajs_pv: List[List[float]], # Lista de trayectorias de PV [[pv_var1], [pv_var2], ...]
                  trajs_control: List[List[float]], # Lista de trayectorias de control [[ctrl_var1], [ctrl_var2], ...]
                  setpoints: List[float]) -> Dict: # Lista de setpoints [sp_var1, sp_var2, ...]
        """
        Evalúa los criterios de estabilidad para todas las variables.
        """
        n_vars = len(trajs_pv)
        vars_ok = []
        criterios_por_var = []
        detalles = []

        for i in range(n_vars):
            result = self._check_variable(
                traj_pv=trajs_pv[i],
                traj_control=trajs_control[i],
                setpoint=setpoints[i]
            )
            vars_ok.append(result['cumple_todos'])
            criterios_por_var.append(result['n_cumplidos'])
            detalles.append(result)

        n_ok = sum(vars_ok)
        ratio = n_ok / n_vars if n_vars > 0 else 0.0

        return {
            'ratio': ratio, #float  — fracción de variables que cumplen los 3 criterios
            'vars_ok': vars_ok, #list[bool] — True si la variable cumplió los 3 criterios
            'criterios_por_var': criterios_por_var, #list[int]  — cuántos criterios cumplió cada variable (0-3)
            'detalles': detalles #list[dict] — detalles de cada criterio por variable
        }

    def _check_variable(self,
                        traj_pv: List[float],
                        traj_control: List[float],
                        setpoint: float) -> Dict:
        """
        Evalúa los 3 criterios para una sola variable.
        """
        # Calcular error a partir de la trayectoria de PV
        error_history = [setpoint - pv for pv in traj_pv]

        c1, detail_c1 = self._check_error_trend(error_history)
        c2, detail_c2 = self._check_oscillations(error_history)
        c3, detail_c3 = self._check_abrupt_changes(traj_control)

        criterios = [c1, c2, c3]
        n_cumplidos = sum(criterios)

        return {
            'cumple_todos': all(criterios),
            'n_cumplidos': n_cumplidos,
            'criterio_1_error_trend': c1,
            'criterio_2_oscilaciones': c2,
            'criterio_3_cambios_abruptos': c3,
            'detalle_c1': detail_c1,
            'detalle_c2': detail_c2,
            'detalle_c3': detail_c3
        }

    def _check_error_trend(self, error_history: List[float]) -> Tuple[bool, dict]:
        """
        Criterio 1: El error disminuye o se mantiene pequeño.
        """
        if len(error_history) < 2:
            return True, {'razon': 'Trayectoria muy corta, criterio omitido'}

        errores_abs = np.abs(error_history)
        mitad = max(1, len(error_history) // 2)

        error_inicial = float(np.mean(errores_abs[:mitad]))
        error_final = float(np.mean(errores_abs[mitad:]))

        cumple = error_final <= error_inicial * self.error_increase_tolerance

        return cumple, {
            'error_inicial': error_inicial,
            'error_final': error_final,
            'tolerancia': self.error_increase_tolerance,
            'razon': (
                'OK' if cumple
                else f'Error aumentó de {error_inicial:.4f} a {error_final:.4f} '
                     f'(tolerancia: {self.error_increase_tolerance}x)'
            )
        }

    def _check_oscillations(self, error_history: List[float]) -> Tuple[bool, dict]:
        """
        Criterio 2: No hay oscilaciones evidentes.
        """
        if len(error_history) < 3:
            return True, {'razon': 'Trayectoria muy corta, criterio omitido'}

        signos = np.sign(error_history)
        # Ignorar ceros (error exactamente en SP)
        signos_no_cero = signos[signos != 0]

        if len(signos_no_cero) < 2:
            return True, {'razon': 'Sin suficientes puntos con error ≠ 0'}

        cambios_signo = int(np.sum(np.diff(signos_no_cero) != 0))
        ratio_cambios = cambios_signo / len(signos_no_cero)

        cumple = ratio_cambios < self.max_sign_changes_ratio

        return cumple, {
            'cambios_signo': cambios_signo,
            'ratio_cambios': ratio_cambios,
            'umbral': self.max_sign_changes_ratio,
            'razon': (
                'OK' if cumple
                else f'Oscilaciones detectadas: {cambios_signo} cambios de signo '
                     f'(ratio {ratio_cambios:.2%}, umbral {self.max_sign_changes_ratio:.2%})'
            )
        }

    def _check_abrupt_changes(self, traj_control: List[float]) -> Tuple[bool, dict]:
        """
        Criterio 3: No hay cambios abruptos en la señal de control.
        """
        if len(traj_control) < 2:
            return True, {'razon': 'Trayectoria de control muy corta, criterio omitido'}

        control_arr = np.array(traj_control, dtype=float)
        cambios = np.abs(np.diff(control_arr))

        rango = float(np.max(control_arr) - np.min(control_arr))
        if rango > 1e-8:
            cambios_norm = cambios / rango
        else:
            # Señal casi constante → no hay cambios abruptos
            return True, {'razon': 'Señal de control casi constante, sin cambios abruptos'}

        umbral = self.abrupt_change_threshold
        n_abruptos = int(np.sum(cambios_norm > umbral))
        ratio_abruptos = n_abruptos / len(cambios_norm)

        cumple = ratio_abruptos < self.max_abrupt_change_ratio

        return cumple, {
            'n_cambios_abruptos': n_abruptos,
            'ratio_abruptos': ratio_abruptos,
            'max_cambio_norm': float(np.max(cambios_norm)),
            'umbral_cambio': umbral,
            'umbral_ratio': self.max_abrupt_change_ratio,
            'razon': (
                'OK' if cumple
                else f'Cambios abruptos: {n_abruptos} pasos sobre umbral '
                     f'(ratio {ratio_abruptos:.2%}, umbral {self.max_abrupt_change_ratio:.2%})'
            )
        }