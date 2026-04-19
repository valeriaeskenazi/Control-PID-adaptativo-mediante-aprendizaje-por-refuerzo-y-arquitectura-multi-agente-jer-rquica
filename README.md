# Control PID Adaptativo mediante Aprendizaje por Refuerzo y Arquitectura Multi-Agente Jerárquica

Proyecto de la maestría de IA. Se trata de una propuesta con agentes agnósticos al problema matemático para el ajuste de controladores industriales.

## Estructura del proyecto

```
PID_AGENT/
├── Agente/
│   ├── Abstract_agent.py
│   ├── memory.py
│   ├── Actor_Critic/
│   │   ├── algorithm_AC.py
│   │   ├── model_AC.py
│   │   ├── train_AC.py
│   │   └── transfer_learning.py
│   ├── PPO/
│   │   ├── algorithm_PPO.py
│   │   ├── model_PPO.py
│   │   └── train_PPO.py
│   └── DQN/
│       ├── algorithm_DQN.py
│       ├── model_DQN.py
│       └── train_DQN.py
├── Aux/
│   ├── PIDComponents_StabilityCriteria.py
│   ├── PIDComponents_translate.py
│   ├── PIDComponents_PID.py
│   ├── PIDComponents_Reward.py
│   ├── PIDComponents_tima.py
│   └── Plots.py
├── Environment/
│   ├── PIDControlEnv_simple.py
│   ├── PIDControlEnv_complex.py
│   └── Simulation_Env/
│       ├── Heat_Exchanger.py
│       ├── Reactor_CSTR.py
│       ├── Reactor_Cyclopentanol.py
│       ├── Tanque_simple.py
│       └── SimulationEnv.py
└── Entrenamiento/
    ├── CTRL/
    │   ├── AC/
    │   │   ├── CSTR/
    │   │   │   ├── AC_CTRL_COLAB.ipynb
    │   │   │   └── Agent_ctrl_best.pt
    │   │   ├── Cyclopentanol/
    │   │   │   ├── AC_hiperparametros_Cyclopentanol.ipynb
    │   │   │   ├── AC_hiperparametros_Cyclopentanol_V2.ipynb
    │   │   │   ├── AC_hiperparametros_Cyclopentanol_V3.ipynb
    │   │   │   └── AC_TransferLearining_Cyclopentanol.ipynb
    │   │   └── HeatExchanger/
    │   │       └── AC_hiperparametros_HeatExchanger_v3.ipynb
    │   ├── DQN/
    │   │   ├── agent_ctrl_best_Test2.pt
    │   │   ├── agent_ctrl_best.pt
    │   │   ├── DQN_CTRL_Colab_Test2.ipynb
    │   │   ├── DQN_CTRL_Colab.ipynb
    │   │   ├── DQN_CTRL_Test2_graficos.ipynb
    │   │   ├── sweep_DQN_colab_1.ipynb
    │   │   └── sweep_DQN_Test_2_Colab.ipynb
    │   └── PPO/
    │       ├── agent_ctrl_best.pt
    │       ├── PPO_CTRL_Colab_New_Run_graficos.ipynb
    │       ├── PPO_CTRL_Colab_New_Run.ipynb
    │       └── PPO_CTRL_Colab.ipynb
    └── ORCH/
        ├── AC/
        │   ├── AC_ORCH_Colab_15000_PRUEBAS.ipynb
        │   ├── AC_ORCH_Colab_TEST_4.ipynb
        │   ├── AC_ORCH_Colab_freq_1.ipynb
        │   └── AC_ORCH_Colab_freq_7.ipynb
        └── DQN/
            ├── agent_orch_best_test-4.pt
            ├── agent_orch_best_test2.pt
            ├── DQN_ORCH_Colab_Test2.ipynb
            ├── DQN_ORCH_Colab_Test3.ipynb
            ├── DQN_ORCH_Colab_Test4.ipynb
            └── DQN_ORCH_Colab.ipynb
```