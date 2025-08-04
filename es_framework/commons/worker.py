import os
import torch
import numpy as np
from typing import Dict, Any, Tuple

from es_framework.commons.nn_parameters import unflatten_nn_parameters
from es_framework.commons.control_rule import ControlRule

sim_instance = None
control_rule = None
is_discrete = False
"""
    Using the continuous worker, the ControlRule produces a torque response baised into the system states.
    When usinh the discreate one, the ControlRule produces the index of one controler from the avaliable ones fro the switch_rule
"""


def init_worker(config: Dict[str, Any]):
    global sim_instance, control_rule, is_discrete

    process_id = os.getpid()

    is_discrete = config.get('is_discrete', False)

    env_cfg = config.get('env_config', {})
    control_rule_cfg = config.get('model_config', {})

    if is_discrete:
        from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
        sim_instance = InvPendulumEnv(env_id=process_id, **env_cfg, rendering=False)
        output_dim = sim_instance.action_space.n
    else:
        from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv
        sim_instance = InvPendulumEnv(env_id=process_id, **env_cfg, rendering=False)
        output_dim = sim_instance.action_space.shape[0]

    # This part is now fully robust
    control_rule = ControlRule(observation_dim=sim_instance.observation_space.shape[0], output_dim=output_dim, **control_rule_cfg)


def run_worker(task_args: Tuple[int, np.ndarray]):
    global sim_instance, control_rule

    if sim_instance is None:
        raise RuntimeError("Worker not initialized correctly.")

    task_id, nn_params_flat = task_args

    try:
        state_dict = unflatten_nn_parameters(nn_params_flat, control_rule)
        control_rule.load_state_dict(state_dict)
        max_step = sim_instance.max_step
        cumulative_fitness = 0
        for _ in range(5):
            fitness = 0
            obs, info = sim_instance.reset()
            for step in range(max_step):
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                if is_discrete:
                    probabilities = control_rule.forward(obs_tensor)
                    action = torch.argmax(probabilities, dim=-1).detach().numpy()
                else:
                    action_tensor = control_rule.forward(obs_tensor)
                    action = action_tensor.detach().numpy()
                obs, reward, terminated, truncated, info = sim_instance.step(action)
                fitness += reward
                if terminated or truncated:
                    break
            cumulative_fitness += fitness
        return task_id, cumulative_fitness / 5

    except Exception as e:
        print(f"[Proc {os.getpid()}, Task {task_id}] ERROR in worker task: {e}")
        import traceback
        traceback.print_exc()
        return task_id, -float('inf')
