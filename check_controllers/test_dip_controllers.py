# run_single_controller.py

import numpy as np
import matplotlib.pyplot as plt
from environment.double_cart_pendulum.double_cart_pendulum_dynamics import DoubleCartPendulum
from environment.double_cart_pendulum.double_pendulum_animation import DoublePendulumLiveRenderer
import time
# --- Configuration ---
# Change this variable to test different controllers: 'LQR', 'SM', or 'VF'

SIMULATION_TIME = 5.0  # seconds
DT = 0.002

# We use the same initial condition as the paper's demonstration trajectory in Figure 7
INITIAL_STATE = np.array([0, 0.0, 0.2, 0.0, 0, 0])

if __name__ == '__main__':
    # --- Initialize Environment and All Controllers ---
    env = DoubleCartPendulum(dt=DT)

    # --- Run Simulation Loop ---
    state = env.reset(initial_state=INITIAL_STATE)

    history = {'time': [], 'states': [], 'control_effort': [], 'cost': []}

    pendulum_renderer = DoublePendulumLiveRenderer(env)

    num_steps = int(SIMULATION_TIME / DT)
    pendulum_renderer.init_live_render()
    for i in range(num_steps):
        # Log data before stepping
        history['time'].append(i * DT)
        history['states'].append(state)
        # v = np.array(state).reshape(4, 1)
        # history['cost'].append((v.T @ p_mtx @ v)[0])

        # # Calculate the control action (force u)

        # action = chosen_controller.update_control(state)
        # history['control_effort'].append(action)

        # # Step the simulation
        state = env.step_sim(0)
        time.sleep(env.dt)
        if (i % 30 == 0):
            pendulum_renderer.update_live_render()

    print("Simulation finished.")

    # # --- Plot Results ---
    # states_history = np.array(history['states'])

    # fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    # fig.suptitle(f"Performance of Individual '{CONTROLLER_TO_TEST}' Controller", fontsize=16)

    # # Plot 1: Position States
    # axs[0].plot(history['time'], states_history[:, 2], label=r'$\alpha(t)$ - Pendulum Angle')
    # axs[0].plot(history['time'], states_history[:, 0], label=r'$x(t)$ - Cart Position', linestyle='--')
    # axs[0].set_ylabel("Position (m) / Angle (rad)")
    # axs[0].legend()
    # axs[0].grid(True)
    # axs[0].set_title("State Trajectories")

    # # Plot 2: Control Effort
    # axs[1].plot(history['time'], history['control_effort'], label=r'$u(t)$ - Force', color='g')
    # axs[1].set_ylabel("Control Effort (N)")
    # axs[1].set_xlabel("Time (s)")
    # axs[1].legend()
    # axs[1].grid(True)
    # axs[1].set_title("Control Effort Over Time")

    # # Plot 2: Control Effort
    # axs[2].plot(history['time'], history['cost'], label=r'cost', color='g')
    # axs[2].set_ylabel("Cost")
    # axs[2].set_xlabel("Time (s)")
    # axs[2].legend()
    # axs[2].grid(True)
    # axs[2].set_title("Control cost Over Time")

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()
