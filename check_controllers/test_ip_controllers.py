import numpy as np
import matplotlib.pyplot as plt
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode

# --- Configuration ---
SIMULATION_TIME = 5.0  # seconds
DT = 0.002
INITIAL_STATE = np.array([0, 0.0, 0.4, 0.0])

# Cost function matrix and parameters
p_mtx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

UPRIGHT_ANGLE_THRESHOLD = 0.02
UPRIGHT_ANG_VEL_THRESHOLD = 0.2
STUCK_PENALTY_FACTOR = 1.0


def eval_cost(state):
    v = state.reshape(4, 1)
    if abs(v[2,0])< UPRIGHT_ANGLE_THRESHOLD and abs(v[3,0]<UPRIGHT_ANG_VEL_THRESHOLD):
        b = abs(v[0,0])
    else:
        b = 0
    return (v.T @ p_mtx @ v).item()+2*b


# --- Simulation Function ---
def sim(initial_state, controller):
    """
    Runs an inverted pendulum simulation for a given initial state and controller.

    Args:
        initial_state (np.ndarray): The initial state vector [x, x_dot, alpha, alpha_dot].
        controller: A controller object with an `update_control(state)` method.

    Returns:
        tuple: A tuple containing three NumPy arrays:
            - states (np.ndarray): The history of states over time.
            - control_efforts (np.ndarray): The history of control efforts (force u).
            - costs (np.ndarray): The history of the calculated cost.
    """
    env = InvePendulum(dt=DT)
    state = np.array(env.reset(initial_state=initial_state))

    states_history = []
    control_effort_history = []
    cost_history = []

    num_steps = int(SIMULATION_TIME / DT)

    for _ in range(num_steps):
        states_history.append(state)

        cost = eval_cost(state)
        cost_history.append(cost)

        action = controller.update_control(state)
        control_effort_history.append(action)

        state = np.array(env.step_sim(action))

    return np.array(states_history), np.array(control_effort_history), np.array(cost_history)


# --- Main Execution Block ---
if __name__ == '__main__':
    # Initialize Environment and All Controllers
    env = InvePendulum(dt=DT)
    controllers = {"LQR": LQR(10.0, 12.60, 48.33, 9.09), "SM": SlidingMode(env), "VF": LQR(0, 30.92, 87.63, 20.40)}
    # controllers = {"LQR": LQR(10.0, 12.60, 48.33, 9.09)}

    # Dictionary to store simulation data for each controller
    simulation_data = {}

    # --- Run Simulation for Each Controller ---
    for name, controller_obj in controllers.items():
        states, control_effort, cost = sim(INITIAL_STATE, controller_obj)
        costs_to_go = np.zeros_like(cost)
        costs_to_go[-1] = cost[-1]
        for i in reversed(range(len(cost) - 1)):
            costs_to_go[i] = cost[i] + 0.5 * costs_to_go[i + 1]
        simulation_data[name] = {'states': states, 'control_effort': control_effort, 'cost': cost, 'cost_to_go': costs_to_go}

    # --- Plot Comparative Results ---
    time_vector = np.arange(0, SIMULATION_TIME, DT)

    # Use a color cycle for consistent plotting
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Controller Performance Comparison", fontsize=16)

    # Plot 1: State Trajectories
    for i, (name, data) in enumerate(simulation_data.items()):
        # Plot Pendulum Angle (solid line)
        axs[0].plot(time_vector, data['states'][:, 2], label=fr'{name} - $\alpha(t)$', color=colors[i])
        # Plot Cart Position (dashed line)
        axs[0].plot(time_vector, data['states'][:, 0], label=fr'{name} - $x(t)$', color=colors[i], linestyle='--')
    axs[0].set_ylabel("Position (m) / Angle (rad)")
    axs[0].legend(loc='upper right')
    axs[0].grid(True)
    axs[0].set_title("State Trajectories")

    # Plot 2: Control Effort
    for i, (name, data) in enumerate(simulation_data.items()):
        axs[1].plot(time_vector, data['control_effort'], label=f'{name}', color=colors[i])
    axs[1].set_ylabel("Control Effort (N)")
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    axs[1].set_title("Control Effort")

    # Plot 3: Cost
    for i, (name, data) in enumerate(simulation_data.items()):
        axs[2].plot(time_vector, data['cost'], label=f'{name}', color=colors[i])
        axs[2].plot(time_vector, data['cost_to_go'], label=f'{name}', color=colors[i], linestyle='--')
    axs[2].set_ylabel("Cost")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc='upper right')
    axs[2].grid(True)
    axs[2].set_title("Cost Over Time")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
