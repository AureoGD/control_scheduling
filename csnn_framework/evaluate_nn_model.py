import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from datetime import datetime
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode
from csnn_framework.nn_model import NNModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration ---
SIMULATION_TIME = 5.0  # seconds
DT = 0.001
INITIAL_STATE = np.array([0.0, 0, 0.25, 0.0])

# Cost function matrix (MUST match training)
P_MATRIX = np.array(
    [
        [1.25, 0.0, 0.0, 0.0],  # x position
        [0.0, 1.0, 0.0, 0.0],  # x velocity
        [0.0, 0.0, 8.0, 0.0],  # theta angle (PRIORITY - pendulum upright)
        [0.0, 0.0, 0.0, 1.0]  # theta velocity
    ],
    dtype=np.float32)

DELTA = 0.5
MODEL_DIR = "models/csnn/31181649"  # Model directory
SAVE_DATA_DIR = "csnn_framework/analysis"  # Directory to save analysis data


def ensure_directory_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_analysis_data(simulation_data, save_dir):
    """Save analysis data to CSV files for each controller."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for controller_name, data in simulation_data.items():
        # Create DataFrame with all the data
        df_data = {
            'time': np.linspace(0, SIMULATION_TIME, len(data['states'])),
            'x': data['states'][:, 0],
            'dx': data['states'][:, 1],
            'angle': data['states'][:, 2],
            'dangle': data['states'][:, 3],
            'control_effort': data['control_effort'],
            'immediate_cost': data['immediate_cost'],
            'predicted_cost_to_go': data['cost_pred'],
            'true_cost_to_go': data['cost_to_go']
        }

        df = pd.DataFrame(df_data)

        # Save to CSV
        filename = f"{controller_name.lower()}_analysis_{timestamp}.csv"
        filepath = os.path.join(save_dir, filename)
        df.to_csv(filepath, index=False)

        # Also save a summary statistics file
        summary_stats = {
            'controller': [controller_name],
            'timestamp': [timestamp],
            'total_steps': [len(data['states'])],
            'final_angle': [data['states'][-1, 2]],
            'final_position': [data['states'][-1, 0]],
            'max_control_effort': [np.max(np.abs(data['control_effort']))],
            'rmse': [np.sqrt(np.mean((data['cost_pred'] - data['cost_to_go'])**2))],
            'mae': [np.mean(np.abs(data['cost_pred'] - data['cost_to_go']))],
            'correlation': [np.corrcoef(data['cost_pred'], data['cost_to_go'])[0, 1]]
        }

        summary_df = pd.DataFrame(summary_stats)
        summary_filename = f"summary_{timestamp}.csv"
        summary_filepath = os.path.join(save_dir, summary_filename)

        # Append to summary file if it exists, otherwise create new
        if os.path.exists(summary_filepath):
            existing_summary = pd.read_csv(summary_filepath)
            updated_summary = pd.concat([existing_summary, summary_df], ignore_index=True)
            updated_summary.to_csv(summary_filepath, index=False)
        else:
            summary_df.to_csv(summary_filepath, index=False)


def load_model_and_scalers(controller_name, model_dir):
    """Load model and corresponding scalers."""
    model_path = os.path.join(model_dir, f"{controller_name.lower()}_model.pth")
    scalers_path = os.path.join(model_dir, f"{controller_name.lower()}_scalers.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Model or scalers not found for {controller_name}")

    # Load model
    model = NNModel().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load scalers
    scalers = joblib.load(scalers_path)

    return model, scalers


def predict_cost_to_go(model, scalers, state):
    """Predict cost-to-go for a state using normalized model."""
    # Convert to numpy array if needed
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    # Normalize state
    state_norm = scalers['state_scaler'].transform(state.reshape(1, -1))
    state_norm_tensor = torch.tensor(state_norm, dtype=torch.float32).to(DEVICE)

    # Predict (normalized output)
    with torch.no_grad():
        pred_norm = model(state_norm_tensor).cpu().numpy()

    # Denormalize prediction
    pred = scalers['target_scaler'].inverse_transform(pred_norm.reshape(-1, 1))

    return pred.item()


def eval_cost(state):
    """Calculate immediate cost for evaluation."""
    return (state.T @ P_MATRIX @ state).item()


def calculate_true_cost_to_go(immediate_costs, delta):
    """
    Calculate the true cost-to-go for each state in a trajectory.
    This is what the neural network should be predicting.
    J(xₖ) = U(xₖ) + δ * J(xₖ₊₁)
    """
    if len(immediate_costs) == 0:
        return np.array([])

    costs_to_go = np.zeros_like(immediate_costs)
    costs_to_go[-1] = immediate_costs[-1]  # Final cost is just immediate cost

    # Backward pass: J(xₖ) = U(xₖ) + δ * J(xₖ₊₁)
    for i in reversed(range(len(immediate_costs) - 1)):
        costs_to_go[i] = immediate_costs[i] + delta * costs_to_go[i + 1]

    return costs_to_go


def sim(initial_state, controller, controller_name, model_dir):
    """
    Run simulation with cost prediction using the trained model.
    """
    # Load model and scalers
    model, scalers = load_model_and_scalers(controller_name, model_dir)

    env = InvePendulum(dt=DT)
    state = np.array(env.reset(initial_state=initial_state))

    states_history = []
    control_effort_history = []
    immediate_cost_history = []  # Immediate cost U(x)
    cost_pred_history = []  # NN predicted cost-to-go Ĵ(x)
    # cost_to_go_history will be calculated after simulation

    num_steps = int(SIMULATION_TIME / DT)

    for idx in range(num_steps):
        current_state = state.copy()

        # Predict cost-to-go using model (what we want to validate)
        predicted_cost = predict_cost_to_go(model, scalers, current_state)

        # Calculate immediate cost
        immediate_cost = eval_cost(current_state)

        # Get control action
        action = controller.update_control(current_state)

        # Record data every step
        states_history.append(current_state)
        immediate_cost_history.append(immediate_cost)
        cost_pred_history.append(predicted_cost)
        control_effort_history.append(action)

        # Step simulation
        state = np.array(env.step_sim(action))

        # Early termination if pendulum falls
        if abs(state[2]) > 0.55:
            break

    # Calculate true cost-to-go for comparison (after simulation)
    true_cost_to_go = calculate_true_cost_to_go(immediate_cost_history, DELTA)

    return (np.array(states_history), np.array(control_effort_history), np.array(immediate_cost_history), np.array(cost_pred_history), np.array(true_cost_to_go))


def plot_results(simulation_data, time_vector):
    """Plot comparative results with proper cost comparison."""
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    colors2 = ["#d62728", '#9467bd', '#8c564b']  # Different colors for predictions

    fig, axs = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Controller Performance Comparison - Cost-to-Go Validation", fontsize=16)

    # Plot 1: State Trajectories
    for i, (name, data) in enumerate(simulation_data.items()):
        axs[0].plot(time_vector, data['states'][:, 2], label=fr'{name} - $\alpha(t)$', color=colors[i], linewidth=2)
        axs[0].plot(time_vector, data['states'][:, 0], label=fr'{name} - $x(t)$', color=colors[i], linestyle='--', linewidth=2)
    axs[0].set_ylabel("State Variables")
    axs[0].legend(loc='upper right')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("State Trajectories")

    # Plot 2: Control Effort
    for i, (name, data) in enumerate(simulation_data.items()):
        axs[1].plot(time_vector, data['control_effort'], label=f'{name}', color=colors[i], linewidth=2)
    axs[1].set_ylabel("Control Effort (N)")
    axs[1].legend(loc='upper right')
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title("Control Effort")

    # Plot 3: Immediate Cost (for reference)
    for i, (name, data) in enumerate(simulation_data.items()):
        axs[2].plot(time_vector, data['immediate_cost'], label=f'{name}', color=colors[i], alpha=0.7, linewidth=1.5)
    axs[2].set_ylabel("Immediate Cost")
    axs[2].legend(loc='upper right')
    axs[2].grid(True, alpha=0.3)
    axs[2].set_title("Immediate Cost U(x) = xᵀPx")

    # Plot 4: Cost-to-Go Comparison (THE IMPORTANT ONE)
    for i, (name, data) in enumerate(simulation_data.items()):
        # True cost-to-go
        axs[3].plot(time_vector, data['cost_to_go'], label=f'{name}_true', color=colors[i], linestyle='-', linewidth=3, alpha=0.8)
        # Predicted cost-to-go
        axs[3].plot(time_vector, data['cost_pred'], label=f'{name}_predicted', color=colors2[i], linestyle='-', linewidth=2, alpha=0.8)

    axs[3].set_ylabel("Cost-to-Go J(x)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(loc='upper right')
    axs[3].grid(True, alpha=0.3)
    axs[3].set_title("Cost-to-Go: True vs Predicted (Should Match!)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def main():
    """Main test function."""
    # Ensure analysis directory exists
    ensure_directory_exists(SAVE_DATA_DIR)

    # Initialize controllers
    env = InvePendulum(dt=DT)
    controllers = {"LQR": LQR(-2.91, -3.67, -25.43, -4.94), "SM": SlidingMode(env), "VF": LQR(0, -33.90, -153.30, -32.07)}

    # Test each controller
    simulation_data = {}
    for name, controller_obj in controllers.items():
        try:
            states, control_effort, immediate_cost, cost_pred, cost_to_go = sim(INITIAL_STATE, controller_obj, name, MODEL_DIR)
            simulation_data[name] = {'states': states, 'control_effort': control_effort, 'immediate_cost': immediate_cost, 'cost_pred': cost_pred, 'cost_to_go': cost_to_go}
        except Exception as e:
            raise Exception(f"Error testing {name}: {e}")

    # Plot results
    if simulation_data:
        # Find minimum length for consistent plotting
        min_length = min(len(data['states']) for data in simulation_data.values())
        time_vector = np.linspace(0, SIMULATION_TIME, min_length)

        # Trim all arrays to same length
        trimmed_data = {}
        for name, data in simulation_data.items():
            trimmed_data[name] = {
                'states': data['states'][:min_length],
                'control_effort': data['control_effort'][:min_length],
                'immediate_cost': data['immediate_cost'][:min_length],
                'cost_pred': data['cost_pred'][:min_length],
                'cost_to_go': data['cost_to_go'][:min_length]
            }

        # Plot and save analysis data
        plot_results(trimmed_data, time_vector)
        save_analysis_data(trimmed_data, SAVE_DATA_DIR)
    else:
        raise Exception("No simulation data to plot or save.")


if __name__ == '__main__':
    main()
