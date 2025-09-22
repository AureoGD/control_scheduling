import os
import torch
import numpy as np
import random
import joblib
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode
from csnn_framework.switch_rule import SwitchRule
from csnn_framework.nn_model import NNModel
from csnn_framework.plot_sim import SimulationPlotter

# --- Configuration for Evaluation ---

# Paste the path to the timestamped model directory you want to evaluate.
MODEL_DIR = "models/csnn/19165411"  #31174103  # Your new model directory with scalers
# Define the initial state for the simulation [x, dx, a, da].
START_CONDITION = [0, 0.0, 0.25, 0.0]

# --- Core Simulation and System Parameters ---
SIMULATION_TIME = 6.0
DT = 0.001
SAMPLING_INTERVAL = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_scalers(controller_name, model_dir):
    """Load model and corresponding scalers for a controller."""
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


def run_evaluation(env, controllers, model_dir, initial_state):
    """
    Runs a final simulation to test the switching performance with a specific model set.
    """
    print(f"\nRunning final evaluation with models from: {model_dir}")
    print(f"Initial State: {initial_state}")

    # Load models and scalers for all controllers
    models = {}
    scalers = {}
    controller_names = list(controllers.keys())

    for name in controller_names:
        try:
            model, controller_scalers = load_model_and_scalers(name, model_dir)
            models[name] = model
            scalers[name] = controller_scalers
            print(f"✓ Loaded {name} model and scalers")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return None

    # Initialize history and switching rule
    history = {'time': [], 'x': [], 'angle': [], 'dx': [], 'da': [], 'active_controller': [], 'j_lqr': [], 'j_sm': [], 'j_vf': [], 'action': []}

    state = env.reset(initial_state=np.array(initial_state))
    num_steps = int(SIMULATION_TIME / (DT * 10))
    switcher = SwitchRule(controller_names)
    best_controller_name = controller_names[0]  # Start with first controller
    j_predictions = {name: 0 for name in controller_names}

    print("Starting simulation...")
    for step in range(num_steps):
        # Predict costs and select controller at sampling interval
        if step % SAMPLING_INTERVAL == 0:
            current_state = np.array([state])

            # Get predictions from all models (with proper normalization)
            for name in controller_names:
                try:
                    j_predictions[name] = predict_cost_to_go(models[name], scalers[name], current_state)
                except Exception as e:
                    print(f"Error predicting for {name}: {e}")
                    j_predictions[name] = float('inf')  # High cost if prediction fails

            # Use switching rule to select controller
            best_controller_name = switcher.select_controller(j_predictions)
            switcher.update_state(best_controller_name, j_predictions)

        # Apply control from selected controller
        selected_controller = controllers[best_controller_name]
        action = selected_controller.update_control(state)
        state = env.step_sim(action)

        # Record history
        history['time'].append(step * DT)
        history['x'].append(state[0])
        history['dx'].append(state[1])
        history['angle'].append(state[2])
        history['da'].append(state[3])
        history['active_controller'].append(controller_names.index(best_controller_name))
        history['j_lqr'].append(j_predictions.get('LQR', 0))
        history['j_sm'].append(j_predictions.get('SM', 0))
        history['j_vf'].append(j_predictions.get('VF', 0))
        history['action'].append(np.clip(action, -env.f_max, env.f_max))

        # Print progress
        if step % 1000 == 0:
            print(f"Step {step}/{num_steps} - Active: {best_controller_name} - Angle: {state[2]:.3f}")

        # Early termination if pendulum falls
        if abs(state[2]) > 0.6:  # More than 34 degrees
            print(f"Pendulum fell at step {step}. Stopping simulation.")
            break

    return history


if __name__ == '__main__':
    if not os.path.isdir(MODEL_DIR):
        print(f"Error: The specified model directory does not exist: {MODEL_DIR}")
    else:
        env = InvePendulum(dt=DT, soft_wall=True)
        # env.x_max = 5
        controllers = {"LQR": LQR(-2.91, -3.67, -25.43, -4.94), "SM": SlidingMode(env), "VF": LQR(0, -33.90, -153.30, -32.07)}

        simulation_history = run_evaluation(env, controllers, model_dir=MODEL_DIR, initial_state=START_CONDITION)

        if simulation_history:
            plotter = SimulationPlotter(list(controllers.keys()))
            plotter.plot(simulation_history)

            # Print final statistics
            print("\n=== Simulation Statistics ===")
            print(f"Duration: {simulation_history['time'][-1]:.2f} seconds")
            print(f"Final angle: {simulation_history['angle'][-1]:.3f} rad")
            print(f"Final position: {simulation_history['x'][-1]:.3f} m")

            # Count controller usage
            active_controllers = simulation_history['active_controller']
            controller_names = list(controllers.keys())
            for i, name in enumerate(controller_names):
                count = sum(1 for x in active_controllers if x == i)
                percentage = (count / len(active_controllers)) * 100
                print(f"{name} usage: {count} steps ({percentage:.1f}%)")
