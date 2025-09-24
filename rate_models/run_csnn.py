import os
import csv
import torch
import numpy as np
import joblib
from tqdm import tqdm
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode
from csnn_framework.nn_model import NNModel
from csnn_framework.switch_rule import SwitchRule

# --- Configuration ---
MODEL_DIR = "models/csnn/31174103"  # Change to your CSNN model directory
MODEL_NAME = "csnn"
EXP_ID = "exp1"

# Evaluation parameters
IS_DISCRETE = True
RENDERING = False
NOISE = False
DISTURBANCE = False
NUM_SAMPLES = 350
SIM_TIME = 5
DT = 0.001
SAMPLING_INTERVAL = 1
MAX_STEP = int(SIM_TIME / (10*DT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "eval_data")
IC_PATH = os.path.join(DATA_DIR, "initial_conditions.csv")
PREFIX = "d" if IS_DISCRETE else "c"
OUT_FILE = f"{PREFIX}_{MODEL_NAME}_data_{EXP_ID}.csv"
OUT_PATH = os.path.join(DATA_DIR, OUT_FILE)

def read_initial_conditions(path):
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def load_csnn_model_and_scalers(controller_name, model_dir, device):
    """Load CSNN model and corresponding scalers for a controller."""
    model_path = os.path.join(model_dir, f"{controller_name.lower()}_model.pth")
    scalers_path = os.path.join(model_dir, f"{controller_name.lower()}_scalers.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        raise FileNotFoundError(f"Model or scalers not found for {controller_name}")

    # Load model
    model = NNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load scalers
    scalers = joblib.load(scalers_path)

    return model, scalers

def predict_cost_to_go(model, scalers, state, device):
    """Predict cost-to-go for a state using normalized model."""
    if isinstance(state, torch.Tensor):
        state = state.cpu().numpy()

    # Normalize state
    state_norm = scalers['state_scaler'].transform(state.reshape(1, -1))
    state_norm_tensor = torch.tensor(state_norm, dtype=torch.float32).to(device)

    # Predict (normalized output)
    with torch.no_grad():
        pred_norm = model(state_norm_tensor).cpu().numpy()

    # Denormalize prediction
    pred = scalers['target_scaler'].inverse_transform(pred_norm.reshape(-1, 1))

    return pred.item()

# Load initial conditions
X0 = read_initial_conditions(IC_PATH)
if len(X0) > NUM_SAMPLES:
    X0 = X0[:NUM_SAMPLES]  # Use only the first NUM_SAMPLES

# Initialize environment and controllers
env = InvePendulum(dt=DT, soft_wall=True)
controllers = {
    "LQR": LQR(-2.91, -3.67, -25.43, -4.94), 
    "SM": SlidingMode(env), 
    "VF": LQR(0, -33.90, -153.30, -32.07)
}
controller_names = list(controllers.keys())

# Load CSNN models and scalers
csnn_models = {}
csnn_scalers = {}
for name in controller_names:
    try:
        model, scalers = load_csnn_model_and_scalers(name, MODEL_DIR, DEVICE)
        csnn_models[name] = model
        csnn_scalers[name] = scalers
        print(f"✓ Loaded {name} model and scalers")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        exit(1)

# Initialize switching rule
switcher = SwitchRule(controller_names)

# Run evaluation
with open(OUT_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "ic_id", "step", "time", "x", "xdot", "theta", "thetadot", "mode", 
        "control_effort", "reward_step", "total_reward", "terminated", "truncated"
    ])

    for ic_id, x0 in enumerate(tqdm(X0, desc="Initial conditions")):
        # Reset environment with initial condition
        state = env.reset(initial_state=x0)
        
        # Initialize tracking variables
        done = False
        step = 0
        total_reward = 0.0
        best_controller_name = controller_names[0]
        j_predictions = {name: 0 for name in controller_names}

        while not done and step < MAX_STEP:
            # Predict costs and select controller at sampling interval
            if step % SAMPLING_INTERVAL == 0:
                current_state = np.array([state])
                
                # Get predictions from all CSNN models
                for name in controller_names:
                    try:
                        j_predictions[name] = predict_cost_to_go(
                            csnn_models[name], csnn_scalers[name], current_state, DEVICE
                        )
                    except Exception as e:
                        print(f"Error predicting for {name}: {e}")
                        j_predictions[name] = float('inf')

                # Use switching rule to select controller
                best_controller_name = switcher.select_controller(j_predictions)
                switcher.update_state(best_controller_name, j_predictions)

            # Apply control from selected controller
            selected_controller = controllers[best_controller_name]
            action = selected_controller.update_control(state)
            
            # Step the environment
            state_next = env.step_sim(action)
            
            # Use 0 for reward as requested
            reward = 0.0
            terminated = False
            truncated = (step >= MAX_STEP - 1)
            
            # Check termination conditions
            if abs(state[2]) >= np.pi / 2:  # Pendulum fell
                terminated = True
                done = True
            
            if truncated:
                done = True

            # Extract state components
            x, xdot, theta, thetadot = state
            control_effort = np.clip(action, -env.f_max, env.f_max)
            total_reward += reward
            
            # Map controller name to mode index
            mode = controller_names.index(best_controller_name) if best_controller_name in controller_names else -1

            # Write data (same format as first script)
            w.writerow([
                ic_id, step, step * DT, 
                float(x), float(xdot), float(theta), float(thetadot),
                mode, float(control_effort), float(reward), float(total_reward),
                int(terminated), int(truncated)
            ])

            state = state_next
            step += 1

print(f"CSNN evaluation finished. Data saved to: {OUT_PATH}")
print(f"Evaluated {len(X0)} initial conditions")