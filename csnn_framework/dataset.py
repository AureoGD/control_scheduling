import os
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode

# --- Configuration ---
DT = 0.001
DATASET_DIR = "csnn_framework/datasets"

# Reproducibility
GLOBAL_SEED = 44
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

# Cost function parameters
DELTA = 0.5  # Discount factor

# PROPERLY SCALED Cost matrix - CRITICAL FOR MEANINGFUL COSTS
P_MATRIX = np.array(
    [
        [1.25, 0.0, 0.0, 0.0],  # x position
        [0.0, 1.0, 0.0, 0.0],  # x velocity
        [0.0, 0.0, 8.0, 0.0],  # theta angle (PRIORITY - pendulum upright)
        [0.0, 0.0, 0.0, 1.0]  # theta velocity
    ],
    dtype=np.float32)

# Data Generation
NUM_TRAJECTORIES = 5000
MAX_SIMULATION_TIME = 4.0  # seconds
MAX_SIMULATION_STEPS = int(MAX_SIMULATION_TIME / (10 * DT))
RECORD_EVERY_N_STEPS = 5  # Downsample recording

# System constraints (for realistic sampling)
X_MAX = 2.0
X_DOT_MAX = 3.0
THETA_MAX = 0.5  # ~28.6 degrees
THETA_DOT_MAX = 8.0


def calculate_quadratic_cost(state, p_matrix):
    return state @ p_matrix @ state


def sample_initial_conditions(n):
    inits = []
    for _ in range(n):
        x = random.uniform(-X_MAX * 0.9, X_MAX * 0.9)
        dx = random.uniform(-X_DOT_MAX * 0.8, X_DOT_MAX * 0.8)
        theta = random.uniform(-THETA_MAX, THETA_MAX)
        dtheta = random.uniform(-THETA_DOT_MAX, THETA_DOT_MAX)
        inits.append(np.array([x, dx, theta, dtheta], dtype=np.float32))
    return inits


def ensure_numpy_array(state):
    if isinstance(state, tuple):
        return np.array(state, dtype=np.float32)
    elif isinstance(state, list):
        return np.array(state, dtype=np.float32)
    elif isinstance(state, np.ndarray):
        return state.astype(np.float32)
    else:
        raise ValueError(f"Unknown state type: {type(state)}")


def simulate_trajectory(env, controller, initial_state, max_steps):
    # Reset environment and ensure state is numpy array
    state_obj = env.reset(initial_state=initial_state.copy())
    state = ensure_numpy_array(state_obj)

    recorded_states = []
    recorded_costs = []
    is_stable = True

    for step in range(max_steps):
        # Get control action
        action = controller.update_control(state)

        # Record state and cost periodically
        if step % RECORD_EVERY_N_STEPS == 0:
            recorded_states.append(state.copy())
            cost = calculate_quadratic_cost(state, P_MATRIX)
            recorded_costs.append(cost)

        # Step simulation and ensure result is numpy array
        next_state_obj = env.step_sim(action)
        state = ensure_numpy_array(next_state_obj)

        # Check for failure (pendulum fell)
        if abs(state[2]) > THETA_MAX * 1.1:
            is_stable = False
            break

    return recorded_states, recorded_costs, is_stable


def calculate_cost_to_go(immediate_costs, delta):
    if not immediate_costs:
        return np.array([])

    costs_to_go = np.zeros_like(immediate_costs)
    costs_to_go[-1] = immediate_costs[-1]

    # Backward pass for Bellman equation
    for i in reversed(range(len(immediate_costs) - 1)):
        costs_to_go[i] = immediate_costs[i] + delta * costs_to_go[i + 1]

    return costs_to_go


def analyze_controller_dataset(df, controller_name):

    print(f"\n=== {controller_name} DATASET ANALYSIS ===")
    print(f"Total samples: {len(df):,}")
    print(f"Stable trajectories: {df['stable'].sum():,}")
    print(f"Unstable trajectories: {len(df) - df['stable'].sum():,}")

    costs = df['cost_to_go']
    print(f"Cost statistics:")
    print(f"  Min: {costs.min():.3f}, Max: {costs.max():.3f}")
    print(f"  Mean: {costs.mean():.3f}, Std: {costs.std():.3f}")
    print(f"  Median: {costs.median():.3f}")

    # Analyze different regions
    regions = {
        'Near origin (|x|<0.1, |θ|<0.1)': (df['x_k'].abs() < 0.1) & (df['a_k'].abs() < 0.1),
        'Moderate (|x|<0.5, |θ|<0.25)': (df['x_k'].abs() < 0.5) & (df['a_k'].abs() < 0.25),
        'Large angle (|θ|>0.3)': (df['a_k'].abs() > 0.3),
        'Large position (|x|>1.0)': (df['x_k'].abs() > 1.0)
    }

    for region_name, mask in regions.items():
        region_data = df[mask]
        if len(region_data) > 0:
            avg_cost = region_data['cost_to_go'].mean()
            print(f"{region_name}: {len(region_data):,} samples, avg cost = {avg_cost:.3f}")


def generate_controller_dataset(controller, controller_name, env, initial_states):
    """
    Generate dataset for a single controller.
    """
    print(f"\n--- Generating {controller_name} Dataset ---")
    all_data = []
    stable_count = 0
    unstable_count = 0

    pbar = tqdm(initial_states, desc=f"{controller_name} trajectories")

    for traj_idx, initial_state in enumerate(pbar):
        # Simulate trajectory
        states, immediate_costs, is_stable = simulate_trajectory(env, controller, initial_state, MAX_SIMULATION_STEPS)

        if is_stable:
            stable_count += 1
        else:
            unstable_count += 1

        # Calculate cost-to-go if we have data
        if states and immediate_costs:
            costs_to_go = calculate_cost_to_go(immediate_costs, DELTA)

            # Store all data points
            for i, state in enumerate(states):
                all_data.append({'traj_idx': traj_idx, 'x_k': state[0], 'dx_k': state[1], 'a_k': state[2], 'da_k': state[3], 'cost_to_go': costs_to_go[i], 'stable': is_stable})

        # Update progress
        pbar.set_postfix({'stable': stable_count, 'unstable': unstable_count, 'points': len(all_data)})

    pbar.close()

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Analysis
    analyze_controller_dataset(df, controller_name)

    return df


def compare_controllers_datasets(controller_dfs):
    """
    Compare cost distributions across all controllers.
    """
    print(f"\n{'='*60}")
    print("CONTROLLER COMPARISON ANALYSIS")
    print(f"{'='*60}")

    # Compare near origin performance (most important for switching)
    print("\n--- Near Origin Performance (|x| < 0.1, |θ| < 0.1) ---")
    for name, df in controller_dfs.items():
        near_origin_mask = ((np.abs(df['x_k']) < 0.1) & (np.abs(df['a_k']) < 0.1))
        region_data = df[near_origin_mask]
        if len(region_data) > 0:
            avg_cost = region_data['cost_to_go'].mean()
            print(f"{name}: {avg_cost:.3f} (n={len(region_data)})")
        else:
            print(f"{name}: No data near origin")


def main():
    """Main dataset generation routine."""
    print("=== Neural Network Training Dataset Generation ===")
    print(f"Target directory: {DATASET_DIR}")
    print(f"Number of trajectories: {NUM_TRAJECTORIES}")
    print(f"Cost matrix:\n{P_MATRIX}")

    # Create output directory
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Sample initial conditions
    print("\nSampling initial conditions...")
    initial_states = sample_initial_conditions(NUM_TRAJECTORIES)

    # Save initial conditions for reproducibility
    initials_df = pd.DataFrame(initial_states, columns=['x0', 'dx0', 'theta0', 'dtheta0'])
    initials_path = os.path.join(DATASET_DIR, "initial_conditions.csv")
    initials_df.to_csv(initials_path, index=False)
    print(f"Saved initial conditions to {initials_path}")

    # Initialize environment and controllers
    env = InvePendulum(dt=DT)
    controllers = {
        "LQR": LQR(-2.91, -3.67, -25.43, -4.94),
        "SM": SlidingMode(env),
        "VF": LQR(0, -33.90, -153.30, -32.07),
    }

    # Generate datasets for each controller
    controller_dfs = {}
    for name, controller in controllers.items():
        df = generate_controller_dataset(controller, name, env, initial_states)

        # Save dataset
        dataset_path = os.path.join(DATASET_DIR, f"{name.lower()}.csv")
        df.to_csv(dataset_path, index=False)
        print(f"Saved {name} dataset to {dataset_path}")

        controller_dfs[name] = df

    # Comparative analysis
    compare_controllers_datasets(controller_dfs)

    print(f"\n=== Dataset generation completed ===")
    print(f"All datasets saved to: {DATASET_DIR}")


if __name__ == "__main__":
    main()
