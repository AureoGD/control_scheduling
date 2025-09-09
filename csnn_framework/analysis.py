import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import the necessary directory paths from your config file
DATASET_DIR = "csnn_framework/datasets"
SAVE_DATA_DIR = "csnn_framework/analysis"
DEVICE = 'cuda'
DT = 0.001
SAMPLING_INTERVAL = 1


def plot_initial_states(dataset_dir, controller_names, output_dir, suffix='_offline'):
    """
    Loads the generated datasets and creates a separate scatter plot subplot
    for the initial (x, a) positions of each controller's trajectories.
    """
    print("Plotting initial state distribution in separate subplots...")

    num_controllers = len(controller_names)
    fig, axes = plt.subplots(1, num_controllers, figsize=(6 * num_controllers, 5), squeeze=False)

    for i, name in enumerate(controller_names):
        ax = axes[0, i]
        # --- UPDATED: Look for the correct processed file name ---
        filepath = os.path.join(dataset_dir, f"{name.lower()}.csv")

        if not os.path.exists(filepath):
            print(f"Warning: Dataset for {name} not found at {filepath}. Skipping plot.")
            ax.set_title(f"{name}\n(Dataset not found)")
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        print(f"Loading data for {name}...")
        df = pd.read_csv(filepath)

        # Find the first point of each trajectory
        initial_points_df = df.drop_duplicates(subset=['traj_idx'], keep='first')

        x_initial = initial_points_df['x_k']
        a_initial = initial_points_df['a_k']

        ax.scatter(x_initial, a_initial, alpha=0.7, s=15, edgecolors='k', linewidth=0.5)
        ax.set_title(f'Initial States for {name}', fontsize=14)
        ax.set_xlabel('Cart Position (x) [m]', fontsize=10)
        ax.set_ylabel('Pendulum Angle (a) [rad]', fontsize=10)  # Corrected label
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.suptitle('Distribution of Initial States per Controller', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "initial_states_distribution_subplots.png")
    plt.savefig(save_path)
    print(f"\nSaved initial states plot to {save_path}")
    plt.close(fig)


def plot_trajectory_analysis(dataset_dir, controller_names, output_dir, num_trajectories=10, suffix='_offline'):
    """
    Loads pre-processed datasets and plots two rows of analysis:
    1. Phase Diagram (Cart Position vs. Pendulum Angle)
    2. Pre-calculated Cost-to-Go vs. Time
    """
    print("\nPlotting trajectory analysis (Phase & Cost)...")

    num_controllers = len(controller_names)
    fig, axes = plt.subplots(2, num_controllers, figsize=(7 * num_controllers, 10), squeeze=False)

    for i, name in enumerate(controller_names):
        phase_ax = axes[0, i]
        cost_ax = axes[1, i]
        # --- UPDATED: Look for the correct processed file name ---
        filepath = os.path.join(dataset_dir, f"{name.lower()}.csv")

        if not os.path.exists(filepath):
            print(f"Warning: Dataset for {name} not found at {filepath}. Skipping plot.")
            phase_ax.set_title(f"{name}\n(Dataset not found)")
            phase_ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=phase_ax.transAxes)
            cost_ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=cost_ax.transAxes)
            continue

        print(f"Loading data for {name} analysis...")
        df = pd.read_csv(filepath)

        all_traj_ids = df['traj_idx'].unique()
        num_to_select = min(num_trajectories, len(all_traj_ids))
        selected_ids = np.random.choice(all_traj_ids, size=num_to_select, replace=False)

        colors = plt.cm.get_cmap('plasma', num_to_select)

        for j, traj_id in enumerate(selected_ids):
            traj_df = df[df['traj_idx'] == traj_id]

            # --- 1. Plot Phase Diagram (Cart Position vs. Angle) ---
            cart_pos = traj_df['x_k']
            angular_pos = traj_df['a_k']

            phase_ax.plot(cart_pos, angular_pos, color=colors(j), alpha=0.8)
            phase_ax.plot(cart_pos.iloc[0], angular_pos.iloc[0], 'o', color='limegreen', markersize=6, markeredgecolor='k', label='Start' if j == 0 else "")
            phase_ax.plot(cart_pos.iloc[-1], angular_pos.iloc[-1], '.', color='red', markersize=8, label='End' if j == 0 else "")

            # --- 2. Plot Pre-calculated Cost-to-Go vs. Time ---
            cost_values = traj_df['cost_to_go']
            time_steps = np.arange(len(traj_df)) * (DT * SAMPLING_INTERVAL)

            cost_ax.plot(time_steps, cost_values, color=colors(j), alpha=0.8)

        # --- Set titles and labels for the plots ---
        phase_ax.set_title(f'State Evolution for {name}', fontsize=14)
        phase_ax.set_xlabel('Cart Position (x) [m]', fontsize=10)
        phase_ax.set_ylabel('Pendulum Angle (a) [rad]', fontsize=10)
        phase_ax.grid(True, linestyle='--')
        if i == 0: phase_ax.legend()

        cost_ax.set_title(f'Cost-to-Go Evolution for {name}', fontsize=14)
        cost_ax.set_xlabel('Time (s)', fontsize=10)
        cost_ax.set_ylabel('Discounted Cost-to-Go', fontsize=10)
        cost_ax.grid(True, linestyle='--')

    fig.suptitle(f'Trajectory Analysis ({num_trajectories} Random Trajectories per Controller)', fontsize=18, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_path = os.path.join(output_dir, "trajectory_analysis.png")
    plt.savefig(save_path)
    print(f"\nSaved trajectory analysis plot to {save_path}")
    plt.show()


if __name__ == '__main__':
    # This list should match the controllers you trained
    controllers_to_plot = ["LQR", "SM", "VF"]

    # The output directory will be the base model directory from your config
    output_directory = SAVE_DATA_DIR

    # Create the plots
    plot_initial_states(DATASET_DIR, controllers_to_plot, output_directory)
    plot_trajectory_analysis(DATASET_DIR, controllers_to_plot, output_directory, num_trajectories=10)
