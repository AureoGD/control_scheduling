import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for white background with grey grid
plt.style.use('default')
sns.set_style("whitegrid")
ALG = 'cem'
# Load the CSV file
file_path = f'rate_models/eval_data/d_{ALG}_data_exp1.csv'
df = pd.read_csv(file_path)

# Filter data for ic_id = 5
df_ic0 = df[df['ic_id'] == 0].copy()

# Check if data exists for ic_id = 5
if df_ic0.empty:
    print("No data found for ic_id = 5")
    print(f"Available ic_id values: {df['ic_id'].unique()}")
else:
    # Create 5 subplots with shared x-axis
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    
    # Set background color to white
    fig.patch.set_facecolor('white')
    
    # Use same black color for all lines
    line_color = 'black'
    
    # Limit time to 0-5 seconds (multiply by 10)
    time_limit = 5.0
    df_filtered = df_ic0[df_ic0['time'] * 10 <= time_limit].copy()
    
    # Reset index to avoid issues with original indices
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Define mode colors (light colors for background)
    mode_colors = {
        0: 'lightblue',
        1: 'lightgreen', 
        2: 'peachpuff',
        3: 'lightcoral',
        4: 'lavender',
    }
    
    # Get unique modes in the filtered data
    df_filtered['scaled_time'] = df_filtered['time'] * 10
    modes = df_filtered['mode'].unique()
    modes.sort()
    
    # print(f"Available modes in ic_id=5: {modes}")
    # print(f"Mode value counts:\n{df_filtered['mode'].value_counts().sort_index()}")
    # print(f"Total data points in filtered data: {len(df_filtered)}")
    
    # Improved mode transition detection using reset indices
    mode_changes = []
    current_mode = None
    
    for i in range(len(df_filtered)):
        if current_mode is None:
            current_mode = df_filtered.iloc[i]['mode']
            mode_changes.append(i)  # Start with first index
        elif df_filtered.iloc[i]['mode'] != current_mode:
            current_mode = df_filtered.iloc[i]['mode']
            mode_changes.append(i)  # Mode change at current index
    
    # Add the last index to mark the end of the final mode
    if mode_changes[-1] != len(df_filtered) - 1:
        mode_changes.append(len(df_filtered) - 1)
    
    # print(f"Mode changes detected at indices: {mode_changes}")
    
    # Add colored backgrounds for each mode region
    for ax in axes:
        for i in range(len(mode_changes) - 1):
            start_idx = mode_changes[i]
            end_idx = mode_changes[i + 1]
            
            mode_val = df_filtered.iloc[start_idx]['mode']
            start_time = df_filtered.iloc[start_idx]['scaled_time']
            end_time = df_filtered.iloc[end_idx]['scaled_time']
            
            if mode_val in mode_colors:
                ax.axvspan(start_time, end_time, alpha=0.3, color=mode_colors[mode_val])
                # print(f"Mode {mode_val}: {start_time:.2f}s to {end_time:.2f}s (duration: {end_time-start_time:.2f}s)")
    
    # Plot 1: Time vs x
    axes[0].set_facecolor('white')
    axes[0].plot(df_filtered['scaled_time'], df_filtered['x'], linewidth=1.5, color=line_color)
    axes[0].set_ylabel(r'$x$ (m)', fontsize=12)
    axes[0].grid(True, color='lightgray', alpha=0.7)

    # Plot 2: Time vs x_dot
    axes[1].set_facecolor('white')
    axes[1].plot(df_filtered['scaled_time'], df_filtered['xdot'], linewidth=1.5, color=line_color)
    axes[1].set_ylabel(r'$\dot{x}$ (m/s)', fontsize=12)
    axes[1].grid(True, color='lightgray', alpha=0.7)

    # Plot 3: Time vs theta
    axes[2].set_facecolor('white')
    axes[2].plot(df_filtered['scaled_time'], df_filtered['theta'], linewidth=1.5, color=line_color)
    axes[2].set_ylabel(r'$\theta$ (rad)', fontsize=12)
    axes[2].grid(True, color='lightgray', alpha=0.7)

    # Plot 4: Time vs theta_dot
    axes[3].set_facecolor('white')
    axes[3].plot(df_filtered['scaled_time'], df_filtered['thetadot'], linewidth=1.5, color=line_color)
    axes[3].set_ylabel(r'$\dot{\theta}$ (rad/s)', fontsize=12)
    axes[3].grid(True, color='lightgray', alpha=0.7)

    # Plot 5: Time vs control_effort (now F in Newtons)
    axes[4].set_facecolor('white')
    axes[4].plot(df_filtered['scaled_time'], df_filtered['control_effort'], linewidth=1.5, color=line_color)
    axes[4].set_xlabel('Time (s)', fontsize=12)
    axes[4].set_ylabel(r'$F$ (N)', fontsize=12)
    axes[4].grid(True, color='lightgray', alpha=0.7)
    
    # Set x-axis limits to 0-5 seconds
    for ax in axes:
        ax.set_xlim(0, time_limit)

    # Create mode legend only if there are multiple modes
    if len(modes) > 1:
        legend_elements = [plt.Rectangle((0,0), 1, 1, alpha=0.3, color=mode_colors[mode], 
                                       label=f'Mode {mode}') for mode in sorted(modes) if mode in mode_colors]
        axes[0].legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure (optional)
    plt.savefig(f'{ALG}.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

    # # Display summary statistics for the filtered data
    # print("\nSummary Statistics for ic_id = 5 (0-5 seconds):")
    # print("=" * 50)
    # print(f"Time steps: {len(df_filtered)}")
    # print(f"Final x: {df_filtered['x'].iloc[-1]:.4f} m")
    # print(f"Final $\dot{{x}}$: {df_filtered['xdot'].iloc[-1]:.4f} m/s")
    # print(f"Final $\theta$: {df_filtered['theta'].iloc[-1]:.4f} rad")
    # print(f"Final $\dot{{\theta}}$: {df_filtered['thetadot'].iloc[-1]:.4f} rad/s")
    # print(f"Max control force: {df_filtered['control_effort'].abs().max():.4f} N")