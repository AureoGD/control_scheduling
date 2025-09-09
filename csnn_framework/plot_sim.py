import matplotlib.pyplot as plt


class SimulationPlotter:
    """A dedicated class for plotting simulation results."""

    def __init__(self, controller_names):
        """
        Initializes the plotter.
        Args:
            controller_names (list): A list of strings with the names of the controllers.
        """
        self.controller_names = controller_names
        self.controller_colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # Green, Blue, Orange

    def plot(self, history, title='Final Performance after Online Learning & Switching'):
        """
        Generates and displays the simulation plots.
        Args:
            history (dict): The dictionary containing simulation data.
            title (str): The main title for the plot figure.
        """
        fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(title, fontsize=16)

        # --- Plot 1: Angle and Active Controller with Colored Background ---
        axs[0].plot(history['time'], history['angle'], color='k', label='Pendulum Angle')
        axs[0].set_ylabel('Angle (rad)')
        axs[0].set_title('Pendulum Angle and Active Controller')
        axs[0].grid(alpha=0.3)

        start_time = history['time'][0]
        current_controller_idx = history['active_controller'][0]
        labels_added = set()
        for i in range(1, len(history['time'])):
            if history['active_controller'][i] != current_controller_idx:
                label = self.controller_names[current_controller_idx] if current_controller_idx not in labels_added else ""
                axs[0].axvspan(start_time, history['time'][i], facecolor=self.controller_colors[current_controller_idx], alpha=0.2, label=label)
                labels_added.add(current_controller_idx)
                start_time = history['time'][i]
                current_controller_idx = history['active_controller'][i]
        label = self.controller_names[current_controller_idx] if current_controller_idx not in labels_added else ""
        axs[0].axvspan(start_time, history['time'][-1], facecolor=self.controller_colors[current_controller_idx], alpha=0.2, label=label)
        axs[0].legend(loc='upper right')

        # --- Plot 2: Cart Position ---
        axs[1].plot(history['time'], history['x'], color='k', label='Cart Position')
        axs[1].set_ylabel('Position (m)')
        axs[1].set_title('Cart Position')
        axs[1].grid(alpha=0.3)

        # --- Plot 3: Performance Index (J) Estimates ---
        axs[2].plot(history['time'], history['j_lqr'], color=self.controller_colors[0], label=f'J_hat {self.controller_names[0]}')
        axs[2].plot(history['time'], history['j_sm'], color=self.controller_colors[1], label=f'J_hat {self.controller_names[1]}')
        axs[2].plot(history['time'], history['j_vf'], color=self.controller_colors[2], label=f'J_hat {self.controller_names[2]}')
        #axs[2].plot(history['time'], history['j_true'], color=self.controller_colors[2], label=f'J_true')
        axs[2].set_ylabel('Cost')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_title('Evolution of Performance Index Estimates')
        axs[2].grid(alpha=0.3)
        axs[2].legend()
        axs[2].set_ylim(bottom=0)

        axs[3].plot(history['time'], history['action'], color='k', label='Control effort')
        axs[3].set_ylabel('Control effort (N)')
        axs[3].set_title('Control effort')
        axs[3].grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
