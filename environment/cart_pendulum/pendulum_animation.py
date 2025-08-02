import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from math import pi
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
import numpy as np


class PendulumLiveRenderer:
    """
    Handles only the real-time animation visualization for an Inverted Pendulum's current state.
    It does not store or animate historical data.
    """

    def __init__(self, pendulum: InvePendulum):
        self.pendulum = pendulum  # Store reference to the pendulum object

        self.anim_fig = None
        self.anim_ax = None
        self.cart_patch = None
        self.pendulum_line = None

    def init_live_render(self):
        """Initializes the figure for real-time animation updates."""
        # Close any existing figure before creating a new one
        self.close_render()

        self.anim_fig, self.anim_ax = plt.subplots(figsize=(8, 6))
        self.anim_ax.set_xlim(-self.pendulum.x_max - 0.5, self.pendulum.x_max + 0.5)
        self.anim_ax.set_ylim(-self.pendulum.l - 0.5, self.pendulum.l + 0.5)
        self.anim_ax.set_aspect('equal')
        self.anim_ax.set_title(f"Inverted Pendulum Live Animation")
        self.anim_ax.axhline(0, color='k', linewidth=2)  # Ground line

        cart_width = 0.3
        cart_height = 0.2
        self.cart_patch = Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black')
        self.anim_ax.add_patch(self.cart_patch)

        self.pendulum_line, = self.anim_ax.plot([], [], lw=4, color='red')
        plt.show(block=False)  # Show non-blocking

    def update_live_render(self):
        """Updates the live animation frame based on the pendulum's current state."""
        if self.anim_fig and self.anim_ax and self.cart_patch and self.pendulum_line:
            current_state = self.pendulum.get_state()  # Directly use the current state
            cart_x = current_state[0]
            pendulum_angle = current_state[2]

            cart_width = 0.3
            cart_height = 0.2
            # Update cart position
            self.cart_patch.set_xy((cart_x - cart_width / 2, -cart_height / 2))

            # Update pendulum rod position
            pivot_y = 0  # Pivot is at y=0 (on the cart)
            px = cart_x + self.pendulum.l * np.sin(pendulum_angle)
            py = pivot_y + self.pendulum.l * np.cos(pendulum_angle)

            self.pendulum_line.set_data([cart_x, px], [pivot_y, py])

            # Draw and flush events to update the plot immediately
            self.anim_fig.canvas.draw_idle()
            self.anim_fig.canvas.flush_events()
        else:
            print("PendulumLiveRenderer: Renderer not initialized. Call init_live_render() first.")

    def close_render(self):
        """Closes the matplotlib figure associated with this renderer instance."""
        if self.anim_fig is not None:
            plt.close(self.anim_fig)
            self.anim_fig = None
            self.anim_ax = None
            self.cart_patch = None
            self.pendulum_line = None
