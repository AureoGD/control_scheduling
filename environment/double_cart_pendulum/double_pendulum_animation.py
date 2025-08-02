import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from environment.double_cart_pendulum.double_cart_pendulum_dynamics import DoubleCartPendulum


class DoublePendulumLiveRenderer:
    """
    Handles the real-time animation for a Double Inverted Pendulum's current state.
    It visualizes the cart and both pendulum links.
    """

    def __init__(self, pendulum: DoubleCartPendulum):
        """
        Initializes the renderer with a DoubleCartPendulum instance.

        Args:
            pendulum (DoubleCartPendulum): The pendulum environment to render.
        """
        self.pendulum = pendulum  # Store reference to the double pendulum object

        self.anim_fig = None
        self.anim_ax = None
        self.cart_patch = None
        self.pendulum_line1 = None
        self.pendulum_line2 = None

    def init_live_render(self):
        """Initializes the Matplotlib figure for real-time animation."""
        # Close any existing figure before creating a new one
        self.close_render()

        self.anim_fig, self.anim_ax = plt.subplots(figsize=(8, 6))

        # Adjust plot limits to see both pendulums fully extended
        total_length = self.pendulum.l1 + self.pendulum.l2
        self.anim_ax.set_xlim(-self.pendulum.x_max - 0.5, self.pendulum.x_max + 0.5)
        self.anim_ax.set_ylim(-total_length - 0.5, total_length + 0.5)

        self.anim_ax.set_aspect('equal')
        self.anim_ax.set_title("Double Inverted Pendulum Live Animation")
        self.anim_ax.axhline(0, color='k', linewidth=2)  # Ground line

        # Create cart patch
        cart_width = 0.3
        cart_height = 0.2
        self.cart_patch = Rectangle((0, 0), cart_width, cart_height, fc='blue', ec='black', zorder=3)
        self.anim_ax.add_patch(self.cart_patch)

        # Create lines for both pendulums
        self.pendulum_line1, = self.anim_ax.plot([], [], lw=4, color='red', zorder=2)
        self.pendulum_line2, = self.anim_ax.plot([], [], lw=4, color='purple', zorder=1)

        plt.show(block=False)  # Show the plot in non-blocking mode

    def update_live_render(self):
        """Updates the live animation frame based on the pendulum's current state."""
        if not all([self.anim_fig, self.anim_ax, self.cart_patch, self.pendulum_line1, self.pendulum_line2]):
            print("DoublePendulumLiveRenderer: Renderer not initialized. Call init_live_render() first.")
            return

        # Get the full 6-variable state: (x, dx, a1, da1, a2, da2)
        current_state = self.pendulum.get_state()
        cart_x = current_state[0]
        a1 = current_state[2]
        a2 = current_state[4]

        # --- Update Cart ---
        cart_width = 0.3
        cart_height = 0.2
        self.cart_patch.set_xy((cart_x - cart_width / 2, -cart_height / 2))

        # --- Update First Pendulum (attached to cart) ---
        pivot1_x, pivot1_y = cart_x, 0
        end1_x = pivot1_x + self.pendulum.l1 * np.sin(a1)
        end1_y = pivot1_y + self.pendulum.l1 * np.cos(a1)
        self.pendulum_line1.set_data([pivot1_x, end1_x], [pivot1_y, end1_y])

        # --- Update Second Pendulum (attached to the end of the first) ---
        pivot2_x, pivot2_y = end1_x, end1_y  # The pivot is the end of the first link
        end2_x = pivot2_x + self.pendulum.l2 * np.sin(a2)
        end2_y = pivot2_y + self.pendulum.l2 * np.cos(a2)
        self.pendulum_line2.set_data([pivot2_x, end2_x], [pivot2_y, end2_y])

        # Draw and flush events to update the plot immediately
        self.anim_fig.canvas.draw_idle()
        self.anim_fig.canvas.flush_events()

    def close_render(self):
        """Closes the Matplotlib figure associated with this renderer."""
        if self.anim_fig is not None:
            plt.close(self.anim_fig)
            self.anim_fig = None
            self.anim_ax = None
            self.cart_patch = None
            self.pendulum_line1 = None
            self.pendulum_line2 = None
