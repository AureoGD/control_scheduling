from math import cos, sin, pi
import numpy as np


class DoubleCartPendulum():
    """
    A class to simulate a double inverted pendulum on a cart.

    This simulation is based on the Lagrangian dynamics of the system,
    resulting in a set of coupled second-order differential equations.
    The second pendulum is attached to the end of the first pendulum.

    Attributes:
        dt (float): The time step for the simulation in seconds.
        mc (float): Mass of the cart (base).
        mr1 (float): Mass of the first pendulum bar.
        l1 (float): Length of the first pendulum bar.
        mr2 (float): Mass of the second pendulum bar.
        l2 (float): Length of the second pendulum bar.
        br (float): Friction coefficient for the cart.
        g (float): Acceleration due to gravity.
        f_max (float): Maximum applicable force.
        v_max (float): Maximum cart velocity.
        x_max (float): Maximum cart position from the center.
        x (float): Current position of the cart.
        dx (float): Current velocity of the cart.
        a1 (float): Current angle of the first pendulum in radians.
        da1 (float): Current angular velocity of the first pendulum.
        a2 (float): Current angle of the second pendulum in radians.
        da2 (float): Current angular velocity of the second pendulum.
    """

    def __init__(self, dt=0.002):
        """
        Initializes the double inverted pendulum simulation environment.

        Args:
            dt (float, optional): The simulation time step. Defaults to 0.002.
        """
        # Parameters for the cart and first pendulum (from original example)
        self.mc = 0.6650  # Mass of the base (Kg)
        self.mr1 = 0.21  # Mass of the first pendulum (Kg)
        self.l1 = 0.61  # Length of the first pendulum (m)
        self.br = 0.2  # Friction coefficient of the base (Kg/s)
        self.g = 9.8  # Gravitational acceleration (m/s^2)

        # Parameters for the second pendulum (same characteristics as the first)
        self.mr2 = 0.21  # Mass of the second pendulum (Kg)
        self.l2 = 0.61  # Length of the second pendulum (m)

        # System constraints
        self.f_max = 10.0  # Max force (N) - Increased for the harder problem
        self.v_max = 3.0  # Max velocity (m/s)
        self.x_max = 2.0  # Max position (m)
        self.da_max = 15  # Max angular speed (rad/s) for both pendulums

        # Simulation time step
        self.dt = dt

        # Initialize state variables
        self.reset()

    def step_sim(self, force):
        """
        Advances the simulation by one time step using the new dynamics.
        """
        # Clamp the input force
        force = np.clip(force, -self.f_max, self.f_max)

        # Get current state for easier access
        x, dx, a1, da1, a2, da2 = self.get_state()

        # Pre-calculate trigonometric values for efficiency
        c1 = cos(a1)
        s1 = sin(a1)
        c2 = cos(a2)
        s2 = sin(a2)
        c12 = cos(a1 - a2)
        s12 = sin(a1 - a2)

        # --- Handle Boundary Conditions ---
        effective_force = force
        effective_dx = dx
        ddx = 0.0
        dda1 = 0.0
        dda2 = 0.0

        at_positive_limit = (x >= self.x_max and (dx > 0 or force > 0))
        at_negative_limit = (x <= -self.x_max and (dx < 0 or force < 0))

        if at_positive_limit or at_negative_limit:
            effective_dx = 0.0  # Cart velocity is zero at the wall
            ddx = 0.0  # Cart acceleration is zero at the wall

            # If at a boundary, solve a simpler 2x2 system for dda1, dda2
            # with ddx = 0.

            # A_sub matrix for the fixed-base double pendulum
            a22 = (self.mr1 / 3 + self.mr2) * self.l1**2
            a23 = 0.5 * self.mr2 * self.l1 * self.l2 * c12
            a32 = 0.5 * self.mr2 * self.l1 * self.l2 * c12
            a33 = (self.mr2 / 3) * self.l2**2
            A_sub = np.array([[a22, a23], [a32, a33]])

            # B_sub vector
            b2 = (0.5 * self.mr1 + self.mr2) * self.g * self.l1 * s1 - 0.5 * self.mr2 * self.l1 * self.l2 * s12 * da2**2
            b3 = 0.5 * self.mr2 * self.g * self.l2 * s2 + 0.5 * self.mr2 * self.l1 * self.l2 * s12 * da1**2
            B_sub = np.array([b2, b3])

            try:
                acc_sub = np.linalg.solve(A_sub, B_sub)
                dda1, dda2 = acc_sub[0], acc_sub[1]
            except np.linalg.LinAlgError:
                dda1, dda2 = 0.0, 0.0

        else:
            # --- Solve the full 3x3 system of linear equations ---
            # Define the A matrix (Inertia matrix)
            a11 = self.mc + self.mr1 + self.mr2
            a12 = (0.5 * self.mr1 + self.mr2) * self.l1 * c1
            a13 = 0.5 * self.mr2 * self.l2 * c2

            a21 = a12
            a22 = (self.mr1 / 3 + self.mr2) * self.l1**2
            a23 = 0.5 * self.mr2 * self.l1 * self.l2 * c12

            a31 = a13
            a32 = a23
            a33 = (self.mr2 / 3) * self.l2**2

            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

            # Define the B vector (Forces vector)
            b1 = (force - self.br * dx + (0.5 * self.mr1 + self.mr2) * self.l1 * s1 * da1**2 +
                  0.5 * self.mr2 * self.l2 * s2 * da2**2)
            b2 = ((0.5 * self.mr1 + self.mr2) * self.g * self.l1 * s1 -
                  0.5 * self.mr2 * self.l1 * self.l2 * s12 * da2**2)
            b3 = (0.5 * self.mr2 * self.g * self.l2 * s2 + 0.5 * self.mr2 * self.l1 * self.l2 * s12 * da1**2)

            B = np.array([b1, b2, b3])

            try:
                # Solve for accelerations [ddx, dda1, dda2]
                acc = np.linalg.solve(A, B)
                ddx, dda1, dda2 = acc[0], acc[1], acc[2]
            except np.linalg.LinAlgError:
                ddx, dda1, dda2 = 0.0, 0.0, 0.0

        # --- Update state using Euler integration ---
        self.dx = effective_dx + ddx * self.dt
        self.x += self.dx * self.dt

        self.da1 += dda1 * self.dt
        self.a1 += self.da1 * self.dt

        self.da2 += dda2 * self.dt
        self.a2 += self.da2 * self.dt

        # --- Enforce state constraints ---
        self.x = np.clip(self.x, -self.x_max, self.x_max)
        self.dx = np.clip(self.dx, -self.v_max, self.v_max)
        self.da1 = np.clip(self.da1, -self.da_max, self.da_max)
        self.da2 = np.clip(self.da2, -self.da_max, self.da_max)

        # Normalize angles to be within [-pi, pi] for stability
        self.a1 = (self.a1 + pi) % (2 * pi) - pi
        self.a2 = (self.a2 + pi) % (2 * pi) - pi

        # If position was clipped, velocity in that direction should be zero
        if (self.x == self.x_max and self.dx > 0) or \
           (self.x == -self.x_max and self.dx < 0):
            self.dx = 0.0

        return self.get_state()

    def reset(self, initial_state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)):
        """
        Resets the pendulum's state to a specified initial condition.

        Args:
            initial_state (tuple, optional): The initial state (x, dx, a1, da1, a2, da2).
                                             Defaults to all zeros.

        Returns:
            tuple: The initial state.
        """
        self.x, self.dx, self.a1, self.da1, self.a2, self.da2 = initial_state
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the system.

        Returns:
            tuple: A tuple containing the current state (x, dx, a1, da1, a2, da2).
        """
        return (self.x, self.dx, self.a1, self.da1, self.a2, self.da2)
