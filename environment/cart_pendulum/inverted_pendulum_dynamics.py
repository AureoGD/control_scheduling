from math import cos, sin, sqrt
import numpy as np


class InvePendulum():
    """
    A class to simulate the inverted pendulum system as described in the paper
    "Controller Scheduling by Neural Networks" by Ferreira and Krogh (1997).

    The simulation is based on solving the differential equations of motion
    for the pendulum and cart system.

    Attributes:
        dt (float): The time step for the simulation in seconds.
        mc (float): Inertia of the cart (base).
        mr (float): Mass of the pendulum bar.
        l (float): Length of the pendulum bar.
        br (float): Friction coefficient for the cart.
        g (float): Acceleration due to gravity.
        f_max (float): Maximum applicable force.
        v_max (float): Maximum cart velocity.
        x_max (float): Maximum cart position from the center.
        x (float): Current position of the cart.
        dx (float): Current velocity of the cart.
        a (float): Current angle of the pendulum in radians.
        da (float): Current angular velocity of the pendulum.
    """

    def __init__(self, dt=0.002):
        """
        Initializes the inverted pendulum simulation environment.

        Args:
            dt (float, optional): The simulation time step. Defaults to 0.02.
        """
        # Parameters from Table in "Controller Scheduling by Neural Networks"
        self.mc = 0.6650  # Inertia of the base (Kg)
        self.mr = 0.21  # Mass of the pendulum (Kg)
        self.l = 0.61  # Length of the pendulum (m)
        self.br = 0.2  # Friction coefficient of the base (Kg/s)
        self.g = 9.8  # Gravitational acceleration (m/s^2)

        # System constraints
        self.f_max = 5.0  # Max force (N) 6.5 stabilize VF
        self.v_max = 3.0  # Max velocity (m/s)
        self.x_max = 2.0  # Max position (m)
        self.da_max = 10  # Max angular speed (rad/s)

        # Simulation time step
        self.dt = dt

        # Initialize state variables
        self.reset()

    def step_sim(self, force):
        """
        Advances the simulation by one time step, handling boundary conditions
        for the cart's position more robustly.
        """
        # Clamp the input force to its maximum value (this is fine here)
        force = np.clip(force, -self.f_max, self.f_max)

        # Get current state for easier access
        x, dx, a, da = self.x, self.dx, self.a, self.da

        # --- PRE-CALCULATE Potential Linear Acceleration for Boundary Check ---
        # Calculate the potential ddx if there were no boundaries,
        # but don't commit to it yet. This is tricky because ddx depends on dda
        # and vice-versa. The most robust way is to modify the `B` vector or `A` matrix
        # if a boundary condition is active.

        # Let's adjust the system based on boundary conditions *before* solving.
        effective_force = force
        effective_dx = dx

        # Check if the cart is at a boundary and being pushed further into it
        # If at x_max and dx > 0 or force > 0
        at_positive_limit = (x >= self.x_max and (dx > 0 or effective_force > 0))
        # If at -x_max and dx < 0 or force < 0
        at_negative_limit = (x <= -self.x_max and (dx < 0 or effective_force < 0))

        if at_positive_limit or at_negative_limit:
            effective_dx = 0.0  # Effectively zero out velocity at boundary
            # If at limit and force is pushing against it, clamp the effective force
            # This is a simplification; a full collision model would involve impulses.
            # Here, we assume the wall is perfectly inelastic and absorbs the force.

            # Modify the equations if the cart is "stuck" at the boundary
            # If ddx must be 0, then the first equation becomes:
            # a_11 * 0 + a_12 * dda + Br * dx - 0.5*m*l*sin(a)*da^2 = F (incorrect for this system)

            # A better way is to solve for dda independently if ddx is forced to 0.
            # From the second equation: 0.5*m*cos(a)*ddx + (1/3)*m*l*dda - 0.5*m*g*sin(a) = 0
            # If ddx = 0, then:
            # (1/3)*m*l*dda = 0.5*m*g*sin(a)
            # dda = (0.5 * self.mr * self.g * sin(a)) / ((1.0 / 3.0) * self.mr * self.l)
            # dda = (1.5 * self.g * sin(a)) / self.l

            if at_positive_limit and effective_force > 0:
                effective_force = 0  # Force pushing against wall is nullified
            if at_negative_limit and effective_force < 0:
                effective_force = 0  # Force pushing against wall is nullified

            # In this case, the cart's acceleration is clamped to zero,
            # and the pendulum's acceleration is solely due to gravity/its own swing.
            # This becomes a simpler system where ddx = 0.
            # So, we can directly calculate dda from the second equation with ddx=0.
            ddx = 0.0
            dda = (0.5 * self.mr * self.g * sin(a)) / ((1.0 / 3.0) * self.mr * self.l)
            # Simplified dda = (1.5 * self.g * sin(a)) / self.l

        else:
            # --- Solve the system of linear equations for accelerations ---
            # Define the A matrix
            a_11 = self.mc
            a_12 = 0.5 * self.mr * self.l * cos(a)
            a_21 = 0.5 * self.mr * cos(a)
            a_22 = (1.0 / 3.0) * self.mr * self.l
            A = np.array([[a_11, a_12], [a_21, a_22]])

            # Define the B vector
            # Use effective_force and effective_dx if they were modified, otherwise they are original.
            b_1 = effective_force - self.br * effective_dx + 0.5 * self.mr * self.l * (da**2) * sin(a)
            b_2 = 0.5 * self.mr * self.g * sin(a)
            B = np.array([b_1, b_2])

            try:
                # Solve for the acceleration vector [ddx, dda]
                acc = np.linalg.solve(A, B)
                ddx, dda = acc[0], acc[1]
            except np.linalg.LinAlgError:
                # If the matrix is singular (e.g., at a=pi/2), accelerations are zero
                ddx, dda = 0.0, 0.0

        # --- Update state using Euler integration ---
        # Use effective_dx if it was set to zero due to boundary, otherwise use current dx
        self.dx = effective_dx + ddx * self.dt  # Use effective_dx as starting point for integration
        self.x += self.dx * self.dt
        self.da += dda * self.dt
        self.a += self.da * self.dt

        # Enforce state constraints (clipping position)
        # This clipping is still needed to strictly enforce boundaries
        # in case of minor numerical inaccuracies or large dt.
        prev_x = self.x  # Store previous x for boundary check if needed
        self.x = np.clip(self.x, -self.x_max, self.x_max)

        # Re-check velocity if position was clipped (due to numerical overshoot after integration)
        # If the cart was at a boundary and moved slightly past it then clipped back,
        # its velocity in that direction should be zeroed.
        if (self.x == self.x_max and dx > 0) or \
           (self.x == -self.x_max and dx < 0):
            self.dx = 0.0

        # Ensure velocities are within max limits
        self.dx = np.clip(self.dx, -self.v_max, self.v_max)
        self.da = np.clip(self.da, -self.da_max, self.da_max)

        return self.get_state()

    def reset(self, initial_state=(0.0, 0.0, 0.0, 0.0)):
        """
        Resets the pendulum's state to a specified initial condition.

        Args:
            initial_state (tuple, optional): The initial state (x, dx, a, da).
                                             Defaults to (0.0, 0.0, 0.0, 0.0).

        Returns:
            tuple: The initial state.
        """
        self.x, self.dx, self.a, self.da = initial_state
        return self.get_state()

    def get_state(self):
        """
        Returns the current state of the system.

        Returns:
            tuple: A tuple containing the current state (x, dx, a, da).
        """
        return (self.x, self.dx, self.a, self.da)
