from math import cos, sin, sqrt
import numpy as np
import random


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

    def __init__(self, dt=0.001, disturbance=False, soft_wall=False):
        """
        Initializes the inverted pendulum simulation environment.

        Args:
            dt (float, optional): The simulation time step. Defaults to 0.001.
        """
        # Parameters from Table in "Controller Scheduling by Neural Networks"
        self.mc = 0.5  # Inertia of the base (Kg)
        self.mr = 0.2  # Mass of the pendulum (Kg)
        self.l = 0.3  # Length of the pendulum (m)
        self.br = 0.1  # Friction coefficient of the base (Kg/s)
        self.g = 9.81  # Gravitational acceleration (m/s^2)
        self.d = 0.01
        self.J = 6e-3

        # System constraints
        self.f_max = 4.0  # Max force (N)
        self.v_max = 2.75  # Max velocity (m/s)
        self.x_max = 2.0  # Max position (m)
        self.da_max = 10  # Max angular speed (rad/s)
        self.a_max = 0.5

        # Simulation time step
        self.dt = dt

        self.disturbance = disturbance  # Allow disturbance in the simualtion
        self.disturb = self.disturbance  # Controll variable for disturbance
        self.f_dist = 0  # Disturbance force
        self.int_disturb = 0  # How many interation of disturbance

        self.soft_wall = soft_wall
        self.k_wall = 100
        self.c_wall = 20

        self.n = 0

        self.st_ = None
        self.disturb_detected = False

        self.noise = False
        self.ddan = 0
        self.ddxn = 0
        # Initialize state variables
        self.reset()

    def step_sim(self, force):
        force = np.clip(force, -self.f_max, self.f_max)
        # self.pred_states(force)
        for _ in range(10):
            self.n += 1
            x, dx, a, da = self.x, self.dx, self.a, self.da

            effective_force = force
            effective_dx = dx

            if self.disturb and random.random() < 0.05 and self.n > 2500:
                self.disturb = False
                self._disturbance()

            if self.int_disturb > 0:
                f_dis = self.f_dist
                self.int_disturb -= 1
            else:
                f_dis = 0

            if self.soft_wall:
                f_wall = 0
                if x > (self.x_max * 0.99):
                    f_spring = -self.k_wall * (x - self.x_max)
                    f_damp = -self.c_wall * dx if dx > 0 else 0
                    f_wall = f_spring + f_damp
                elif x < -(self.x_max * 0.99):
                    f_spring = -self.k_wall * (x + self.x_max)
                    f_damp = -self.c_wall * dx if dx < 0 else 0
                    f_wall = f_spring + f_damp
            else:
                f_wall = 0
                at_positive_limit = (x >= self.x_max and (dx > 0 or effective_force > 0))
                at_negative_limit = (x <= -self.x_max and (dx < 0 or effective_force < 0))
                if at_positive_limit or at_negative_limit:
                    effective_dx = 0.0
                    if at_positive_limit and effective_force > 0:
                        effective_force = 0  # Force pushing against wall is nullified
                    if at_negative_limit and effective_force < 0:
                        effective_force = 0  # Force pushing against wall is nullified

            I = self.J + self.mr * self.l**2
            Phi = I * (self.mc + self.mr) - (self.mr * self.l * cos(a))**2
            f2 = -self.mr**2 * self.l**2 * da**2 * sin(a) * cos(a) + (
                self.mr + self.mc) * self.mr * self.g * self.l * sin(a) + self.mr * self.l * self.br * dx * cos(
                    a) - self.d * (self.mc + self.mr) * da
            g2 = -self.mr * self.l * cos(a)
            f1 = I * self.mr * self.l * sin(a) * da**2 - I * self.br * dx - self.mr**2 * self.l**2 * self.g * cos(
                a) * sin(a) + self.mr * self.l * self.d * cos(a) * da
            g1 = I
            f_total = effective_force + f_dis + f_wall
            dda = (f2 + g2 * f_total) / Phi
            ddx = (f1 + g1 * f_total) / Phi

            if self.noise:
                self.ddan = random.uniform(-6, 6)
                self.ddxn = random.uniform(-5, 5)

            self.dx = effective_dx + (ddx + self.ddxn) * self.dt
            self.x += self.dx * self.dt
            self.da += (dda + self.ddan) * self.dt
            self.a += self.da * self.dt

            self.x = np.clip(self.x, -self.x_max, self.x_max)
            if (self.x == self.x_max and dx > 0) or \
            (self.x == -self.x_max and dx < 0):
                self.dx = 0.0

            self.dx = np.clip(self.dx, -self.v_max, self.v_max)
            self.da = np.clip(self.da, -self.da_max, self.da_max)

        # self.detect_disturbance()

        return self.get_state()

    def pred_states(self, force):
        x_, dx_, a_, da_ = self.x, self.dx, self.a, self.da

        for _ in range(10):

            I = self.J + self.mr * self.l**2
            Phi = I * (self.mc + self.mr) - (self.mr * self.l * cos(a_))**2
            f2 = -self.mr**2 * self.l**2 * da_**2 * sin(a_) * cos(a_) + (
                self.mr + self.mc) * self.mr * self.g * self.l * sin(a_) + self.mr * self.l * self.br * dx_ * cos(
                    a_) - self.d * (self.mc + self.mr) * da_
            g2 = -self.mr * self.l * cos(a_)
            f1 = I * self.mr * self.l * sin(a_) * da_**2 - I * self.br * dx_ - self.mr**2 * self.l**2 * self.g * cos(
                a_) * sin(a_) + self.mr * self.l * self.d * cos(a_) * da_
            g1 = I
            dda = (f2 + g2 * force) / Phi
            ddx = (f1 + g1 * force) / Phi
            dx_ = dx_ + ddx * self.dt
            x_ += dx_ * self.dt
            da_ += dda * self.dt
            a_ += da_ * self.dt
        self.st_ = (x_, dx_, a_, da_)

    def detect_disturbance(self):
        if not self.disturb_detected and abs(self.st_[1] - self.dx) > 0.01 and abs(self.st_[3] - self.da) > 0.01:
            self.disturb_detected = True
        elif self.disturb_detected and abs(self.st_[1] - self.dx) < 0.005 and abs(self.st_[3] - self.da) < 0.005:
            self.disturb_detected = False

    def reset(self, initial_state=(0.0, 0.0, 0.0, 0.0)):
        """
        Resets the pendulum's state to a specified initial condition.

        Args:
            initial_state (tuple, optional): The initial state (x, dx, a, da).
                                             Defaults to (0.0, 0.0, 0.0, 0.0).

        Returns:
            tuple: The initial state.
        """
        self.n = 0
        self.disturb = self.disturbance

        self.x, self.dx, self.a, self.da = initial_state
        return self.get_state()

    def _disturbance(self):
        self.f_dist = random.uniform(-5, 5)
        self.int_disturb = random.randint(200, 500)
        print(self.f_dist)
        print(self.int_disturb)

    def get_state(self):
        """
        Returns the current state of the system.

        Returns:
            tuple: A tuple containing the current state (x, dx, a, da).
        """
        return (self.x, self.dx, self.a, self.da)
