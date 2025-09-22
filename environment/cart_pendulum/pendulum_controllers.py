import numpy as np
from math import sin, cos, pi
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum


class LQR():
    """
    Generic LQR control for pendulum dynamics
    """

    def __init__(self, k1, k2, k3, k4):
        self.K = np.array(([k1, k2, k3, k4]))

    def update_control(self, states):
        u = -self.K @ states
        return u


class SlidingMode():
    """
    Implements the Sliding Mode (SM) controller from the paper 
    "Controller Scheduling by Neural Networks" by Ferreira and Krogh (1997).
    """

    def __init__(self, pendulum_params):
        """
        Initializes the SM controller with system and control parameters.

        Args:
            pendulum_params (object): An object or dict containing the physical
                                      parameters of the pendulum (mc, m, l, br, g).
        """
        self.lambda_ = 10  # 5
        self.phi = 0.5  #0.05
        self.gamma = 2.5  #1.5

        # Physical parameters of the system to be controlled
        self.mc = pendulum_params.mc
        self.mr = pendulum_params.mr
        self.l = pendulum_params.l
        self.br = pendulum_params.br
        self.g = pendulum_params.g
        self.d = 0
        self.J = pendulum_params.J

    def update_control(self, states):
        _x, dx, a, da = states

        s = da + self.lambda_ * a

        dda_r = -self.lambda_ * da
        cos_a = cos(a)

        # Robust: avoid division by near-zero
        if abs(cos_a) < 1e-3:
            ddx_r = 0.0
        else:
            num = -(self.J + self.mr * self.l**2) * dda_r + self.mr * self.g * self.l * sin(a)
            ddx_r = num / (self.mr * self.l * cos(a))

        u_hat = (self.mc + self.mr) * ddx_r + self.mr * self.l * cos(a) * dda_r - self.mr * self.l * sin(a) * da**2 + self.br * dx

        u = u_hat + self.gamma * np.clip(s / self.phi, -1, 1)

        return u


# class SwingUp():
#     """
#     An energy-based swing-up controller.
#     It injects energy into the pendulum to swing it up to the top.
#     """

#     def __init__(self, pendulum_params, gain=2.5):
#         """
#         Args:
#             pendulum_params: An object or namespace with m, l, g, f_max.
#             gain (float): Energy control gain (tune as needed).
#         """
#         self.mc = pendulum_params.mc
#         self.mr = pendulum_params.mr
#         self.l = pendulum_params.l
#         self.g = pendulum_params.g
#         self.f_max = pendulum_params.f_max
#         self.J = pendulum_params.J

#         # Moment of inertia of a rod pivoting at the base
#         self.I = self.J + self.mr * self.l**2

#         # Desired total energy at the upright position
#         self.E_desired = self.mr * self.g * self.l

#         # Gain for energy pump
#         self.k = gain

#     def update_control(self, states):
#         """
#         Compute swing-up force based on energy difference.
#         Args:
#             states (tuple): (x, dx, a, da)
#         Returns:
#             float: Force to apply to the cart.
#         """
#         _, _, a, da = states

#         # Current pendulum energy
#         E = 0.5 * self.I * da**2 - self.mr * self.g * self.l * cos(a)

#         # Energy pumping law: sign synchronizes with swing phase
#         u = -self.k * E * np.sign(-da)

#         return u

class SwingUp:
    """
    Energy-shaping swing-up with cart recentering.
    Plug-and-play replacement for your current controller.
    """
    def __init__(self, pendulum_params,
                 k_e=2.0,      # energy gain
                 kx=0.6,       # recentering on x
                 kdx=1.2,      # damping on dx
                 friction_comp=True):
        self.mc = pendulum_params.mc
        self.mr = pendulum_params.mr
        self.l  = pendulum_params.l
        self.g  = pendulum_params.g
        self.J  = pendulum_params.J
        self.br = getattr(pendulum_params, "br", 0.0)
        self.f_max = pendulum_params.f_max

        # Inertia around pivot
        self.I = self.J + self.mr * self.l**2

        # Gains
        self.k_e  = float(k_e)
        self.kx   = float(kx)
        self.kdx  = float(kdx)
        self.friction_comp = bool(friction_comp)

    def update_control(self, states):
        x, dx, a, da = states

        if a == np.pi:
            F = 1
        else:
            # Upright-referenced energy: E* = 0 at upright rest
            E = 0.5 * self.I * da**2 + self.mr * self.g * self.l * (1.0 - cos(a))
            E_err = E  # target is 0

            # Desired cart acceleration (energy pump + recentering)
            # a_cart = - k_e * E_err * da * cos(a) - kx*x - kdx*dx
            a_cart = - self.k_e * E_err * da * np.cos(a) - self.kx * x - self.kdx * dx

            # Map desired a_cart to force; add mild friction compensation
            F = (self.mc + self.mr) * a_cart
            if self.friction_comp:
                F += self.br * dx

        # Saturate to plant limit
        return F

class PendulumStatesPredict(InvePendulum):

    def __init__(self, dt=0.001, disturbance=False):
        super().__init__(dt, disturbance)
        self.st_ = None
        self.disturb_detected = False
        self.model_mode = None

    def pred_states(self, force, x, dx, a, da):
        x_, dx_, a_, da_ = x, dx, a, da

        for _ in range(10):
            I = self.J + self.mr * self.l**2
            Phi = I * (self.mc + self.mr) - (self.mr * self.l * cos(a_))**2
            f2 = -self.mr**2 * self.l**2 * da_**2 * sin(a_) * cos(a_) + (self.mr + self.mc) * self.mr * self.g * self.l * sin(a_) + self.mr * self.l * self.br * dx_ * cos(a_) - self.d * (
                self.mc + self.mr) * da_
            g2 = -self.mr * self.l * cos(a_)
            f1 = I * self.mr * self.l * sin(a_) * da_**2 - I * self.br * dx_ - self.mr**2 * self.l**2 * self.g * cos(a_) * sin(a_) + self.mr * self.l * self.d * cos(a_) * da_
            g1 = I
            dda = (f2 + g2 * force) / Phi
            ddx = (f1 + g1 * force) / Phi
            dx_ = dx_ + ddx * self.dt
            x_ += dx_ * self.dt
            da_ += dda * self.dt
            a_ += da_ * self.dt
        self.st_ = (x_, dx_, a_, da_)

    def detect_disturbance(self, dx, da):
        if not self.disturb_detected and abs(self.st_[1] - dx) > 0.01 and abs(self.st_[3] - da) > 0.01:
            self.disturb_detected = True
        elif self.disturb_detected and abs(self.st_[1] - dx) < 0.005 and abs(self.st_[3] - da) < 0.005:
            self.disturb_detected = False

        #  if self.disturb_detected and abs(a)>0.55 -> uncontrolled
        #  if ucontolled and a = pi ->controlled donward
        #  if abs(a) < 0.55 controling
        #  if abs(a) < 0.05 upward
        #  if upward and abs(x)< 0.05 stabiled

    def model_(self, x, dx, a, da):
        self.detect_disturbance(self, dx, da)
        # if self.disturb and abs(a) > 0.55:
        #     self.model_mode = -1  # uncontrolled
        # if self.model_mode == -1 and abs(a) == np.pi and abs(dx) <= 1e-3 and abs(da) <= 1e-3 and abs(x) <= 1e-3:
        #     self.model_mode = -0.5  # controlled donward

    def reset(self):
        self.disturb_detected = False
