import numpy as np
from environment.cart_pendulum.env_pendulum_disc import SchedullerRule
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode, SwingUp


class SchedullerRule(SchedullerRule):

    def __init__(self):
        self.ip = InvePendulum()
        self.lqr = LQR(-2.91, -3.67, -25.43, -4.94)  # Q = [5 1 10 1] R = 0.5
        self.vf = LQR(0, -33.90, -153.30, -32.07)  # Q = [1e-6 59.50 10 5] R = 0.01
        self.sm = SlidingMode(self.ip)
        self.swup = SwingUp(self.ip)
        self.down_lqr = LQR(2.24, 2.27, 3.21, 0.78)
        
        self.n_controllers = 5

    def update_control_action(self, controller_index, state):
        if controller_index == 0:
            return self.sm.update_control(state)
        elif controller_index == 1:
            return self.vf.update_control(state)
        elif controller_index == 2:
            return self.lqr.update_control(state)
        elif controller_index == 3:
            return self.swup.update_control(state)
        elif controller_index ==4:
            x, dx, a, da = state
            a = a-np.pi
            return self.down_lqr.update_control((x,dx,a,da))
        else:
            return 0
