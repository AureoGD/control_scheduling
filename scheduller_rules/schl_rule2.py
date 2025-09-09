from environment.cart_pendulum.env_pendulum_disc import SchedullerRule
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_controllers import LQR, SlidingMode, SwingUp


class SchedullerRule(SchedullerRule):

    def __init__(self):
        self.ip = InvePendulum()
        self.lqr = LQR(10.0, 12.60, 48.33, 9.09)
        self.sm = SlidingMode(self.ip)
        self.vf = LQR(0, 30.92, 87.63, 20.40)
        self.swup = SwingUp(self.ip)
        self.n_controllers = 4

    def update_control_action(self, controller_index, state):
        if controller_index == 0:
            return self.lqr.update_control(state)
        elif controller_index == 1:
            return self.sm.update_control(state)
        elif controller_index == 2:
            return self.vf.update_control(state)
        elif controller_index == 3:
            return self.swup.update_control(state)
        else:
            return 0
