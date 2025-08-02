import pybullet as p
import pybullet_data
import numpy as np
import time
from math import pi
import matplotlib.pyplot as plt

from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum


class InvertedPendulumPyBullet:
    """
    Updated class with multiple methods for constraining cart position.
    """

    def __init__(self, dt=0.002, render=False):
        self.dt = dt
        self.mode = p.GUI if render else p.DIRECT
        # --- Connect to PyBullet ---
        # Check if a client is already connected to avoid creating a new one
        self.client = p.connect(self.mode) if p.getConnectionInfo()['isConnected'] == 0 else 0

        ip = InvePendulum()

        # --- System Parameters & Constraints ---
        self.g = ip.g
        self.br = ip.br
        self.f_max = ip.f_max
        self.v_max = ip.v_max
        self.x_max = ip.x_max
        self.da_max = ip.da_max

        # --- Soft Wall Parameters ---
        self.boundary_stiffness = 5000  # k: Spring stiffness
        self.boundary_damping = 50  # c: Damping

        # --- Setup Simulation Environment ---
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.g)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt)

        self.plane = p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.7]  # Lifted to allow swing-up
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.pendulum = p.loadURDF("model/inve_pendulum.urdf", start_pos, start_orn)

        self.joint_indices = {p.getJointInfo(self.pendulum, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(self.pendulum))}

        p.changeDynamics(self.pendulum, -1, linearDamping=self.br)
        p.changeDynamics(self.pendulum, self.joint_indices['cart_to_rod'], linearDamping=0, angularDamping=0.1)

        # Disable default motors
        for joint_index in self.joint_indices.values():
            p.setJointMotorControl2(self.pendulum, joint_index, p.VELOCITY_CONTROL, force=0)

        self.reset()

    def step_sim_soft_walls(self, control_force):
        """
        Advances the simulation, applying a spring-damper force at the boundaries.
        """
        x, dx, _, _ = self.get_state()

        # Calculate boundary force
        boundary_force = 0.0
        if x > self.x_max:
            displacement = x - self.x_max
            boundary_force = -self.boundary_stiffness * displacement - self.boundary_damping * dx
        elif x < -self.x_max:
            displacement = x + self.x_max
            boundary_force = -self.boundary_stiffness * displacement - self.boundary_damping * dx

        # Combine control force and boundary force
        total_force = np.clip(control_force, -self.f_max, self.f_max) + boundary_force

        # Apply total force
        p.setJointMotorControl2(self.pendulum, self.joint_indices['slider_to_cart'], p.TORQUE_CONTROL, force=total_force)
        p.stepSimulation()

        return self._enforce_velocity_limits_and_get_state()

    def step_sim_hard_clamp(self, control_force):
        """
        Advances the simulation and then manually clamps position if out of bounds.
        """
        # Step 1: Apply force and step simulation as normal
        force = np.clip(control_force, -self.f_max, self.f_max)
        p.setJointMotorControl2(self.pendulum, self.joint_indices['slider_to_cart'], p.TORQUE_CONTROL, force=force)
        p.stepSimulation()

        # Step 2: Get the state from the physics engine
        x, dx, a, da = self.get_state()

        # Step 3: Check if position is out of bounds and clamp if necessary
        if abs(x) > self.x_max:
            # Clamp the position to the boundary
            x_clamped = np.clip(x, -self.x_max, self.x_max)
            # Reset the joint state with the new position and zero velocity
            p.resetJointState(self.pendulum, self.joint_indices['slider_to_cart'], targetValue=x_clamped, targetVelocity=0.0)
            # Update state variables after clamping
            x, dx = x_clamped, 0.0

        return self._enforce_velocity_limits_and_get_state()

    def _enforce_velocity_limits_and_get_state(self):
        """Helper function to clip velocities and return the final state."""
        x, dx, a, da = self.get_state()

        dx_clipped = np.clip(dx, -self.v_max, self.v_max)
        da_clipped = np.clip(da, -self.da_max, self.da_max)

        # If velocity was clipped, reset the state to enforce the hard limit
        if dx_clipped != dx or da_clipped != da:
            p.resetJointState(self.pendulum, self.joint_indices['slider_to_cart'], targetValue=x, targetVelocity=dx_clipped)
            p.resetJointState(self.pendulum, self.joint_indices['cart_to_rod'], targetValue=a, targetVelocity=da_clipped)

        return self.get_state()

    def reset(self, initial_state=(0.0, 0.0, 0.0, 0.0)):
        """Resets the pendulum's state."""
        x, dx, a, da = initial_state
        p.resetJointState(self.pendulum, self.joint_indices['slider_to_cart'], targetValue=x, targetVelocity=dx)
        p.resetJointState(self.pendulum, self.joint_indices['cart_to_rod'], targetValue=a, targetVelocity=da)
        return self.get_state()

    def get_state(self):
        """Returns the current state of the system."""
        x, dx = p.getJointState(self.pendulum, self.joint_indices['slider_to_cart'])[:2]
        a, da = p.getJointState(self.pendulum, self.joint_indices['cart_to_rod'])[:2]
        return (x, dx, a, da)

    def close(self):
        if p.getConnectionInfo()['isConnected'] == 1:
            p.disconnect(self.client)
