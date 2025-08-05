import numpy as np
import matplotlib.pyplot as plt
from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv
# from envirioment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
# from scheduller_rules.schl_rule1 import SchedullerRule


def run_test():
    # Create the environment
    # sw_rule = SchedullerRule()
    env = InvPendulumEnv(env_id=0, max_step=5000, rendering=True)

    # Reset environment
    obs, info = env.reset()
    done = False
    total_reward = 0

    print("Episode started")
    history = {'states': [], 'time': [], 'control_effort': [], 'reward': []}
    ep = 1

    while not done:
        action = env.action_space.sample()  # Random action for now
        # action = [10]
        obs, reward, terminated, truncated, info = env.step([0])
        history['states'].append(obs.copy())
        history['control_effort'].append(action)
        history['time'].append(ep * 0.002)
        total_reward += reward
        history['reward'].append(total_reward)
        done = terminated or truncated
        ep += 1

    print("Episode finished.")
    print("Total reward:", total_reward)

    env.close()

    states_history = np.array(history['states'])
    control_effort_history = np.array(history['control_effort'])
    time_history = np.array(history['time'])
    reward_history = np.array(history['reward'])

    # angles_unwrapped = states_history[:, 2]
    # angles_wrapped = (angles_unwrapped + np.pi) % (2 * np.pi) - np.pi  # Wrap for display

    plot_fig, plot_axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    plot_fig.suptitle("Simulation History: Inverted Pendulum Control",
                      fontsize=16)
    plot_axs[0].plot(time_history, states_history[:, 2])
    plot_axs[0].set_ylabel("cos(a)")
    plot_axs[0].plot(time_history, states_history[:, 3])
    plot_axs[0].set_ylabel("sin(a)")
    plot_axs[0].set_title("Pendulum Angle Trajectory")
    plot_axs[0].grid(True)
    # plot_axs[0].legend()
    plot_axs[1].plot(time_history,
                     states_history[:, 0] * env.inv_pendulum.x_max, 'g-')
    plot_axs[1].set_ylabel("Position (m)")
    plot_axs[1].set_title("Cart Position")
    plot_axs[1].grid(True)
    plot_axs[2].plot(time_history, control_effort_history, 'k-')
    plot_axs[2].set_ylabel("Force (N)")
    plot_axs[2].set_xlabel("Time (s)")
    plot_axs[2].set_title("Control Effort")
    plot_axs[2].grid(True)

    plot_axs[3].plot(time_history, reward_history, 'k-')
    plot_axs[3].set_ylabel("Reward")
    plot_axs[3].set_xlabel("Time (s)")
    plot_axs[3].set_title("Reward")
    plot_axs[3].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)  # Show non-blocking


if __name__ == "__main__":
    run_test()
