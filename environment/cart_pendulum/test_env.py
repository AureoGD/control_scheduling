import numpy as np
import matplotlib.pyplot as plt
# from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv
from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
from scheduller_rules.schl_rule1 import SchedullerRule

DT = 0.001


def run_test():
    # Create the environment
    sw_rule = SchedullerRule()
    env = InvPendulumEnv(env_id=0, dt=DT, max_step=500, disturbance=False, rendering=True, sw_rule=sw_rule)
    # env = InvPendulumEnv(env_id=0, max_step=500, rendering=True)

    # Reset environment
    # obs, info = env.reset(x0=[-1.0, -0.46652448177337646, -0.09152679145336151, -2.5])
    x0 = np.array([
        np.random.uniform(-1.5, 1.5),
        np.random.uniform(-2.5, 2.5),
        np.random.uniform(-0.5, 0.5),
        np.random.uniform(-2.5, 2.5)
    ],
                  dtype=np.float32)

    obs, info = env.reset(x0=x0)

    # obs, info = env.reset(x0=[0, 0, 0.1, 0])
    done = False
    total_reward = 0

    print("Episode started")
    history = {
        'states': [],
        'time': [],
        'control_effort': [],
        'reward': [],
        'pred_state': [],
        'disturbance': [],
        'scores': []
    }
    ep = 1
    i = 0
    while not done:
        action = env.action_space.sample()  # Random action for now
        # if i < 70:
        #     action = 0
        # elif i >= 70 and i < 200:
        #     action = 1
        # else:
        #     action = 2
        action = 0
        obs, reward, terminated, truncated, info = env.step(action)
        history['states'].append(obs.copy())
        f = float(info.get("control_effort", np.nan))
        history['control_effort'].append(np.clip(f, -4, 4))
        # history['pred_state'].append(info.get("pred_state"))
        # history['disturbance'].append(info.get("dist_detected"))
        history['time'].append(ep * DT * 10)
        total_reward += reward
        history['reward'].append(total_reward)
        history['scores'].append(info.get('scores'))
        done = terminated or truncated
        ep += 1
        i += 1

    print("Episode finished.")
    print("Total reward:", total_reward)

    env.close()

    states_history = np.array(history['states'])
    control_effort_history = np.array(history['control_effort'])
    time_history = np.array(history['time'])
    reward_history = np.array(history['reward'])
    scores = np.array(history['scores'])
    # pred_states = np.array(history['pred_state'])
    # disturbance = np.array(history["disturbance"])

    # angles_unwrapped = states_history[:, 2]
    # angles_wrapped = (angles_unwrapped + np.pi) % (2 * np.pi) - np.pi  # Wrap for display

    plot_fig, plot_axs = plt.subplots(7, 1, figsize=(12, 10), sharex=True)
    plot_fig.suptitle("Simulation History: Inverted Pendulum Control", fontsize=16)

    # Cart position
    plot_axs[0].plot(time_history, states_history[:, 0] * env.inv_pendulum.x_max, label="x (m)")
    # plot_axs[0].plot(time_history, pred_states[:, 0], 'r:', label="x_ (m)")
    plot_axs[0].set_ylabel("Cart Position")
    plot_axs[0].set_title("Cart Position")
    plot_axs[0].grid(True)
    plot_axs[0].legend()

    # Cart velocity
    plot_axs[1].plot(time_history, states_history[:, 1] * env.inv_pendulum.v_max, label="dx (m/m)")
    # plot_axs[1].plot(time_history, pred_states[:, 1], 'r:', label="dx_ (m/s)")
    plot_axs[1].set_ylabel("Cart Velocity")
    plot_axs[1].set_title("Cart Velocity")
    plot_axs[1].grid(True)
    plot_axs[1].legend()

    # Pendulum angle
    plot_axs[2].plot(time_history, np.arctan2(states_history[:, 3], states_history[:, 2]), label="q (rad)")
    # plot_axs[2].plot(time_history, pred_states[:, 2], 'r:', label="q_ (rad)")
    plot_axs[2].set_ylabel("Angle")
    plot_axs[2].set_title("Pendulum Angle")
    plot_axs[2].grid(True)
    plot_axs[2].legend()

    # Pendulum angle spped
    plot_axs[3].plot(time_history, states_history[:, 4] * env.inv_pendulum.da_max, label="dq (rad/s)")
    # plot_axs[3].plot(time_history, pred_states[:, 3], 'r:', label="dq_ (rad/s)")
    plot_axs[3].set_ylabel("Angle speed")
    plot_axs[3].set_title("Pendulum Angle Speed")
    plot_axs[3].grid(True)
    plot_axs[3].legend()

    # Control effort
    plot_axs[4].plot(time_history, control_effort_history, 'k-', label="Force (N)")
    plot_axs[4].set_ylabel("Force (N)")
    plot_axs[4].set_xlabel("Time (s)")
    plot_axs[4].set_title("Control Effort")
    plot_axs[4].grid(True)
    plot_axs[4].legend()

    # Reward
    plot_axs[5].plot(time_history, reward_history, 'k-', label="Reward")
    plot_axs[5].set_ylabel("Reward")
    plot_axs[5].set_xlabel("Time (s)")
    plot_axs[5].set_title("Reward")
    plot_axs[5].grid(True)
    plot_axs[5].legend()

    # Disturbance
    # plot_axs[6].plot(time_history, disturbance, 'k-', label="Disturbance")
    plot_axs[6].set_ylabel("Disturbance")
    plot_axs[6].set_xlabel("Time (s)")
    plot_axs[6].set_title("Disturbance")
    plot_axs[6].grid(True)
    plot_axs[6].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)

    # plot_fig, plot_axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # plot_axs[0].plot(time_history, scores[:, 0])
    # plot_axs[0].grid(True)
    # plot_axs[0].legend()

    # plot_axs[1].plot(time_history, scores[:, 1])
    # plot_axs[1].grid(True)
    # plot_axs[1].legend()

    # plot_axs[2].plot(time_history, scores[:, 2])
    # plot_axs[2].grid(True)
    # plot_axs[2].legend()

    # plot_axs[3].plot(time_history, scores[:, 3])
    # plot_axs[3].grid(True)
    # plot_axs[3].legend()

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show(block=True)


if __name__ == "__main__":
    run_test()
