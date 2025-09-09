import numpy as np
import time
from collections import deque


def evaluate_agent(env, agent, n_episodes):
    """
    Runs the agent for n_episodes in the environment using a deterministic policy
    and returns the average reward.
    """
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
        total_reward += episode_reward
    return total_reward / n_episodes


def train_agent(env, eval_env, agent, buffer, logger, buffer_zize, config):
    """
    The main training loop for the SAC agent.

    Args:
        env: The training environment.
        eval_env: The evaluation environment.
        agent: The agent to train.
        buffer: The replay buffer.
        logger: The logger for saving models and metrics.
        config (dict): A dictionary of training hyperparameters.
    """
    # --- Training Loop ---
    obs, _ = env.reset()
    episode_reward = 0
    episode_len = 0
    episodes = 0
    n_updates = 0

    ep_rew_history = deque(maxlen=100)
    ep_len_history = deque(maxlen=100)
    start_time = time.time()

    print("\n--- Starting Training ---")
    for step in range(config["total_timesteps"]):
        # Select action
        if step < config["learning_starts"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        # Environment step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.store(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward
        episode_len += 1

        # Handle episode end
        if done:
            episodes += 1
            ep_rew_history.append(episode_reward)
            ep_len_history.append(episode_len)

            logger.log_scalar("rollout/ep_rew_mean", np.mean(ep_rew_history),
                              step)
            logger.log_scalar("rollout/ep_len_mean", np.mean(ep_len_history),
                              step)
            logger.log_scalar("time/episodes", episodes, step)

            obs, _ = env.reset()
            episode_reward = 0
            episode_len = 0

        # Update the agent
        if step >= config["learning_starts"]:
            batch = buffer.sample(config["batch_size"])
            update_info = agent.update(batch)
            n_updates += 1

            if n_updates % 100 == 0:
                for key, value in update_info.items():
                    logger.log_scalar(f"train/{key}", value, step)
                logger.log_scalar("train/n_updates", n_updates, step)

        # --- Periodic Evaluation and Saving ---
        if (step + 1) % config["eval_freq"] == 0:
            eval_reward = evaluate_agent(eval_env, agent,
                                         config["n_eval_episodes"])

            time_elapsed = time.time() - start_time
            fps = (step + 1) / time_elapsed
            train_reward_mean = np.mean(
                ep_rew_history) if ep_rew_history else 0.0
            print("-" * 80)
            print(
                f"Step: {step+1} | Eval Reward: {eval_reward:.2f} | Train Reward: {train_reward_mean:.2f} | FPS: {fps:.2f}"
            )
            print("-" * 80)

            logger.log_scalar("eval/mean_reward", eval_reward, step)
            logger.log_scalar("time/fps", fps, step)
            logger.log_scalar("time/time_elapsed", time_elapsed, step)
            logger.log_scalar("time/total_timesteps", step + 1, step)

            logger.save_periodic_model(agent, step + 1)
            logger.save_best_model(agent, eval_reward)

    # --- Final Save and Cleanup ---
    print("\n--- Training Finished ---")
    logger.save_periodic_model(agent, "final")
    logger.close()
    env.close()
    eval_env.close()
    print(f"Final models and logs saved in their respective directories.")
    print("-------------------------")
