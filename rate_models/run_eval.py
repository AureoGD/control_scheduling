import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from stable_baselines3 import A2C, PPO, DQN
from es_framework.commons.control_rule import ControlRule

# Change only here !
IS_DISCRETE = True
GENERATE_X0 = True
RENDERING = False
NOISE = False
DISTURBANCE = True
NUM_SAMPLES = 10

MODEL_PATH = "models/cem/D_15165010/cem_model_final_mean.pth" #cem_model_final_mean
# MODEL_PATH = "models/dqn/D_15150819/final_model.zip" #final_model
MODEL_NAME = "cem"  # 'cmaes', 'ppo', 'a2c', 'dqn'
EXP_ID = "exp2"

# -------------------------------------------
SIM_TIME = 5
DT = 0.001
MAX_STEP = int(SIM_TIME / DT)
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cpu'
ROOT = os.path.dirname(os.path.abspath(__file__))  # .../eval_models
DATA_DIR = os.path.join(ROOT, "eval_data")
os.makedirs(DATA_DIR, exist_ok=True)
IC_PATH = os.path.join(DATA_DIR, "initial_conditions.csv")
PREFIX = "d" if IS_DISCRETE else "c"
OUT_FILE = f"{PREFIX}_{MODEL_NAME}_data_{EXP_ID}.csv"
OUT_PATH = os.path.join(DATA_DIR, OUT_FILE)


def write_initial_conditions(path, X0):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "xdot", "theta", "thetadot"])
        w.writerows(X0.tolist())


def read_initial_conditions(path):
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=1)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


if GENERATE_X0:
    rng = np.random.default_rng(55)
    X0 = []
    for _ in range(NUM_SAMPLES):
        x0 = np.array([
            np.random.uniform(-1.0, 1.0),
            np.random.uniform(-2.5, 2.5),
            np.random.uniform(-0.5, 0.5),
            np.random.uniform(-1.5, 1.5)
        ],
                            dtype=np.float32)
        X0.append(x0)
    X0 = np.stack(X0, axis=0)
    write_initial_conditions(IC_PATH, X0)
else:
    X0 = read_initial_conditions(IC_PATH)

if IS_DISCRETE:
    from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
    from scheduller_rules.schl_rule1 import SchedullerRule
    sw_rule = SchedullerRule()
    env = InvPendulumEnv(env_id=0, sw_rule=sw_rule, rendering=RENDERING, noise=NOISE, disturbance=DISTURBANCE)
else:
    from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv
    env = InvPendulumEnv(env_id=0, rendering=RENDERING, noise=NOISE, disturbance=DISTURBANCE)

# ---- Load model ----
if MODEL_NAME in ['cem', 'cmaes']:
    saved_state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    output_dim = env.action_space.n if IS_DISCRETE else env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]
    config = {'discrete': IS_DISCRETE}
    model = ControlRule(observation_dim=observation_dim, output_dim=output_dim, **config).to(DEVICE)
    model.load_state_dict(saved_state_dict)
    model.eval()
elif MODEL_NAME == 'ppo':
    model = PPO.load(MODEL_PATH, device=DEVICE)
elif MODEL_NAME == 'a2c':
    model = A2C.load(MODEL_PATH, device=DEVICE)
elif MODEL_NAME == 'dqn':
    model = DQN.load(MODEL_PATH, device=DEVICE)
else:
    raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")

with open(OUT_PATH, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow([
        "ic_id", "step", "time", "x", "xdot", "theta", "thetadot", "mode", "control_effort", "reward_step",
        "total_reward", "terminated", "truncated"
    ])

    for ic_id, x0 in enumerate(tqdm(X0, desc="Initial conditions")):
        obs, info = env.reset(x0=x0)
        done = False
        step = 0
        total_reward = 0.0

        while not done and step < MAX_STEP:
            if IS_DISCRETE:
                action, _ = model.predict(obs, deterministic=True)
                mode = int(action)  # keep mode = chosen discrete action
            else:
                action, _ = model.predict(obs, deterministic=True)
                mode = -1
            obs_next, r, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)

            raw = info.get("raw_state", info.get("raw_states", None))
            x = float(raw[0]) if raw is not None else np.nan
            xdot = float(raw[1]) if raw is not None else np.nan
            theta = float(raw[2]) if raw is not None else np.nan
            thetadot = float(raw[3]) if raw is not None else np.nan

            control_effort = float(np.clip(info.get("control_effort", np.nan), -env.inv_pendulum.f_max, env.inv_pendulum.f_max))
            total_reward += float(r)

            # Optional safety cutoff
            if theta >= np.pi / 2:
                done = True

            w.writerow([
                ic_id, step, step * DT, x, xdot, theta, thetadot, mode, control_effort,
                float(r), total_reward,
                int(terminated),
                int(truncated)
            ])

            obs = obs_next
            step += 1

print(f"Eval finished. Data saved to: {OUT_PATH}")
