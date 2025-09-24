import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Configuration ----------------
ALG = "cem"
DISCREATE = False   
EXP = "exp1"
# ------------------------------------------------

PREFIX = "d" if DISCREATE else "c"
FILE_PATH = f"rate_models/eval_data/{PREFIX}_{ALG}_data_{EXP}.csv"

# Energy-like cost J (control_effort is RAW scalar action u)
P = np.diag([1.0, 1.0, 5.0, 1.0])   # [x, xdot, theta, thetadot]
R = np.diag([0.01])                 # scalar action weight
S = P.copy()                        # terminal penalty (set None to disable)
DEFAULT_DT = None                   # if CSV lacks 'time', set e.g. 0.001

# Fail penalty policy
AUTO_FAIL_PENALTY = True            # penalty = 10× median(non-failed J) (min floor below)
FIXED_FAIL_PENALTY_FLOOR = 1000.0   # minimum fail penalty if needed

# Summary files
SUMMARY_DIR = os.path.dirname(FILE_PATH)
SUMMARY_PATH = os.path.join(SUMMARY_DIR, "metrics_summary.csv")   # summary per (key, exp)
FAIL_INDEX_PATH = os.path.join(SUMMARY_DIR, "fail_indices.csv")   # only key; IC_ID; exp

# ----------------- Load data -------------------
df = pd.read_csv(FILE_PATH)

# -------------- Helpers (minimal) --------------
def _x_from_row(r: pd.Series) -> np.ndarray:
    needed = ["x", "xdot", "theta", "thetadot"]
    missing = [k for k in needed if k not in r.index]
    if missing:
        raise KeyError(f"Missing columns in CSV: {missing}")
    return np.array([r["x"], r["xdot"], r["theta"], r["thetadot"]], dtype=float)

def _dt_series_for_group(g: pd.DataFrame, default_dt=None):
    if "time" in g.columns:
        t = g["time"].to_numpy(float)
        dts = np.diff(t, prepend=t[0])
        if len(dts) > 1:
            med_dt = np.median(dts[1:][dts[1:] > 0]) if np.any(dts[1:] > 0) else (default_dt if default_dt else 1.0)
        else:
            med_dt = default_dt if default_dt else 1.0
        dts[0] = med_dt
        total_time = t[-1] - t[0] if len(t) > 1 else np.nan
        return dts, total_time
    else:
        if default_dt is None:
            default_dt = 1.0
        return np.full(len(g), float(default_dt)), np.nan

def compute_episode_cost(g: pd.DataFrame,
                         P: np.ndarray,
                         R: np.ndarray,
                         S: np.ndarray | None,
                         default_dt: float | None,
                         fail_penalty: float | None):
    """Unique path: control_effort is RAW scalar action u; uRu = u^T R u."""
    sort_cols = [c for c in ["time", "step"] if c in g.columns]
    if sort_cols:
        g = g.sort_values(sort_cols).reset_index(drop=True)

    dts, total_time = _dt_series_for_group(g, default_dt)
    n = 4
    if P.shape != (n, n):
        raise ValueError(f"P must be {n}x{n}, got {P.shape}")
    if R.shape != (1, 1):
        raise ValueError(f"R must be 1x1 for scalar action, got {R.shape}")
    if S is not None and S.shape != (n, n):
        raise ValueError(f"S must be {n}x{n}, got {S.shape}")

    J_run = 0.0
    for (_, row), dt in zip(g.iterrows(), dts):
        x = _x_from_row(row)
        xPx = float(x @ P @ x)

        # uRu from raw scalar 'control_effort'
        if "control_effort" not in row.index or pd.isna(row["control_effort"]):
            uRu = 0.0
        else:
            u = float(row["control_effort"])
            uRu = float(np.array([u]) @ R @ np.array([u]))

        J_run += (xPx + uRu) * float(dt)

    J_term = 0.0 if S is None else float((_x_from_row(g.iloc[-1]) @ S @ _x_from_row(g.iloc[-1])))

    terminated = int(g.iloc[-1]["terminated"]) if "terminated" in g.columns else 0
    J_fail = float(fail_penalty) if (fail_penalty is not None and terminated == 1) else 0.0

    J = J_run + J_term + J_fail
    J_rate = J / total_time if (isinstance(total_time, (int, float)) and np.isfinite(total_time) and total_time > 0) else np.nan
    return J, J_rate

def compute_J_table(df: pd.DataFrame,
                    P: np.ndarray, R: np.ndarray, S: np.ndarray | None,
                    default_dt: float | None,
                    auto_fail_penalty: bool,
                    fail_penalty_floor: float):
    """Two-pass: get typical non-fail energy, then assign big fail penalty."""
    # Pass 1: no penalty
    pass1 = []
    for ic_id, g in df.groupby("ic_id"):
        J, _ = compute_episode_cost(g, P, R, S, default_dt, fail_penalty=None)
        term = int(g.iloc[-1]["terminated"]) if "terminated" in g.columns else 0
        pass1.append({"ic_id": ic_id, "J_base": J, "terminated": term})
    pass1_df = pd.DataFrame(pass1).sort_values("ic_id")

    if auto_fail_penalty:
        non_failed = pass1_df.loc[pass1_df["terminated"] == 0, "J_base"].values
        if len(non_failed) == 0:
            fail_penalty = fail_penalty_floor
        else:
            med = float(np.median(non_failed))
            fail_penalty = max(10.0 * med, fail_penalty_floor)
    else:
        fail_penalty = fail_penalty_floor

    # Pass 2: final J with penalty
    rows = []
    for ic_id, g in df.groupby("ic_id"):
        J, J_rate = compute_episode_cost(g, P, R, S, default_dt, fail_penalty=fail_penalty)
        rec = {"ic_id": ic_id, "J": J, "J_rate": J_rate, "fail_penalty_used": fail_penalty}
        if "terminated" in g.columns:
            rec["terminated"] = int(g.iloc[-1]["terminated"])
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("ic_id").reset_index(drop=True)

# ------------- Compute J & reward/energy stats -------------
J_df = compute_J_table(df, P, R, S, DEFAULT_DT, AUTO_FAIL_PENALTY, FIXED_FAIL_PENALTY_FLOOR)

last_rows = df.sort_values(["ic_id", "step"]).groupby("ic_id").tail(1)
rewards = last_rows["total_reward"].astype(float)
num_fails = int((last_rows["terminated"] == 1).sum()) if "terminated" in last_rows.columns else 0

n = len(rewards)
if n == 0:
    raise ValueError("No episodes found to summarize.")

# Reward stats
reward_mean   = float(rewards.mean())
reward_std    = float(rewards.std(ddof=1))
reward_median = float(rewards.median())
reward_min    = float(rewards.min())
reward_max    = float(rewards.max())
reward_se     = reward_std / np.sqrt(n)
reward_ci95   = 1.96 * reward_se
reward_low_ci = reward_mean - reward_ci95
reward_high_ci= reward_mean + reward_ci95

# Energy stats (use per-IC J)
energy = J_df["J"].astype(float)
energy_mean   = float(energy.mean())
energy_median = float(energy.median())
energy_min    = float(energy.min())
energy_max    = float(energy.max())
energy_std    = float(energy.std(ddof=1))
energy_se     = energy_std / np.sqrt(len(energy))
energy_ci95   = 1.96 * energy_se
energy_low_ci = energy_mean - energy_ci95
energy_high_ci= energy_mean + energy_ci95

# Print quick summary to console (optional)
print("\nGeneral metrics")
print("-" * 40)
print(f"Algorithm      : {ALG}{'-D' if DISCREATE else '-C'} / {EXP}")
print(f"Episodes       : {n}")
print(f"Mean reward    : {reward_mean:.3f}")
print(f"Std reward     : {reward_std:.3f}")
print(f"Min reward     : {reward_min:.3f}")
print(f"Median reward  : {reward_median:.3f}")
print(f"Max reward     : {reward_max:.3f}")
print(f"Num fails      : {num_fails}")
print(f"Mean energy J  : {energy_mean:.3f}")
print(f"Median energy  : {energy_median:.3f}")
print(f"Min/Max energy : {energy_min:.3f} / {energy_max:.3f}")
print(f"95% CI reward  : [{reward_low_ci:.3f}, {reward_high_ci:.3f}]")
print(f"95% CI energy  : [{energy_low_ci:.3f}, {energy_high_ci:.3f}]")
print("-" * 40)

# ------------- Write metrics_summary.csv (overwrite only if key & exp both match) -------------
key = f"{ALG}-{'D' if DISCREATE else 'C'}"
exp = f"{EXP}"

summary_row = {
    "key": key,
    "exp": exp,
    "mean_reward": round(reward_mean, 2),
    "std_reward": round(reward_std, 2),
    "median_reward": round(reward_median, 2),
    "min_reward": round(reward_min, 2),
    "max_reward": round(reward_max, 2),
    "reward_CI": round(reward_ci95, 2),
    "reward_low_CI": round(reward_low_ci, 2),
    "reward_high_CI": round(reward_high_ci, 2),
    "mean_energy": round(energy_mean, 2),
    "median_energy": round(energy_median, 2),
    "min_energy": round(energy_min, 2),
    "max_energy": round(energy_max, 2),
    "energy_CI": round(energy_ci95, 2),
    "energy_low_CI": round(energy_low_ci, 2),
    "energy_high_CI": round(energy_high_ci, 2),
    "num_fails": int(num_fails),
    "fail_percent": round(float(num_fails * 100 / n), 2)
}

columns_order = [
    "key", "exp",
    "mean_reward","std_reward","reward_CI","mean_energy","energy_CI","fail_percent",
    "median_reward","min_reward","max_reward",
    "reward_low_CI","reward_high_CI","median_energy",
    "min_energy","max_energy","energy_low_CI","energy_high_CI","num_fails"
]

if not os.path.exists(SUMMARY_PATH):
    pd.DataFrame([summary_row], columns=columns_order).to_csv(SUMMARY_PATH, index=False)
else:
    s = pd.read_csv(SUMMARY_PATH)
    # Ensure 'exp' column exists in older files
    if "exp" not in s.columns:
        s["exp"] = np.nan
    # Drop only rows where BOTH key and exp match
    s = s[~((s["key"] == key) & (s["exp"] == exp))]
    s = pd.concat([s, pd.DataFrame([summary_row])], ignore_index=True)
    # Keep only requested columns and order
    for c in columns_order:
        if c not in s.columns:
            s[c] = np.nan
    s = s[columns_order]
    s.to_csv(SUMMARY_PATH, index=False)

print(f"Summary row written to: {SUMMARY_PATH}")

# --------- Build IC table & fail indices (overwrite only if key & exp both match) ----------
first_rows = df[df["step"] == 0][["ic_id", "x", "theta", "xdot", "thetadot"]].drop_duplicates(subset=["ic_id"])
terminated_last = last_rows[["ic_id", "terminated"]]
ic_info = first_rows.merge(terminated_last, on="ic_id", how="left")
ic_info["terminated"] = ic_info["terminated"].fillna(0).astype(int)

fail_ids = ic_info.loc[ic_info["terminated"] == 1, "ic_id"].astype(int).tolist()
fail_rows = pd.DataFrame({
    "key": key,
    "exp": exp,
    "IC_ID": fail_ids,
})

if not os.path.exists(FAIL_INDEX_PATH):
    fail_rows.to_csv(FAIL_INDEX_PATH, index=False)
else:
    fi = pd.read_csv(FAIL_INDEX_PATH)
    if "exp" not in fi.columns:
        fi["exp"] = np.nan
    # Drop only rows where BOTH key and exp match
    fi = fi[~((fi["key"] == key) & (fi["exp"] == exp))]
    fi = pd.concat([fi, fail_rows], ignore_index=True)
    fi = fi.drop_duplicates(subset=["key", "exp", "IC_ID"])
    fi.to_csv(FAIL_INDEX_PATH, index=False)

print(f"Fail indices written to: {FAIL_INDEX_PATH} (rows for key={key}, exp={exp}: {len(fail_ids)})")
