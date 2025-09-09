import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Change this to your CSV file path ----
FILE_PATH = "rate_models/eval_data/c_ppo_data_exp1.csv"

# Load data
df = pd.read_csv(FILE_PATH)

# Take the last row of each episode (ic_id)
last_rows = df.sort_values(["ic_id", "step"]).groupby("ic_id").tail(1)

# Extract rewards
rewards = last_rows["total_reward"]

# Summary metrics
print("\nReward metrics")
print("-" * 40)
print(f"Episodes       : {len(rewards)}")
print(f"Mean reward    : {rewards.mean():.3f}")
print(f"Std reward     : {rewards.std():.3f}")
print(f"Min reward     : {rewards.min():.3f}")
print(f"Median reward  : {rewards.median():.3f}")
print(f"Max reward     : {rewards.max():.3f}")
print("-" * 40)


def count_switches(group):
    if "mode" not in group.columns:
        return 0
    modes = group["mode"].values
    # consecutive changes (ignore -1 if continuous)
    switches = (modes[1:] != modes[:-1]).sum()
    return switches


switch_counts = df.groupby("ic_id").apply(count_switches)

print("\nMode switch counts per episode")
print("-" * 40)
print(switch_counts)

print("\nSummary:")
print(f"Mean switches   : {switch_counts.mean():.2f}")
print(f"Std switches    : {switch_counts.std():.2f}")
print(f"Min switches    : {switch_counts.min()}")
print(f"Median switches : {switch_counts.median()}")
print(f"Max switches    : {switch_counts.max()}")

# Get first and last rows per ic_id
first_rows = df[df["step"] == 0][["ic_id", "x", "theta", "xdot", "thetadot"]]
last_rows = df.sort_values(["ic_id", "step"]).groupby("ic_id").tail(1)[["ic_id", "terminated"]]

# Merge to know if each initial condition failed
init_info = first_rows.merge(last_rows, on="ic_id")

# Plot: blue if success, red if fail
plt.figure(figsize=(6, 5))
for _, row in init_info.iterrows():
    color = "red" if row["terminated"] == 1 else "blue"
    plt.scatter(row["x"], row["theta"], c=color, s=50, alpha=0.7)
    plt.text(row["x"], row["theta"], f"{int(row['ic_id'])}", fontsize=8, ha="right")

plt.xlabel("x [m]")
plt.ylabel("theta [rad]")
plt.title("Initial Conditions (red = fail, blue = success)")
plt.grid(True)
plt.tight_layout()
plt.show()

IC_INDEX = 3
g = df[df["ic_id"] == IC_INDEX].copy()

if g.empty:
    raise ValueError(f"No trajectory found for ic_id={IC_INDEX}")

# Detect discrete/continuous
is_discrete = ("mode" in g.columns) and (g["mode"] >= 0).any()


# --- Helper: shade background by mode ---
def shade_by_mode(ax, time, mode):
    palette = ["#1f6aac", "#50df50", "#E48648", "#c9a86b", '#f5f5ff', '#f9f9e6', '#e6f9f9']
    mode_to_color = {}
    seg_start = 0
    for i in range(1, len(mode) + 1):
        if i == len(mode) or mode.iloc[i] != mode.iloc[i - 1]:
            mval = int(mode.iloc[seg_start])
            if mval not in mode_to_color:
                mode_to_color[mval] = palette[len(mode_to_color) % len(palette)]
            color = mode_to_color[mval]
            ax.axvspan(time.iloc[seg_start], time.iloc[i - 1], facecolor=color, alpha=0.25, linewidth=0)
            seg_start = i
    from matplotlib.patches import Patch
    return [Patch(facecolor=c, alpha=0.25, label=f"Mode {m}") for m, c in sorted(mode_to_color.items())]


# --- Plot θ(t) and x(t) ---
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

ax1.plot(g["time"], g["x"], label="x [m]")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("x [m]")
ax1.grid(True)

ax2.plot(g["time"], g["xdot"], label="dx [m/s]")
ax2.set_ylabel("dx [m/s]")
ax2.set_title(f"Episode for ic_id={IC_INDEX}")
ax2.grid(True)

ax3.plot(g["time"], g["theta"], label="θ [rad]")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("θ [rad]")
ax3.grid(True)

ax4.plot(g["time"], g["thetadot"], label="dθ [rad/s]")
ax4.set_ylabel("dθ [rad/s]")
ax4.set_title(f"Episode for ic_id={IC_INDEX}")
ax4.grid(True)

if is_discrete:
    # Shade background for each mode
    handles1 = shade_by_mode(ax1, g["time"], g["mode"])
    handles2 = shade_by_mode(ax2, g["time"], g["mode"])

    # Also plot mode line on θ(t)
    # ax1b = ax1.twinx()
    # ax1b.step(g["time"], g["mode"], where="post", alpha=0.7)
    # ax1b.set_ylabel("Mode")

    # Legends
    l1, lbl1 = ax1.get_legend_handles_labels()
    ax1.legend(l1 + handles1, lbl1 + [h.get_label() for h in handles1], loc="best")

    l2, lbl2 = ax2.get_legend_handles_labels()
    ax2.legend(l2 + handles2, lbl2 + [h.get_label() for h in handles2], loc="best")
else:
    ax1.legend(loc="best")
    ax2.legend(loc="best")

plt.tight_layout()
plt.show()
