import matplotlib.pyplot as plt
from venn import venn

# Define your sets (using actual ICs or just dummy IDs for now)
sets = {
    "Contradictory": set(range(1, 47)),  # 46 ICs
    "Unstable": set([23, 24, 36, 39, 43, 44, 47, 55, 56, 58, 68, 71] +
                    [119, 124, 132, 133, 138, 144, 149, 150, 159, 170, 183,
                     188, 194, 198, 202, 210, 215, 224, 232, 235, 246, 268,
                     291, 294, 301, 312, 319, 330, 338, 341, 343, 344, 345, 346]),
    "Extreme Velocity": set([23, 24, 36, 39, 43, 44, 47, 55, 56, 58, 68, 71]),
    "Energy-Rich": set([23, 24, 36, 39, 43, 44])
}

plt.figure(figsize=(10, 8))
venn(sets)
plt.title("Critical Failure Clusters (4-Set Venn)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
