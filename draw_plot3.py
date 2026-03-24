import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science'])

# Gamma values (for plotting evenly spaced x-axis)
original_beta_values = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
evenly_spaced_x = np.arange(len(original_beta_values))  # Assign evenly spaced values

g01_300step = [8.12, 6.13, 5.18, 4.69, 3.45, 3.66]
g01_450step = [6.55, 4.04, 4.22, 4.03, 4.09, 3.32]
g03_450step = [8.76, 3.59, 4.46, 4.46, 4.46, 4.72]
g05_450step = [9.92, 7.19, 5.71, 5.35, 4.94, 4.83]
g03_600step = [8.66, 4.62, 4.09, 5.66, 4.16, 4.05]
g01_300step_mid = [15.22, 11.25, 11.25, 10.58, 9.93, 10.15]
g03_450step_mid = [17.64, 15.76, 10.97, 9.76, 9.95, 9.51]

# Define colors for each line
colors = sns.color_palette("Dark2", 10)

# Plot settings
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))

# Increase font sizes
plt.rcParams.update({'font.size': 14})

# Plot all metrics in one plot
ax.plot(evenly_spaced_x, g01_300step, marker='o', linestyle='-', linewidth=2.0, color=colors[0], label='$\gamma$ 0.1, 300 steps\ntarget later layers')
ax.plot(evenly_spaced_x, g01_450step, marker='o', linestyle='-', linewidth=2.0, color=colors[1], label='$\gamma$ 0.1, 450 steps\ntarget later layers')
ax.plot(evenly_spaced_x, g03_450step, marker='o', linestyle='-', linewidth=2.0, color=colors[2], label='$\gamma$ 0.3, 450 steps\ntarget later layers')
ax.plot(evenly_spaced_x, g05_450step, marker='o', linestyle='-', linewidth=2.0, color=colors[3], label='$\gamma$ 0.5, 450 steps\ntarget later layers')
ax.plot(evenly_spaced_x, g03_600step, marker='o', linestyle='-', linewidth=2.0, color=colors[4], label='$\gamma$ 0.3, 600 steps\ntarget later layers')
ax.plot(evenly_spaced_x, g01_300step_mid, marker='o', linestyle='-', linewidth=2.0, color=colors[5], label='$\gamma$ 0.1, 300 steps\ntarget middle layers')
ax.plot(evenly_spaced_x, g03_450step_mid, marker='o', linestyle='-', linewidth=2.0, color=colors[6], label='$\gamma$ 0.3, 450 steps\ntarget middle layers')
# Labels and title
ax.set_xlabel("$\\beta$", fontsize=16)
ax.set_ylabel("ASR (\%)", fontsize=16)
ax.set_title("Safety Benchmarks", fontsize=16)
ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(0.98, 1.05))
ax.set_xticks(evenly_spaced_x)
ax.set_xticklabels(original_beta_values, fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig('ablation_beta_combined.pdf', dpi=300, bbox_inches='tight')
