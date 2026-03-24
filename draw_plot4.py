import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

plt.style.use(['science'])

# Gamma values (for plotting evenly spaced x-axis)
original_gamma_values = [0.0, 0.1, 0.2, 0.3, 0.5]
evenly_spaced_x = np.arange(len(original_gamma_values))  # Assign evenly spaced values

b01_300step = [4.31, 6.13, 8.07, 11.79, 12.44]
b01_450step = [4.75, 4.04, 4.23, 3.59, 7.19]
b01_600step = [4.66, 2.83, 4.16, 4.62, 5.23]
b05_450step = [4.53, 4.09, 4.26, 4.46, 4.94]
b10_450step = [4.08, 2.99, 4.32, 4.51, 4.83]
b01_450step_mid = [5.43, 9.48, 11.02, 15.76, 17.17]

# Define colors for each line
colors = sns.color_palette("Dark2", 10)

# Plot settings
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(6, 4))

# Increase font sizes
plt.rcParams.update({'font.size': 14})

# Plot all metrics in one plot
ax.plot(evenly_spaced_x, b01_300step, marker='o', linestyle='-', linewidth=2.0, color=colors[0], label='$\\beta$ 0.1, 300 steps\ntarget later layers')
ax.plot(evenly_spaced_x, b01_450step, marker='o', linestyle='-', linewidth=2.0, color=colors[1], label='$\\beta$ 0.1, 450 steps\ntarget later layers')
ax.plot(evenly_spaced_x, b05_450step, marker='o', linestyle='-', linewidth=2.0, color=colors[2], label='$\\beta$ 0.5, 450 steps\ntarget later layers')
ax.plot(evenly_spaced_x, b10_450step, marker='o', linestyle='-', linewidth=2.0, color=colors[4], label='$\\beta$ 1.0, 450 steps\ntarget later layers')
ax.plot(evenly_spaced_x, b01_600step, marker='o', linestyle='-', linewidth=2.0, color=colors[5], label='$\\beta$ 0.1, 600 steps\ntarget later layers')
ax.plot(evenly_spaced_x, b01_450step_mid, marker='o', linestyle='-', linewidth=2.0, color=colors[6], label='$\\beta$ 0.1, 450 steps\ntarget middle layers')
# Labels and title
ax.set_xlabel("$\\beta$", fontsize=16)
ax.set_ylabel("ASR (\%)", fontsize=16)
ax.set_title("Safety Benchmarks", fontsize=16)
ax.legend(fontsize=14, loc="upper left", bbox_to_anchor=(0.98, 1.05))
ax.set_xticks(evenly_spaced_x)
ax.set_xticklabels(original_gamma_values, fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig('ablation_gamma_combined.pdf', dpi=300, bbox_inches='tight')
