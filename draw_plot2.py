import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science'])

# Gamma values (for plotting evenly spaced x-axis)
original_gamma_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
evenly_spaced_x = np.arange(len(original_gamma_values))  # Assign evenly spaced values

# Safety metrics
harmbench = [0.0, 1.88, 1.88, 0.31, 4.69, 3.75]
wildguardtest = [13.62, 6.81, 7.74, 7.34, 9.35, 12.95]
PAP = [0.625, 3.44, 3.06, 3.12, 7.19, 6.88]

# Over-refusal and general capability metrics
xstest = [45.11, 84.78, 84.44, 84.11, 83.67, 84.89]
wildjailbreak_benign = [20.80, 91.40, 92.40, 89.20, 89.20, 88.00]
mtbench = [10.0, 77.20, 76.90, 77.10, 77.20, 76.60]
mmlu = [25.38, 65.06, 66.07, 65.08, 66.08, 66.23]

# Define colors for each line
colors = sns.color_palette("tab10", 10)

# Plot settings
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Increase font sizes
plt.rcParams.update({'font.size': 14})

# Safety benchmarks plot
lines = []
labels = []

l1, = axes[0].plot(evenly_spaced_x, harmbench, marker='o', linestyle='-', linewidth=2.0, color=colors[0], label='Harmbench (↓)')
l2, = axes[0].plot(evenly_spaced_x, wildguardtest, marker='o', linestyle='-', linewidth=2.0, color=colors[1], label='WildguardTest (↓)')
l3, = axes[0].plot(evenly_spaced_x, PAP, marker='o', linestyle='-', linewidth=2.0, color=colors[6], label='PAP (↓)')
axes[0].set_xlabel("$\\gamma$", fontsize=16)
axes[0].set_ylabel("ASR (\%)", fontsize=16)
axes[0].set_title("Safety Benchmarks", fontsize=16)
axes[0].set_xticks(evenly_spaced_x)
axes[0].set_xticklabels(original_gamma_values, fontsize=12)

# Collect lines and labels
lines.extend([l1, l2, l3])
labels.extend([l.get_label() for l in [l1, l2, l3]])

# Other benchmarks plot
l4, = axes[1].plot(evenly_spaced_x, xstest, marker='s', linestyle='--', linewidth=2.0, color=colors[2], label='XSTest (↑)')
l5, = axes[1].plot(evenly_spaced_x, wildjailbreak_benign, marker='s', linestyle='--', linewidth=2.0, color=colors[3], label='WildJailbreak: \nBenign (↑)')
l6, = axes[1].plot(evenly_spaced_x, mtbench, marker='D', linestyle=':', linewidth=2.0, color=colors[4], label='MT-Bench (↑)')
l7, = axes[1].plot(evenly_spaced_x, mmlu, marker='D', linestyle=':', linewidth=2.0, color=colors[5], label='MMLU (↑)')
axes[1].set_xlabel("$\\gamma$", fontsize=16)
axes[1].set_ylabel("Score (\%)", fontsize=16)
axes[1].set_title("Over-Refusal \& General Capability", fontsize=16)
axes[1].set_xticks(evenly_spaced_x)
axes[1].set_xticklabels(original_gamma_values, fontsize=12)

# Collect lines and labels
lines.extend([l4, l5, l6, l7])
labels.extend([l.get_label() for l in [l4, l5, l6, l7]])

# Add single legend at the bottom
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=14, ncol=4)

# Adjust layout to make space for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.10)  # Adjust bottom space for the legend

# Show and save figure
plt.show()
plt.savefig('ablation_gamma.pdf', dpi=300, bbox_inches='tight')
