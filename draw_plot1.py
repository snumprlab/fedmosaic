import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
plt.style.use(['science'])

# Beta values (for plotting evenly spaced x-axis)
original_beta_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
evenly_spaced_x = np.arange(len(original_beta_values))  # Assign evenly spaced values

# Safety metrics
harmbench = [7.81, 4.06, 0.31, 2.50, 3.13, 3.44, 3.13, 3.44]
wildguardtest = [13.48, 7.74, 7.34, 6.94, 6.94, 7.08, 6.94, 7.08]
PAP = [5, 6.56, 3.12, 3.94, 3.31, 3.63, 3.31, 3.63]

# Over-refusal and general capability metrics
xstest = [82.67, 86.67, 84.11, 85.56, 84.89, 86.22, 85.11, 87.11]
wildjailbreak_benign = [82.00, 94.00, 89.20, 92.00, 93.20, 93.20, 93.20, 92.80]
mtbench = [76.4, 76.9, 77.1, 75.9, 76.2, 78.1, 77.0, 78.1]
mmlu = [65.90, 66.12, 65.08, 66.20, 66.07, 66.08, 66.07, 66.08]

# Define colors for each line
colors = sns.color_palette("tab10", 10)

# Plot settings
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

# Increase font sizes
plt.rcParams.update({'font.size': 14})

# Safety benchmarks plot
lines = []
labels = []

l1, = axes[0].plot(evenly_spaced_x, harmbench, marker='o', linestyle='-', linewidth=2.0, color=colors[0], label='Harmbench (↓)')
l2, = axes[0].plot(evenly_spaced_x, wildguardtest, marker='o', linestyle='-', linewidth=2.0, color=colors[1], label='WildguardTest (↓)')
l3, = axes[0].plot(evenly_spaced_x, PAP, marker='o', linestyle='-', linewidth=2.0, color=colors[6], label='PAP (↓)')
axes[0].set_xlabel("$\\beta$", fontsize=16)
axes[0].set_ylabel("ASR (\%)", fontsize=16)
axes[0].set_title("Safety Benchmarks", fontsize=16)
axes[0].set_xticks(evenly_spaced_x)
axes[0].set_xticklabels(original_beta_values, fontsize=12)

# Collect lines and labels
lines.extend([l1, l2, l3])
labels.extend([l.get_label() for l in [l1, l2, l3]])

# Other benchmarks plot
l4, = axes[1].plot(evenly_spaced_x, xstest, marker='s', linestyle='--', linewidth=2.0, color=colors[2], label='XSTest (↑)')
l5, = axes[1].plot(evenly_spaced_x, wildjailbreak_benign, marker='s', linestyle='--', linewidth=2.0, color=colors[3], label='WildJailbreak: \nBenign (↑)')
l6, = axes[1].plot(evenly_spaced_x, mtbench, marker='D', linestyle=':', linewidth=2.0, color=colors[4], label='MT-Bench (↑)')
l7, = axes[1].plot(evenly_spaced_x, mmlu, marker='D', linestyle=':', linewidth=2.0, color=colors[5], label='MMLU (↑)')
axes[1].set_xlabel("$\\beta$", fontsize=16)
axes[1].set_ylabel("Score (\%)", fontsize=16)
axes[1].set_title("Over-Refusal \& General Capability", fontsize=16)
axes[1].set_xticks(evenly_spaced_x)
axes[1].set_xticklabels(original_beta_values, fontsize=12)

# Collect lines and labels
lines.extend([l4, l5, l6, l7])
labels.extend([l.get_label() for l in [l4, l5, l6, l7]])

# Add single legend at the bottom
fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=16, ncol=3)

# Adjust layout to make space for the legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.10)  # Adjust bottom space for the legend

# Show and save figure
plt.show()
plt.savefig('ablation_beta.pdf', dpi=300, bbox_inches='tight')
