import matplotlib.pyplot as plt
from brokenaxes import brokenaxes



# [ 'EF9DAF', '2ECC71','CBCE91', '195190','FF5733', '3030FF'])

#'EF9DAF', '2ECC71', 'D04C3C', 'E3B448', '1B6535', 'B1624E', 'BDC3C7',


# D81B60 8FBBD9 5C8CA5 A7C38D A0A3C4 9AB8B7 7AAE7A

# 0073E6 C4C4C4 9E9E9E D0B080 B88A8A 9DBE9D 7FBF7F

# D81B60 56B4E9 009E73 F0E442 E58606 8172B3 8FB359

# # ['EF9DAF', '2ECC71', 'D04C3C', 'E3B448', '1B6535', 'B1624E', 'BDC3C7', 'CBCE91', '195190','FF5733', '3030FF']

data = {
    # "Full Training": {"color": "#EF9DAF", "points": [(56.3, 79.66)], "marker": "*"},  # gold
    # "InfoBatch": {"color": "#2ECC71", "points": [(34.95, 75.04)], "marker": "o"},
    # "DivBS": {"color": "#CBCE91", "points": [(34.95, 75.29)], "marker": "o"},
    # "COINCIDE": {"color": "#BDC3C7", "points": [(34.95, 72.19)], "marker": "o"},  # indigo
    # "ADAPT-∞": {"color": "#FFA500", "points": [(43.446, 73.46)], "marker": "o"},  # orange
    # "TIVE": {"color": "#FF5733", "points": [(56.3, 72.75)], "marker": "o"},  # olive
    # "OASIS (Ours)": {"color": "#3030FF", "points": [(34.95, 77.42)], "marker": "X"},  # dodger blue
    
    
    # "DITTO": {"color": "#8FBBD9", "points": [(1.0, 64.09)], "marker": "o"},  # gold
    # "FedSIM": {"color": "#5C8CA5", "points": [(1.0, 67.05)], "marker": "o"},  # gold
    
    # "PerADA": {"color": "#A7C38D", "points": [(1.0, 63.78)], "marker": "o"},  # gold
    # # "TAKFL": {"color": "#D04C3C", "points": [(1.0, 67.02)], "marker": "o"},  # gold
    # "FedDPA": {"color": "#A0A3C4", "points": [(1.0, 66.59)], "marker": "o"},  # gold
    # "FedDAT": {"color": "#9AB8B7", "points": [(1.0, 61.95)], "marker": "o"},  # gold
    # "FedMKT": {"color": "#7AAE7A", "points": [(1.0, 65.20)], "marker": "o"},  # gold
    # "HEFT (Ours)": {"color": "#D81B60", "points": [(1.0, 70.40)], "marker": "o"},
    
    
    # "DITTO_": {"color": "#EF9DAF", "points": [(2.0, 68.23)], "marker": "X"},  # gold
    # "FedSIM_": {"color": "#FF5733", "points": [(1.49, 68.23)], "marker": "X"},  # gold
    # "PerADA_": {"color": "#2ECC71", "points": [(2.29, 67.40)], "marker": "X"},  # gold
    # "TAKFL_": {"color": "#BDC3C7", "points": [(1.29, 67.72)], "marker": "X"},  # gold
    # "FedDPA_": {"color": "#E3B448", "points": [(2.05, 68.78)], "marker": "X"},  # gold
    # "FedDAT_": {"color": "#1B6535", "points": [(2.56, 69.81)], "marker": "X"},  # gold
    # "FedMKT_": {"color": "#B1624E", "points": [(1.57, 68.06)], "marker": "X"},  # gold
    # "HEFT_ (Ours)": {"color": "#3030FF", "points": [(1.09, 70.80)], "marker": "X"},
    
    # "SFT": {"color": "#000000", "points": [(1.0, 67.57)], "marker": "*"},  # gold
    # "FedIT": {"color": "#2ECC71", "points": [(1.0, 67.62)], "marker": "*"},  # gold
    
    "DITTO_": {"color": "#CBCE91", "points": [(2.0, 66.28)], "marker": "X"},  # gold 195190 CBCE91
    "FedSIM_": {"color": "#FF5733", "points": [(1.49, 65.66)], "marker": "X"},  # gold
    "PerADA_": {"color": "#2ECC71", "points": [(2.29, 65.69)], "marker": "X"},  # gold
    "TAKFL_": {"color": "#BDC3C7", "points": [(1.29, 64.99)], "marker": "X"},  # gold
    "FedDPA_": {"color": "#E3B448", "points": [(2.05, 68.09)], "marker": "X"},  # gold
    "FedDAT_": {"color": "#D04C3C", "points": [(2.56, 67.75)], "marker": "X"},  # gold 1B6535
    "FedMKT_": {"color": "#B1624E", "points": [(1.57, 65.98)], "marker": "X"},  # gold 
    "HEFT_ (Ours)": {"color": "#3030FF", "points": [(1.09, 68.50)], "marker": "X"},
    
    "SFT": {"color": "#000000", "points": [(1.0, 65.79)], "marker": "*"},  # gold
}
# ['EF9DAF', '2ECC71', 'D04C3C', 'E3B448', '1B6535', 'B1624E', 'BDC3C7', 'CBCE91', '195190','FF5733', '3030FF']

fig = plt.figure(figsize=(7, 3.5))
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
# bax = brokenaxes(xlims=((33, 45), (55, 58)), hspace=.05)  # break from ~45 to ~55
bax = brokenaxes()

for method, values in data.items():
    x, y = zip(*values["points"])
    if method == 'HEFT_ (Ours)':
        bax.scatter(x, y, color=values["color"], label=method, marker=values["marker"],
                s=250, edgecolors='black')
    else:
        bax.scatter(x, y, color=values["color"], label=method, marker=values["marker"],
                s=200, edgecolors='black')

# bax.set_xlabel("TFLOPS", fontsize=18)
bax.set_xlabel("Relative FLOPs to SFT", fontsize=20, labelpad=30)
bax.set_ylabel(r"$A_{last} (\%)$", fontsize=20, labelpad=40)
# bax.axs[0].legend(loc='upper right', bbox_to_anchor=(1.55, 0.85), frameon=True, fontsize=12)
# fig.legend(
#     handles=bax.axs[0].get_legend_handles_labels()[0],
#     labels=bax.axs[0].get_legend_handles_labels()[1],
#     loc='upper center',
#     bbox_to_anchor=(0.06, 0.76),
#     ncol=2,
#     fontsize=12,
#     frameon=True
# )

# from matplotlib.lines import Line2D

# # ---------- A. marker-shape legend ----------
# shape_handles = [
#     Line2D([], [], marker='o', color='w', markeredgecolor='k',
#            markersize=10, label='With FLOPS-constrained'),
#     Line2D([], [], marker='x', color='w', markeredgecolor='k',
#            markersize=10, lw=0, label='No FLOPS-constrained'),
#     Line2D([], [], marker='*', color='w', markeredgecolor='k',
#            markerfacecolor='k', markersize=14, label='SFT baseline'),
# ]
# first_legend = bax.axs[0].legend(
#     handles=shape_handles, loc='upper center', frameon=True, fontsize=11, bbox_to_anchor=(0.94,0.24)
# )
# bax.axs[0].add_artist(first_legend)   # keep it when the second legend is added

# ---------- B. colour legend ----------
# method_handles = [
#     Line2D([], [], marker='o', color='#CBCE91', markersize=10, label='DITTO'),
#     Line2D([], [], marker='o', color='#BDC3C7', markersize=10, label='FedSIM'),
#     Line2D([], [], marker='o', color='#FFA500', markersize=10, label='PerADA'),
#     Line2D([], [], marker='o', color='#D04C3C', markersize=10, label='FedDPA'),
#     Line2D([], [], marker='o', color='#EF9DAF', markersize=10, label='FedDAT'),
#     Line2D([], [], marker='o', color='#2ECC71', markersize=10, label='FedMKT'),
#     Line2D([], [], marker='o', color='#3030FF', markersize=10, label='HEFT (Ours)'),
# ]
# bax.axs[0].legend(
#     handles=method_handles,
#     loc='center left',      # adjust as needed
#     bbox_to_anchor=(0.85, 0.5),
#     frameon=True,
#     fontsize=11,
#     title='Methods'
# )

bax.grid(True, linestyle='--', alpha=0.6)

# plt.suptitle("Accuracy and Relative FLOPs", fontsize=20)
plt.tight_layout()
# plt.savefig('test.png')
plt.savefig('test2.png', dpi=300, bbox_inches='tight')
# plt.show()