import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib as mpl

mpl.rcParams.update({
    'font.size'           : 9.0        ,
    'font.sans-serif'     : 'Arial'    ,
    'xtick.major.size'    : 2.0          ,
    'xtick.minor.size'    : 0        ,
    'xtick.major.width'   : 0.75       ,
    'xtick.minor.width'   : 0.75       ,
    'xtick.labelsize'     : 7.0        ,
    'xtick.direction'     : 'in'       ,
    'xtick.top'           : True       ,
    'ytick.major.size'    : 2          ,
    'ytick.minor.size'    : 1.5        ,
    'ytick.major.width'   : 0.75       ,
    'ytick.minor.width'   : 0.75       ,
    'ytick.labelsize'     : 7.0        ,
    'ytick.direction'     : 'in'       ,
    'xtick.major.pad'     : 1.5          ,
    'xtick.minor.pad'     : 1.5          ,
    'ytick.major.pad'     : 1.5          ,
    'ytick.minor.pad'     : 1.5          ,
    'ytick.right'         : True       ,
    'savefig.dpi'         : 600        ,
    'savefig.transparent' : True       ,
    'axes.linewidth'      : 0.75       ,
    'lines.linewidth'     : 1.0        ,
    'axes.prop_cycle'     : mpl.cycler('color', ['195190', 'DD4132', 'E3B448', '1B6535', 'B1624E', '595959', '724685', 'FF8C00', 'CBCE91', '3A6B35',
                                                'EF9DAF'])
})


width = 6.0
# height = width * 0.5
height = 2
# k = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
k = [20, 40, 60, 80, 100]

# disjoint_auc = [71.02, 77.36, 78.28, 78.32, 78.77, 78.56, 78.65, 78.83, 78.75]
# disjoint_last = [63.99, 65.73, 66.34, 66.55, 66.20, 66.33, 66.81, 65.95, 66.57]
# disjoint_auc_std = [1.16, 0.79, 0.79, 0.93, 0.48, 0.86, 0.46, 0.71, 0.87]
# disjoint_last_std = [1.43, 1.51, 0.62, 1.35, 1.05, 0.58, 0.68, 1.14, 1.15]

# last
continuous_auc = [67.83, 67.77, 67.86, 67.71, 67.42]
continuous_auc_std = [0.36, 0.09, 0.50, 0.11, 0.90]

# avg
continuous_last = [59.80, 59.50, 59.83, 59.29, 59.23]
continuous_last_std = [0.21, 0.82, 0.15, 0.76, 0.42]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width, height))

ax1.plot(k, continuous_auc, label="A_last")
ax1.fill_between(k, np.array(continuous_auc) - np.array(continuous_auc_std), continuous_auc + np.array(continuous_auc_std), color='b', alpha=.15)
ax1.set_xlabel(r"Gradient sampling ratio (%)", fontsize=11)
ax1.set_ylabel(r'$A_{last}$', fontsize=12)
# ax1.set_xscale('log')
ax1.set_xticks(k, [r"$100$",r"$80$",r"$60$",r"$40$",r"$20$"], fontsize=9)
ax1.set_yticks(range(66, 69, 1), range(66, 69, 1), fontsize=9)
ax1.yaxis.grid()


ax2.plot(k, continuous_last, label="A_AUC")
ax2.fill_between(k, np.array(continuous_last) - np.array(continuous_last_std), continuous_last + np.array(continuous_last_std), color='b', alpha=.15)
ax2.set_xlabel(r"Gradient sampling ratio (%)", fontsize=11)
ax2.set_ylabel(r'$A_\text{AUC}$', fontsize=12)
# ax2.set_xscale('log')
ax2.set_xticks(k, [r"$100$",r"$80$",r"$60$",r"$40$",r"$20$",], fontsize=9)
ax2.set_yticks(range(58, 62, 1), range(58, 62, 1), fontsize=9)
ax2.yaxis.grid()

# ax2.plot(k, continuous_auc, label="A_auc")
# ax2.plot(k, continuous_last, label="A_last")
plt.suptitle("Effect of Gradient sampling ratio", fontsize=13)
plt.tight_layout(pad=0.3, h_pad=-0.01, w_pad=0.8)
plt.savefig("effect_gradient_sample.pdf")