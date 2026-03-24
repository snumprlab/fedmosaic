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
height = 4
k = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
# k = [32, 64, 128, 256, 512, 1024]
# [r"$\frac{1}{16}$",r"$\frac{1}{8}$",r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$", r"$4$"]


# disjoint_auc = [71.02, 77.36, 78.28, 78.32, 78.77, 78.56, 78.65, 78.83, 78.75]
# disjoint_last = [63.99, 65.73, 66.34, 66.55, 66.20, 66.33, 66.81, 65.95, 66.57]
# disjoint_auc_std = [1.16, 0.79, 0.79, 0.93, 0.48, 0.86, 0.46, 0.71, 0.87]
# disjoint_last_std = [1.43, 1.51, 0.62, 1.35, 1.05, 0.58, 0.68, 1.14, 1.15]

# last
continuous_auc = [66.81, 67.78, 67.90, 67.86, 67.68, 67.28, 67.30]
continuous_auc_std = [0.81, 0.06, 0.97, 0.51, 0.36, 0.45, 0.39]

# avg
continuous_last = [58.94, 59.37, 59.51, 59.63, 59.54, 59.41, 59.39]
continuous_last_std = [0.31, 0.55, 0.81, 0.15, 0.47, 0.29, 0.36]

# other last
continuous_auc2 = [50.42, 50.91, 51.02, 51.36, 51.37, 51.34, 51.30]
continuous_auc_std2 = [0.82, 0.05, 0.02, 0.04, 0.01, 0.07, 0.12]

# other avg
continuous_last2 = [48.79, 49.13, 49.28, 49.49, 49.49, 49.42, 49.35]
continuous_last_std2 = [0.31, 0.10, 0.19, 0.06, 0.09, 0.03, 0.05]


fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(width, height))

# breakpoint()

ax1[0].plot(k, continuous_auc, label="A_last")
ax1[0].fill_between(k, np.array(continuous_auc) - np.array(continuous_auc_std), continuous_auc + np.array(continuous_auc_std), color='b', alpha=.15)
ax1[0].set_xlabel(r"Temperature $\tau$", fontsize=11)
ax1[0].set_ylabel(r'Self $A_{last}$', fontsize=12)
ax1[0].set_xscale('log')
ax1[0].set_xticks(k, [r"$\frac{1}{16}$",r"$\frac{1}{8}$",r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$", r"$4$"], fontsize=9)
ax1[0].set_yticks(range(66, 70, 1), range(66, 70, 1), fontsize=9)
ax1[0].yaxis.grid()


ax1[1].plot(k, continuous_last, label="A_AUC")
ax1[1].fill_between(k, np.array(continuous_last) - np.array(continuous_last_std), continuous_last + np.array(continuous_last_std), color='b', alpha=.15)
ax1[1].set_xlabel(r"Temperature $\tau$", fontsize=11)
ax1[1].set_ylabel(r'Self $A_\text{AUC}$', fontsize=12)
ax1[1].set_xscale('log')
ax1[1].set_xticks(k, [r"$\frac{1}{16}$",r"$\frac{1}{8}$",r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$", r"$4$"], fontsize=9)
ax1[1].set_yticks(range(58, 62, 1), range(58, 62, 1), fontsize=9)
ax1[1].yaxis.grid()

ax2[0].plot(k, continuous_auc2, label="A_last")
ax2[0].fill_between(k, np.array(continuous_auc2) - np.array(continuous_auc_std2), continuous_auc2 + np.array(continuous_auc_std2), color='b', alpha=.15)
ax2[0].set_xlabel(r"Temperature $\tau$", fontsize=11)
ax2[0].set_ylabel(r'Others $A_{last}$', fontsize=12)
ax2[0].set_xscale('log')
ax2[0].set_xticks(k, [r"$\frac{1}{16}$",r"$\frac{1}{8}$",r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$", r"$4$"], fontsize=9)
ax2[0].set_yticks(range(49, 53, 1), range(49, 53, 1), fontsize=9)
ax2[0].yaxis.grid()


ax2[1].plot(k, continuous_last2, label="A_AUC")
ax2[1].fill_between(k, np.array(continuous_last2) - np.array(continuous_last_std2), continuous_last2 + np.array(continuous_last_std2), color='b', alpha=.15)
ax2[1].set_xlabel(r"Temperature $\tau$", fontsize=11)
ax2[1].set_ylabel(r'Others $A_\text{AUC}$', fontsize=12)
ax2[1].set_xscale('log')
ax2[1].set_xticks(k, [r"$\frac{1}{16}$",r"$\frac{1}{8}$",r"$\frac{1}{4}$",r"$\frac{1}{2}$",r"$1$",r"$2$", r"$4$"], fontsize=9)
ax2[1].set_yticks(range(48, 51, 1), range(48, 51, 1), fontsize=9)
ax2[1].yaxis.grid()

# ax2.plot(k, continuous_auc, label="A_auc")
# ax2.plot(k, continuous_last, label="A_last")
plt.suptitle(r"Effect of Temperature $\tau$ of RELA", fontsize=13)
plt.tight_layout(pad=0.3, h_pad=-0.01, w_pad=0.8)
plt.savefig("effect_tau.pdf")