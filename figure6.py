import os, h5py, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
cbcolors = sns.color_palette("colorblind").as_hex()

plt.rcParams["font.size"] = 8

with open("sparse_adjusted_results_3.pkl", "rb") as f:
    res = pickle.load(f)

s1_20 = res["s_NMDA_pre1"].reshape((8, 13, 33, -1))
s2_20 = res["s_NMDA_pre2"].reshape((8, 13, 33, -1))

h1_20 = res["hist1"].reshape((8, 13, 33, -1))
h2_20 = res["hist2"].reshape((8, 13, 33, -1))


with open("sparse_adjusted_results_5.pkl", "rb") as f:
    res = pickle.load(f)

s1_50 = res["s_NMDA_pre1"].reshape((8, 13, 21, -1))
s2_50 = res["s_NMDA_pre2"].reshape((8, 13, 21, -1))

h1_50 = res["hist1"].reshape((8, 13, 21, -1))
h2_50 = res["hist2"].reshape((8, 13, 21, -1))

with open("varying_epsilon_adjusted_weight_results_1.pkl", "rb") as f:
    res = pickle.load(f)

s1_varying_eps = res["s_NMDA_pre1"].reshape((8, 11, -1))
s2_varying_eps = res["s_NMDA_pre2"].reshape((8, 11, -1))

##

s1_20_mean = s1_20[...,4000:].mean((0, -1))
s2_20_mean = s2_20[...,4000:].mean((0, -1))

s1_50_mean = s1_50[...,4000:].mean((0, -1))
s2_50_mean = s2_50[...,4000:].mean((0, -1))

eps_inh_20 = np.linspace(0.2, 0.5, 13)
eps_sel_20 = np.linspace(0.2, 1.0, 33)

eps_inh_50 = np.linspace(0.5, 0.8, 13)
eps_sel_50 = np.linspace(0.5, 1.0, 21)

XX_20, YY_20 = np.meshgrid(eps_inh_20, eps_sel_20)
XX_50, YY_50 = np.meshgrid(eps_inh_50, eps_sel_50)

fig, ax = plt.subplots(ncols=2, nrows=3, height_ratios=[1, 1, 1.])#, sharex="row", sharey="row")
fig.set_size_inches([6,4.5])
fig.subplots_adjust(left=0.08, right=0.9, bottom=0.07, top=0.95, wspace=0.14, hspace=0.6)

ax[0,0].plot(s1_varying_eps[:,10,:].mean(0), cbcolors[0], lw=0.5, label="$S_\mathrm{A}(t)$")
ax[0,0].plot(s1_varying_eps[:,10,:].T, cbcolors[0], lw=0.5)
ax[0,0].plot(s2_varying_eps[:,10,:].mean(0), cbcolors[1], lw=0.5, label="$S_\mathrm{B}(t)$")
ax[0,0].plot(s2_varying_eps[:,10,:].T, cbcolors[1], lw=0.5)
ax[0,0].set_xlim(1000, 6000)
ax[0,0].sharex(ax[0,1])
ax[0,0].sharey(ax[0,1])

ax[0,1].plot(s1_varying_eps[:,5,:].mean(0), cbcolors[0], lw=0.5)
ax[0,1].plot(s1_varying_eps[:,5,:].T, cbcolors[0], lw=0.5)
ax[0,1].plot(s2_varying_eps[:,5,:].mean(0), "C1", lw=0.5)
ax[0,1].plot(s2_varying_eps[:,5,:].T, "C1", lw=0.5)
ax[0,1].set_xlim(1000, 6000)

ax[0,0].set_xlabel("Time [ms]", labelpad=0.5)
ax[0,1].set_xlabel("Time [ms]", labelpad=0.5)
# ax[0,0].legend(loc=(0.02,0.17), framealpha=0.5, handlelength=1, handletextpad=0.3)

ax[0,0].set_title("100% connectivity", pad=2, fontsize=8)
ax[0,1].set_title("95% connectivity", pad=2, fontsize=8)

ind1_inh = 3
ind1_sel = 15
ind2_inh = 5
ind2_sel = 23

ax[1,0].plot(s1_20[:,ind1_inh,ind1_sel,:].mean(0), color=cbcolors[0], lw=0.5)
ax[1,0].plot(s1_20[:,ind1_inh,ind1_sel,:].T, color=cbcolors[0], lw=0.5)
ax[1,0].plot(s2_20[:,ind1_inh,ind1_sel,:].mean(0), color=cbcolors[1], lw=0.5)
ax[1,0].plot(s2_20[:,ind1_inh,ind1_sel,:].T, color=cbcolors[1], lw=0.5)
ax[1,0].sharex(ax[1,1])
ax[1,0].sharey(ax[1,1])

ax[1,1].plot(s1_20[:,ind2_inh,ind2_sel,:].mean(0), color=cbcolors[0], lw=0.5, label="$s_1$")
ax[1,1].plot(s1_20[:,ind2_inh,ind2_sel,:].T, color=cbcolors[0], lw=0.5)
ax[1,1].plot(s2_20[:,ind2_inh,ind2_sel,:].mean(0), color=cbcolors[1], lw=0.5, label="$s_2$")
ax[1,1].plot(s2_20[:,ind2_inh,ind2_sel,:].T, color=cbcolors[1], lw=0.5)

ax[1,0].set_title("20% base connectivity, adjusted", pad=2, fontsize=8)
ax[1,1].set_title("20% base connectivity, adjusted", pad=2, fontsize=8)
ax[1,0].set_xlim(1000,6000)
ax[1,0].set_xlabel("Time [ms]", labelpad=0.5)
ax[1,1].set_xlabel("Time [ms]", labelpad=0.5)

# pc = ax[2,0].pcolormesh(YY_50.T[:9], XX_50.T[:9], s1_50_mean[:9] - s2_50_mean[:9], vmin=0, vmax=1)
# pc = ax[2,1].pcolormesh(YY_20.T[:9,12:], XX_20.T[:9,12:], s1_20_mean[:9,12:] - s2_20_mean[:9,12:], vmin=0, vmax=1)
pc = ax[2,0].pcolormesh(YY_20.T[:,:], XX_20.T[:,:], s1_20_mean[:,:], vmin=0, vmax=1)
pc = ax[2,1].pcolormesh(YY_20.T[:,:], XX_20.T[:,:], s1_20_mean[:,:] - s2_20_mean[:,:], vmin=0, vmax=1)

ax[2,0].set_aspect(1.)
ax[2,1].set_aspect(1.)

ax[2,0].scatter(eps_sel_20[[ind1_sel, ind2_sel]], eps_inh_20[[ind1_inh, ind2_inh]], c=["red", "blue"], s=5)
ax[2,1].scatter(eps_sel_20[[ind1_sel, ind2_sel]], eps_inh_20[[ind1_inh, ind2_inh]], c=["red", "blue"], s=5)

ax[2,0].set_ylabel("$\mathrm{I} \\to \mathrm{E_A}, \mathrm{I} \\to \mathrm{E_B}$ conn.")
ax[2,0].set_xlabel("$\mathrm{E_A} \\to \mathrm{E_A}, \mathrm{E_B} \\to \mathrm{E_B}$ conn.", labelpad=0.5)
ax[2,1].set_xlabel("$\mathrm{E_A} \\to \mathrm{E_A}, \mathrm{E_B} \\to \mathrm{E_B}$ conn.", labelpad=0.5)

#ax[2,0].set_title("$\mathbb{E}[S_\mathrm{A} - S_\mathrm{B}]$, 50% base connectivity", pad=2, fontsize=8)
ax[2,0].set_title("$\mathbb{E}[S_\mathrm{A}(t)]$, 20% base connectivity", pad=2, fontsize=8)
ax[2,1].set_title("$\mathbb{E}[S_\mathrm{A}(t) - S_\mathrm{B}(t)]$, 20% base connectivity", pad=2, fontsize=8)

ax[2,0].set_xticks(np.linspace(0.2,1,9))
ax[2,0].set_yticks(np.linspace(0.2,0.5,4))
#ax[2,0].set_yticks(np.linspace(0.5,0.7,3))

ax[2,1].set_xticks(np.linspace(0.2,1,9))
ax[2,1].set_yticks(np.linspace(0.2,0.5,4))

cbar_ax = fig.add_axes([0.91,0.075,0.02,0.2])
plt.colorbar(pc, cax=cbar_ax)

labels = ["A", "B", "C", "D", "E", "F"]
for i in range(6):
    ax.flat[i].text(-0.05, 1.05, labels[i], fontsize=11, transform=ax.flat[i].transAxes)

fig.savefig("figure6.pdf")
plt.show()

