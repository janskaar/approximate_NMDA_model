import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cbcolors = sns.color_palette("colorblind").as_hex()

plt.rcParams["font.size"] = 8

nest_data = np.loadtxt(os.path.join("wang_benchmark_nest.csv"), delimiter=",", skiprows=1)
brian_data = np.loadtxt(os.path.join("wang_benchmark_brian.csv"), delimiter=",", skiprows=1)

exact_times = nest_data[:,1]
approx_times = nest_data[:,0]
brian_times = brian_data[:,0]
nest_num = nest_data[:,-1] * 2000
brian_num = brian_data[:,-1] * 2000

## 

fig, ax = plt.subplots(1)
fig.set_size_inches([7.6 / 2.54, 4 / 2.54])
fig.subplots_adjust(left=0.15, bottom=0.21, top=0.98, right=0.96)

ax.plot(nest_num, exact_times, color=cbcolors[1], label="Exact model (NEST, RKF45)")
ax.scatter(nest_num, exact_times, color=cbcolors[1], marker="x", s=20)
ax.plot(nest_num, approx_times, color=cbcolors[0], label="Approximate model (NEST, RKF45)")
ax.scatter(nest_num, approx_times, color=cbcolors[0], marker="x", s=20)
ax.plot(brian_num, brian_times, color=cbcolors[4], label="Wimmer/Stimberg (Brian2, RK4)")
ax.scatter(brian_num, brian_times, color=cbcolors[4], marker="x", s=20)
ax.semilogy()
ax.set_xlim(0, 16500)
ax.set_xticks([0, 4000, 8000, 12000, 16000])

ax.set_ylabel("Simulation time [s]", labelpad=2)
ax.set_xlabel("Network size", labelpad=2)
fig.savefig("figure5.pdf")
plt.show()

