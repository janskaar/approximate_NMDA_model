import nest
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import seaborn as sns
cbcolors = sns.color_palette("colorblind").as_hex()
import numpy as np
import pandas as pd

plt.rcParams["font.size"] = 8

np.random.seed(123)

w_ext = 40.0
w_ex = 1.0
w_in = 15.0

params = {
    "tau_AMPA": 2.0,
    "tau_GABA": 5.0,
    "tau_rise_NMDA": 2.0,
    "tau_decay_NMDA": 100.0,
    "conc_Mg2": 1.0,
    "E_ex": 0.0,
    "E_in": -70.0,
    "E_L": -70.0,
    "V_th": -55.0,
    "C_m": 500.0,
    "g_L": 25.0,
    "V_reset": -70.0,
    "alpha": 0.5,
    "t_ref": 2.0,
}

def do_sim(n_pre, r_pre, weight, t_sim=1000, show=False, seed=955):
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.rng_seed = seed
    nest.local_num_threads = 4

    sg = nest.Create("poisson_generator", params={"rate": r_pre})
    parrot0 = nest.Create("parrot_neuron", n_pre)
    parrot1 = nest.Create("parrot_neuron", n_pre)
    parrot2 = nest.Create("parrot_neuron", n_pre)

    pp = params.copy()
    pp["t_ref"] = 0.2
    pp["V_th"] = -69
    pre = nest.Create("iaf_bw_2001", n=n_pre, params=pp)
    post_app = nest.Create("iaf_bw_2001", params=params)
    post_exa = nest.Create("iaf_bw_2001_exact", params=params)
    rec_pre, rec_post_app, rec_post_exa = nest.Create("spike_recorder", n=3)
    vm_app, vm_exa = nest.Create("voltmeter", params={"interval": nest.resolution}, n=2)
    vm_in = nest.Create("voltmeter", params={"interval": nest.resolution}, n=n_pre)

    nest.Connect(sg, parrot0)
    nest.Connect(parrot0, parrot1, conn_spec="one_to_one")
    nest.Connect(parrot0, parrot2, conn_spec="one_to_one")
    nest.Connect(parrot1, pre, conn_spec="one_to_one", syn_spec={"receptor_type": 1, "weight": 1000., "delay": 1.0})
    nest.Connect(parrot1, pre, conn_spec="one_to_one",
                 syn_spec={"receptor_type": 1, "weight": -1000. * np.exp(-0.1 / params["tau_AMPA"]), "delay": 1.1})

    nest.Connect(pre, post_app + post_exa, syn_spec={"receptor_type": 3, "weight": weight})
    nest.Connect(pre, rec_pre)
    nest.Connect(post_app, rec_post_app)
    nest.Connect(post_exa, rec_post_exa)
    nest.Connect(vm_app, post_app)
    nest.Connect(vm_exa, post_exa)
    nest.Connect(vm_in, pre, "one_to_one")
    nest.Simulate(t_sim)

    rate_in = rec_pre.n_events / (n_pre * t_sim) * 1000
    rate_post_app = rec_post_app.n_events / t_sim * 1000
    rate_post_exa = rec_post_exa.n_events / t_sim * 1000
    e_app = vm_app.events
    e_exa = vm_exa.events
#     e_in = vm_in.events
    rms_Vm = np.mean((e_app["V_m"] - e_exa["V_m"]) ** 2) ** 0.5
    mean_Vm_exa = np.mean(e_exa["V_m"])
    mean_Vm_app = np.mean(e_app["V_m"])


    return {
        "n_pre": n_pre,
        "w": weight,
        "r_pre": r_pre,
        "r_in": rate_in,
        "r_app": rate_post_app,
        "r_exa": rate_post_exa,
        "rms_Vm": rms_Vm,
        "mean_Vm_exa": mean_Vm_exa,
        "mean_Vm_app": mean_Vm_app,
        "Vm_exa": e_exa["V_m"],
        "Vm_app": e_app["V_m"],
#         "Vm_in": np.array([e["V_m"] for e in e_in]),
        "times": e_app["times"]
    }

weight = [0.1, 1, 10, 20, 50, 100]
r_pre = [1, 10, 50, 100]
n_pre = [1, 10, 100, 800, 1600, 3200]

res = []
for npr in n_pre:
    for w in weight:
        for rp in r_pre:
            print(".", end="", flush=True)
            if npr * rp * w > 1e4:
                continue  # skip unrealistically high input regimes

            seed = np.random.randint(low=1, high=2**32-1)
            res.append(do_sim(npr, rp, w, seed=seed))

        print(":", end="", flush=True)
    print("", flush=True)

d = pd.DataFrame.from_records(res)
d["Total input"] = d.n_pre * d.r_in * d.w

##

gs = GridSpec(10,20)
fig = plt.figure()
fig.set_size_inches([16 / 2.54, 5.5 / 2.54])
fig.subplots_adjust(left=0.065, right=1.06, bottom=0.165, top=0.97, wspace=0.1, hspace=0.4)

# Error plot
ax0 = fig.add_subplot(gs[:,:])
sc = ax0.scatter(d["Total input"], d["rms_Vm"], c=np.log10(d.w), s=d.n_pre**0.5, cmap="viridis")
ax0.set_xscale("log")
ax0.set_ylabel("RMS($V_\mathrm{m,exact} - V_\mathrm{m,approx}$)")
ax0.set_xlabel("Total input", labelpad=0)

cbar = plt.colorbar(sc, pad=0.01)
cbar.ax.get_yaxis().labelpad = 0
cbar.ax.set_ylabel("$\mathrm{log}_{10}(w)$", rotation=270, labelpad=3.)

# Example plots
closest_errs = [0.12, 3.2, 6.6]
inds = [np.abs(d["rms_Vm"].to_numpy() - err).argsort()[0] for err in closest_errs]
share = None

for i in range(3):
    j = inds[2-i] # data index
    ax = fig.add_subplot(gs[1+2*i:1+2*(i+1), 1:8], sharex=share)
    ax.plot(d["times"][inds[i]], d["Vm_app"][j], color=cbcolors[0])
    ax.plot(d["times"][inds[i]], d["Vm_exa"][j], color=cbcolors[1])

    xy = (d["Total input"][j], d["rms_Vm"][j])
    con = ConnectionPatch(xyA=xy, coordsA=ax0.transData,
                          xyB=(1., 0.5), coordsB=ax.transAxes, shrinkA=6, shrinkB=0)
    fig.add_artist(con)
    ax.set_xlim(0, 1000)
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig("figure2.pdf")
plt.show()

