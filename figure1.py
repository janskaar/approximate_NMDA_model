import nest
import matplotlib.pyplot as plt
import seaborn as sns
cbcolors = sns.color_palette("colorblind").as_hex()
import numpy as np

plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8

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


nest.ResetKernel()
nest.rng_seed = 12345

# pre-synaptic neuron, must be approximate model since the post-synaptic approximate model needs the offset
nrn_pre = nest.Create("iaf_bw_2001", params)
nrn_post_approx = nest.Create("iaf_bw_2001", params)
nrn_post_exact = nest.Create("iaf_bw_2001_exact", params)

pg = nest.Create("poisson_generator", {"rate": 50.0})

sr = nest.Create("spike_recorder", {"time_in_steps": True})
mm_pre = nest.Create("multimeter", {"interval": nest.resolution,
                                    "record_from": ["V_m", "I_NMDA"]})
mm_post_approx = nest.Create("multimeter", {"interval": nest.resolution,
                                            "record_from": ["V_m", "I_NMDA"]})
mm_post_exact = nest.Create("multimeter", {"interval": nest.resolution,
                                           "record_from": ["V_m", "I_NMDA"]})

receptors = nest.GetDefaults("iaf_bw_2001")["receptor_types"]
ampa_ext_syn_spec = {"synapse_model": "static_synapse", "weight": w_ext, "receptor_type": receptors["AMPA"]}

rec_syn_specs = nest.CollocatedSynapses(
    {"synapse_model": "static_synapse", "weight": w_ex, "receptor_type": receptors["AMPA"]},
    {"synapse_model": "static_synapse", "weight": w_ex, "receptor_type": receptors["NMDA"]},
    {"synapse_model": "static_synapse", "weight": w_in, "receptor_type": receptors["GABA"]},
)

nest.Connect(pg, nrn_pre, syn_spec=ampa_ext_syn_spec)
nest.Connect(nrn_pre, nrn_post_approx, syn_spec=rec_syn_specs)
nest.Connect(nrn_pre, nrn_post_exact, syn_spec=rec_syn_specs)

nest.Connect(nrn_pre, sr)
nest.Connect(mm_pre, nrn_pre)
nest.Connect(mm_post_approx, nrn_post_approx)
nest.Connect(mm_post_exact, nrn_post_exact)

nest.Simulate(1000.0)

## 

fig, ax = plt.subplots(5, 2, sharex="col", sharey="row", height_ratios=[1, 1, 0.05, 1, 1])
ax[2,0].axis("off")
ax[2,1].axis("off")
fig.set_size_inches([16 / 2.54, 9 / 2.54])
fig.subplots_adjust(left=0.12, right=0.98, top=0.99, bottom=0.10, hspace=0.3, wspace=0.13)

ax[0, 0].plot((ev := mm_post_exact.events)["times"], ev["I_NMDA"], "--", label="Exact model", color=cbcolors[1], zorder=1)
ax[0, 0].plot((ev := mm_post_approx.events)["times"], ev["I_NMDA"], label="Approximate model", color=cbcolors[0], zorder=0)
ax[0, 0].set_xlim(0, 1000)
ax[0, 0].set_ylabel("$I_\mathrm{NMDA}$ [pA]")
ax[0, 1].plot((ev := mm_post_approx.events)["times"], ev["I_NMDA"], label="Approximation", color=cbcolors[0])
ax[0, 1].plot((ev := mm_post_exact.events)["times"], ev["I_NMDA"], "--", label="Exact model", color=cbcolors[1])
ax[0, 1].set_xlim(300, 400)

ax[0, 0].scatter(sr.events["times"] * 0.1, np.ones_like(sr.events["times"]) * ev["I_NMDA"].max(), c="red", s=5., label="Incoming spikes")
ax[0, 1].scatter(sr.events["times"] * 0.1, np.ones_like(sr.events["times"]) * ev["I_NMDA"].max(), c="red", s=5., label="Incoming spikes")



ax[1, 0].plot((ev := mm_post_approx.events)["times"], ev["I_NMDA"] - mm_post_exact.events["I_NMDA"], label="Approximation", color="black")
ax[1, 0].set_ylabel("Error [pA]")
ax[1, 0].set_xlim(0, 1000)

ax[1, 1].plot((ev := mm_post_approx.events)["times"], ev["I_NMDA"] - mm_post_exact.events["I_NMDA"], label="Approximation", color="black")
ax[1, 1].set_xlim(300, 400);

ax[3, 0].plot((ev := mm_post_approx.events)["times"], ev["V_m"], label="Approximation", color=cbcolors[0])
ax[3, 0].plot((ev := mm_post_exact.events)["times"], ev["V_m"], "--", label="Exact model", color=cbcolors[1])
ax[3, 0].set_xlim(0, 1000)
ax[3, 0].set_ylabel("$V_\mathrm{m}$ [mV]")

ax[3, 1].plot((ev := mm_post_approx.events)["times"], ev["V_m"], label="Approximation", color=cbcolors[0])
ax[3, 1].plot((ev := mm_post_exact.events)["times"], ev["V_m"], "--", label="Exact model", color=cbcolors[1])
ax[3, 1].set_xlim(300, 400)
# ax[0,1].text(-0.45, 1.1, "Errors in membrane potential",
#      transform = ax[0,1].transAxes)

ax[4, 0].plot((ev := mm_post_approx.events)["times"], ev["V_m"] - mm_post_exact.events["V_m"], label="Approximation", color="black")
ax[4, 0].set_ylabel("Error [mV]")
ax[4, 0].set_xlim(0, 1000)

ax[4, 1].plot((ev := mm_post_approx.events)["times"], ev["V_m"] - mm_post_exact.events["V_m"], label="Approximation", color="black")
ax[4, 1].set_xlim(300, 400);
ax[4, 0].set_xlabel("Time [ms]", labelpad=2)
ax[4, 1].set_xlabel("Time [ms]", labelpad=2)

fig.savefig("figure1.pdf")
plt.show()

