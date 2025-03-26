import matplotlib.pyplot as plt
import nest
import os, pickle
import numpy as np
from matplotlib.gridspec import GridSpec

np.random.seed(1234)
rng = np.random.default_rng()

plt.rcParams["font.size"] = 8

# Use approximate model, can be replaced by "iaf_wang_2002_exact"
model = "iaf_bw_2001"

dt = 0.1

# number of neurons in each population
NE = 1600
NI = 400


def run_sim(coherence, seed=123):
    nest.ResetKernel()
    nest.set(resolution=dt, print_time=True, rng_seed=seed, total_num_virtual_procs=4)
    ##################################################
    # Set parameter values, taken from [1]_.

    # conductances excitatory population
    # fmt: off
    g_AMPA_ex = 0.05                 # recurrent AMPA conductance
    g_AMPA_ext_ex = 2.1              # external AMPA conductance
    g_NMDA_ex = 0.165                # recurrent NMDA conductance
    g_GABA_ex = 1.3                  # recurrent GABA conductance

    # conductances inhibitory population
    g_AMPA_in = 0.04                 # recurrent AMPA conductance
    g_AMPA_ext_in = 1.62             # external AMPA conductance
    g_NMDA_in = 0.13                 # recurrent NMDA conductance
    g_GABA_in = 1.0                  # recurrent GABA conductance

    # neuron parameters
    epop_params = {
        "tau_GABA": 5.0,             # GABA decay time constant
        "tau_AMPA": 2.0,             # AMPA decay time constant
        "tau_decay_NMDA": 100.0,     # NMDA decay time constant
        "tau_rise_NMDA": 2.0,        # NMDA rise time constant
        "alpha": 0.5,                # NMDA parameter
        "conc_Mg2": 1.0,             # Magnesium concentration
        "g_L": 25.0,                 # leak conductance
        "E_L": -70.0,                # leak reversal potential
        "E_ex": 0.0,                 # excitatory reversal potential
        "E_in": -70.0,               # inhibitory reversal potential
        "V_reset": -55.0,            # reset potential
        "V_th": -50.0,               # threshold
        "C_m": 500.0,                # membrane capacitance
        "t_ref": 2.0,                # refreactory period
    }

    ipop_params = {
        "tau_GABA": 5.0,             # GABA decay time constant
        "tau_AMPA": 2.0,             # AMPA decay time constant
        "tau_decay_NMDA": 100.0,     # NMDA decay time constant
        "tau_rise_NMDA": 2.0,        # NMDA rise time constant
        "alpha": 0.5,                # NMDA parameter
        "conc_Mg2": 1.0,             # Magnesium concentration
        "g_L": 20.0,                 # leak conductance
        "E_L": -70.0,                # leak reversal potential
        "E_ex": 0.0,                 # excitatory reversal potential
        "E_in": -70.0,               # inhibitory reversal potential
        "V_reset": -55.0,            # reset potential
        "V_th": -50.0,               # threshold
        "C_m": 200.0,                # membrane capacitance
        "t_ref": 1.0,                # refreactory period
    }
    # fmt: on

    # signals to the two different excitatory sub-populations
    # the signal is given by a time-inhomogeneous Poisson process,
    # where the expectations are constant over intervals of 50ms,
    # and then change. The values for each interval are normally
    # distributed, with means mu_a and mu_b, and standard deviation
    # sigma.
    signal_start = 1000.0
    signal_duration = 1000.0
    signal_update_interval = 50.0
    f = 0.15  # proportion of neurons receiving signal inputs
    # compute expectations of the time-inhomogeneous Poisson processes
    mu_0 = 40.0  # base rate
    rho_a = mu_0 / 100  # scaling factors coherence
    rho_b = rho_a
    sigma = 4.0  # standard deviation
    mu_a = mu_0 + rho_a * coherence  # expectation for pop A
    mu_b = mu_0 - rho_b * coherence  # expectation for pop B

    # sample values for the Poisson process
    num_updates = int(signal_duration / signal_update_interval)
    update_times = np.arange(0, signal_duration, signal_update_interval)
    update_times[0] = 0.1
    rates_a = np.random.normal(mu_a, sigma, size=num_updates)
    rates_b = np.random.normal(mu_b, sigma, size=num_updates)

    # synaptic weights
    w_plus = 1.7
    w_minus = 1 - f * (w_plus - 1) / (1 - f)

    delay = 0.5

    ##################################################
    # Create neurons and devices

    selective_pop1 = nest.Create(model, int(f * NE), params=epop_params)
    selective_pop2 = nest.Create(model, int(f * NE), params=epop_params)
    nonselective_pop = nest.Create(model, int((1 - 2 * f) * NE), params=epop_params)
    inhibitory_pop = nest.Create(model, NI, params=ipop_params)

    poisson_a = nest.Create(
        "inhomogeneous_poisson_generator",
        params={
            "origin": signal_start - 0.1,
            "start": 0.0,
            "stop": signal_duration,
            "rate_times": update_times,
            "rate_values": rates_a,
        },
    )

    poisson_b = nest.Create(
        "inhomogeneous_poisson_generator",
        params={
            "origin": signal_start - 0.1,
            "start": 0.0,
            "stop": signal_duration,
            "rate_times": update_times,
            "rate_values": rates_b,
        },
    )

    poisson_0 = nest.Create("poisson_generator", params={"rate": 2400.0})

    sr_nonselective = nest.Create("spike_recorder")
    sr_selective1 = nest.Create("spike_recorder")
    sr_selective2 = nest.Create("spike_recorder")
    sr_inhibitory = nest.Create("spike_recorder")

    sr_selective1_raster = nest.Create("spike_recorder", 100)
    sr_selective2_raster = nest.Create("spike_recorder", 100)

    mm_selective1 = nest.Create("multimeter", {"record_from": ["V_m", "s_AMPA", "s_GABA"]})
    mm_selective2 = nest.Create("multimeter", {"record_from": ["V_m", "s_AMPA", "s_GABA"]})
    mm_nonselective = nest.Create("multimeter", {"record_from": ["V_m", "s_AMPA", "s_GABA"]})
    mm_inhibitory = nest.Create("multimeter", {"record_from": ["V_m", "s_AMPA", "s_GABA"]})

    ##################################################
    # Define synapse specifications

    receptor_types = selective_pop1[0].get("receptor_types")

    syn_spec_pot_AMPA = {
        "synapse_model": "static_synapse",
        "weight": w_plus * g_AMPA_ex,
        "delay": delay,
        "receptor_type": receptor_types["AMPA"],
    }
    syn_spec_pot_NMDA = {
        "synapse_model": "static_synapse",
        "weight": w_plus * g_NMDA_ex,
        "delay": delay,
        "receptor_type": receptor_types["NMDA"],
    }

    syn_spec_dep_AMPA = {
        "synapse_model": "static_synapse",
        "weight": w_minus * g_AMPA_ex,
        "delay": delay,
        "receptor_type": receptor_types["AMPA"],
    }

    syn_spec_dep_NMDA = {
        "synapse_model": "static_synapse",
        "weight": w_minus * g_NMDA_ex,
        "delay": delay,
        "receptor_type": receptor_types["NMDA"],
    }

    ie_syn_spec = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_GABA_ex,
        "delay": delay,
        "receptor_type": receptor_types["GABA"],
    }

    ii_syn_spec = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_GABA_in,
        "delay": delay,
        "receptor_type": receptor_types["GABA"],
    }

    ei_syn_spec_AMPA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_AMPA_in,
        "delay": delay,
        "receptor_type": receptor_types["AMPA"],
    }

    ei_syn_spec_NMDA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_NMDA_in,
        "delay": delay,
        "receptor_type": receptor_types["NMDA"],
    }

    ee_syn_spec_AMPA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_AMPA_ex,
        "delay": delay,
        "receptor_type": receptor_types["AMPA"],
    }

    ee_syn_spec_NMDA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_NMDA_ex,
        "delay": delay,
        "receptor_type": receptor_types["NMDA"],
    }

    exte_syn_spec = {
        "synapse_model": "static_synapse",
        "weight": g_AMPA_ext_ex,
        "delay": 0.1,
        "receptor_type": receptor_types["AMPA"],
    }

    exti_syn_spec = {
        "synapse_model": "static_synapse",
        "weight": g_AMPA_ext_in,
        "delay": 0.1,
        "receptor_type": receptor_types["AMPA"],
    }


    ##################################################
    # Create connections

    # from external
    nest.Connect(
        poisson_0, nonselective_pop + selective_pop1 + selective_pop2, conn_spec="all_to_all", syn_spec=exte_syn_spec
    )
    nest.Connect(poisson_0, inhibitory_pop, conn_spec="all_to_all", syn_spec=exti_syn_spec)

    nest.Connect(poisson_a, selective_pop1, conn_spec="all_to_all", syn_spec=exte_syn_spec)
    nest.Connect(poisson_b, selective_pop2, conn_spec="all_to_all", syn_spec=exte_syn_spec)

    # from nonselective pop
    syn_spec = nest.CollocatedSynapses(syn_spec_dep_AMPA,
                                       syn_spec_dep_NMDA)
    nest.Connect(nonselective_pop, selective_pop1 + selective_pop2, conn_spec="all_to_all", syn_spec=syn_spec)


    syn_spec = nest.CollocatedSynapses(ee_syn_spec_AMPA,
                                       ee_syn_spec_NMDA)
    nest.Connect(nonselective_pop, nonselective_pop, conn_spec="all_to_all", syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ei_syn_spec_AMPA,
                                       ei_syn_spec_NMDA)
    nest.Connect(nonselective_pop, inhibitory_pop, conn_spec="all_to_all", syn_spec=syn_spec)

    nest.Connect(nonselective_pop, sr_nonselective)

    # from selective pops
    syn_spec = nest.CollocatedSynapses(syn_spec_pot_AMPA,
                                       syn_spec_pot_NMDA)
    nest.Connect(selective_pop1, selective_pop1, conn_spec="all_to_all", syn_spec=syn_spec)
    nest.Connect(selective_pop2, selective_pop2, conn_spec="all_to_all", syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(syn_spec_dep_AMPA,
                                       syn_spec_dep_NMDA)
    nest.Connect(selective_pop1, selective_pop2, conn_spec="all_to_all", syn_spec=syn_spec)
    nest.Connect(selective_pop2, selective_pop1, conn_spec="all_to_all", syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ee_syn_spec_AMPA,
                                       ee_syn_spec_NMDA)
    nest.Connect(selective_pop1 + selective_pop2, nonselective_pop, conn_spec="all_to_all", syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ei_syn_spec_AMPA,
                                       ei_syn_spec_NMDA)
    nest.Connect(selective_pop1 + selective_pop2, inhibitory_pop, conn_spec="all_to_all", syn_spec=syn_spec)

    nest.Connect(selective_pop1, sr_selective1)
    nest.Connect(selective_pop2, sr_selective2)

    nest.Connect(selective_pop1[:100], sr_selective1_raster, "one_to_one")
    nest.Connect(selective_pop2[:100], sr_selective2_raster, "one_to_one")

    # from inhibitory pop
    nest.Connect(
        inhibitory_pop, selective_pop1 + selective_pop2 + nonselective_pop, conn_spec="all_to_all", syn_spec=ie_syn_spec
    )
    nest.Connect(inhibitory_pop, inhibitory_pop, conn_spec="all_to_all", syn_spec=ii_syn_spec)

    nest.Connect(inhibitory_pop, sr_inhibitory)


    ##################################################
    # Run simulation
    nest.Simulate(4000.0)

    ##################################################
    # Collect data from simulation
    spikes_nonselective = sr_nonselective.get("events", "times")
    spikes_selective1 = sr_selective1.get("events", "times")
    spikes_selective2 = sr_selective2.get("events", "times")
    spikes_inhibitory = sr_inhibitory.get("events", "times")

    spikes_selective1_raster = sr_selective1_raster.get("events", "times")
    spikes_selective2_raster = sr_selective2_raster.get("events", "times")

    return {
        "nonselective": spikes_nonselective,
        "selective1": spikes_selective1,
        "selective2": spikes_selective2,
        "inhibitory": spikes_inhibitory,
        "selective1_raster": spikes_selective1_raster,
        "selective2_raster": spikes_selective2_raster,
    }

seed = 1234
fname = "decision_making_exact_spikes.pkl" if "exact" in model else "decision_making_approx_spikes.pkl"
if not os.path.isfile(fname):
    spikes = []
    for c in [51.2, 12.8, 0.0]:
        spikes.append(run_sim(c, seed=seed))
    with open(fname, "wb") as f:
        pickle.dump(spikes, f)
else:
     with open(fname, "rb") as f:
        spikes = pickle.load(f)


##################################################
# Plots

# bins for histograms
res = 1.0
bins = np.arange(0, 4001, res) - 0.001

fig, ax = plt.subplots(
    ncols=2, nrows=8, sharex=True, sharey=False, height_ratios=[1, 0.7, 0.5, 1, 0.7, 0.5, 1, 0.7]
)
fig.subplots_adjust(left=0.11, right=0.96, bottom=0.08, top=0.85, hspace=0.0)
ax[0,0].set_xlim(0, 800)
#ax[0, 0].set_xticks([])
fig.set_size_inches(3.5, 4.5)

titles = ["c = 51.2", "c = 12.8", "c = 0.0"]
# selective populations
num = NE * 0.15

for j in range(3):
    # compute firing rates as moving averages over 50 ms windows with 5 ms strides
    hist1, _ = np.histogram(spikes[j]["selective1"], bins=bins)
    hist1 = hist1.reshape((-1, 5)).sum(-1)
    hist2, _ = np.histogram(spikes[j]["selective2"], bins=bins)
    hist2 = hist2.reshape((-1, 5)).sum(-1)

    pop1_rate = np.convolve(hist1, np.ones(10) * 0.1, mode="same") / num / 5 * 1000
    pop2_rate = np.convolve(hist2, np.ones(10) * 0.1, mode="same") / num / 5 * 1000

    ax[j*3+1,0].bar(np.arange(len(pop1_rate)), pop1_rate, width=1.0, color="black")
    ax[j*3+1,1].bar(np.arange(len(pop2_rate)), pop2_rate, width=1.0, color="black")
    ax[j*3+1,0].vlines([200, 400], 0, 40, colors="black", linewidths=1.0)
    ax[j*3+1,1].vlines([200, 400], 0, 40, colors="black", linewidths=1.0)
    ax[j*3+1,0].set_ylim(0, 40)
    ax[j*3+1,1].set_ylim(0, 40)
    ax[j*3,0].vlines([200, 400], 0, 100, colors="black", linewidths=1.0)
    ax[j*3,1].vlines([200, 400], 0, 100, colors="black", linewidths=1.0)
    for k in range(100):
        sp = spikes[j]["selective1_raster"][k] / 5.0
        ax[j*3,0].scatter(sp, np.ones_like(sp) * k, s=1., marker="|", c="black", linewidths=0.5)
        ax[j*3,0].set_yticks([])
        ax[j*3,0].set_ylim(0, 99)

        sp = spikes[j]["selective2_raster"][k] / 5.0
        ax[j*3,1].scatter(sp, np.ones_like(sp) * k, s=1., marker="|", c="black", linewidths=0.5)
        ax[j*3,1].set_yticks([])
        ax[j*3,1].set_ylim(0, 99)
    ax[j*3,0].set_title(titles[j])
    ax[j*3,1].set_title(titles[j])

# ax[0,0].set_title("Selective pop. A")
# ax[0,1].set_title("Selective pop. B")
ax[1,0].set_ylabel("Firing rate [sp/s]", labelpad=0.)
ax[4,0].set_ylabel("Firing rate [sp/s]", labelpad=0.)
ax[7,0].set_ylabel("Firing rate [sp/s]", labelpad=0.)

ax[-1,0].set_xlabel("Time [ms]")
ax[-1,1].set_xlabel("Time [ms]")
xticklabels = [str(int(l.get_text()) * 5) for l in ax[-1,0].get_xticklabels()]
ax[-1,1].set_xticklabels(xticklabels)
ax[-1,0].set_xticklabels(xticklabels)
ax[2,0].axis("off")
ax[2,1].axis("off")
ax[5,0].axis("off")
ax[5,1].axis("off")
ax[0,0].text(0.2, 1.5, "Selective pop. A", transform=ax[0,0].transAxes)
ax[0,1].text(0.2, 1.5, "Selective pop. B", transform=ax[0,1].transAxes)

title = "Exact model" if "exact" in model else "Approximate model"
fig.suptitle(title)
outfile = f"figure3_exact.pdf" if "exact" in model else f"figure3_approx.pdf"
fig.savefig(outfile)
plt.show()

