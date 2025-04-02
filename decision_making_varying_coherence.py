"""
This script generates the data required for figure 4. 

The decision making network is run for different coherence values, for both
the exact and the approximate model. The resulting histograms for each 
selective population are recorded to an hdf5-file. 16 simulations are run
for each coherence level, so the script must be run 25 times to get the 400
simulations used in the figure.
"""


import nest
import os, h5py, time, sys
import numpy as np


runner_id = int(sys.argv[1])
n_threads = int(os.environ["SLURM_CPUS_PER_TASK"])
print("Running with n_threads = ", n_threads, flush=True)
job_id = int(os.environ["SLURM_JOB_ID"])
print("Job ID: ", job_id, flush=True)

dt = 0.1

# number of neurons in each population
NE = 1600
NI = 400

outfile = os.path.join("decision_making_results", f"wang_decision_making_results_{job_id}_{runner_id}.h5")

def run_sim(coherence, model, seed=123):
    nest.ResetKernel()
    nest.set(resolution=dt, print_time=False, rng_seed=seed, total_num_virtual_procs=n_threads)
    ##################################################
    # Set parameter values, taken from [1]_.

    # conductances excitatory population
    # fmt: off
    g_AMPA_ex = 0.05                 # recurrent AMPA conductance
    g_AMPA_ext_ex = 2.1              # external AMPA conductance
    g_NMDA_ex = 0.165                # recurrent GABA conductance
    g_GABA_ex = 1.3                  # recurrent GABA conductance

    # conductances inhibitory population
    g_AMPA_in = 0.04                 # recurrent AMPA conductance
    g_AMPA_ext_in = 1.62             # external AMPA conductance
    g_NMDA_in = 0.13                 # recurrent GABA conductance
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
    signal_duration = 2000.0
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

    sr_selective1 = nest.Create("spike_recorder", params={"time_in_steps": True})
    sr_selective2 = nest.Create("spike_recorder", params={"time_in_steps": True})


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

    # from inhibitory pop
    nest.Connect(
        inhibitory_pop, selective_pop1 + selective_pop2 + nonselective_pop, conn_spec="all_to_all", syn_spec=ie_syn_spec
    )
    nest.Connect(inhibitory_pop, inhibitory_pop, conn_spec="all_to_all", syn_spec=ii_syn_spec)


    ##################################################
    # Run simulation
    nest.Simulate(4000.0)

    ##################################################
    # Collect data from simulation
    spikes_selective1 = sr_selective1.get("events", "times")
    spikes_selective2 = sr_selective2.get("events", "times")

    bins = np.arange(0, 4001, 1) - 0.01
    hist_1, _ = np.histogram(spikes_selective1 * 0.1, bins=bins)
    hist_2, _ = np.histogram(spikes_selective2 * 0.1, bins=bins)

    return {
        "selective_1": hist_1,
        "selective_2": hist_2,
    }


base_seed = int(time.time()) - runner_id * 1000000 # space them out in case the processes start at slightly different times
                                                   # and to avoid duplicate seeds for multiple jobs started in succession
print("base_seed = ", base_seed, flush=True)
np.random.seed(base_seed)

for i in range(16):
    tic_outer = time.time()
    for c in [1, 5, 10, 20, 40]:
        seed = np.random.randint(low=1, high=2**32-1)
        tic = time.time()
        res_approx = run_sim(c, model="iaf_bw_2001", seed=seed)
        toc = time.time()
        print(f"Approx time: {toc - tic}", flush=True)
        with h5py.File(outfile, "a") as f:
            f.create_dataset(f"approx_seed_{seed}_coherence_{c}_selective_1", data=res_approx["selective_1"])
            f.create_dataset(f"approx_seed_{seed}_coherence_{c}_selective_2", data=res_approx["selective_2"])

        tic = time.time()
        res_exact = run_sim(c, model="iaf_bw_2001_exact", seed=seed)
        toc = time.time()
        print(f"Exact time: {toc - tic}", flush=True)
        with h5py.File(outfile, "a") as f:
            f.create_dataset(f"exact_seed_{seed}_coherence_{c}_selective_1", data=res_exact["selective_1"])
            f.create_dataset(f"exact_seed_{seed}_coherence_{c}_selective_2", data=res_exact["selective_2"])

    toc_outer = time.time()
    print(f"batch {i} completed in {toc_outer - tic_outer:.1f} seconds", flush=True)

