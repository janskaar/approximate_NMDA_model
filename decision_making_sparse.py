import os, sys, h5py
import matplotlib.pyplot as plt
import nest
import numpy as np

runner_id = int(sys.argv[1])
n_threads = int(os.environ["SLURM_CPUS_PER_TASK"])

outfile = os.path.join("decision_making_sparse", f"runner_{runner_id}.h5")

def run_sim(s1, s2, seed=123):

    dt = 0.1

    NE = 1600
    NI = 400

    f = 0.15  # proportion of neurons receiving signal inputs

    nest.ResetKernel()
    nest.set(resolution=dt, print_time=False, rng_seed=seed, total_num_virtual_procs=n_threads)

    # conductances excitatory population
    # fmt: off
    g_AMPA_ex = 0.05 / epsilon       # recurrent AMPA conductance
    g_AMPA_ext_ex = 2.1              # external AMPA conductance
    g_NMDA_ex = 0.165 / epsilon      # recurrent GABA conductance
    g_GABA_ex = 1.3 / epsilon        # recurrent GABA conductance

    # conductances inhibitory population
    g_AMPA_in = 0.04 / epsilon       # recurrent AMPA conductance
    g_AMPA_ext_in = 1.62             # external AMPA conductance
    g_NMDA_in = 0.13 / epsilon       # recurrent GABA conductance
    g_GABA_in = 1.0 / epsilon        # recurrent GABA conductance

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
        "V_m" : -55.0                # initial membrane potential
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
        "V_m" : -55.0                # initial membrane potential
    }
    # fmt: on

    # synaptic weights
    w_plus = 1.7
    w_minus = 1 - f * (w_plus - 1) / (1 - f)

    delay = 0.5

    ##################################################
    # Create neurons and devices

    selective_pop1 = nest.Create("iaf_bw_2001", int(f * NE), params=epop_params)
    selective_pop2 = nest.Create("iaf_bw_2001", int(f * NE), params=epop_params)
    nonselective_pop = nest.Create("iaf_bw_2001", int((1 - 2 * f) * NE), params=epop_params)
    inhibitory_pop = nest.Create("iaf_bw_2001", NI, params=ipop_params)

    poisson_0 = nest.Create("poisson_generator", params={"rate": 2400.0})

    sr_nonselective = nest.Create("spike_recorder", {"time_in_steps": True})
    sr_selective1 = nest.Create("spike_recorder", {"time_in_steps": True})
    sr_selective2 = nest.Create("spike_recorder", {"time_in_steps": True})
    sr_inhibitory = nest.Create("spike_recorder", {"time_in_steps": True})

    mm_selective1 = nest.Create("multimeter", int(f * NE), params={"record_from": ["V_m", "s_AMPA", "I_AMPA", "s_NMDA", "I_NMDA", "s_GABA", "I_GABA", "s_NMDA_pre"]})
    mm_selective2 = nest.Create("multimeter", int(f * NE), params={"record_from": ["V_m", "s_AMPA", "I_AMPA", "s_NMDA", "I_NMDA", "s_GABA", "I_GABA", "s_NMDA_pre"]})
    mm_nonselective = nest.Create("multimeter", int((1 - 2 * f) * NE), params={"record_from": ["V_m", "s_AMPA", "I_AMPA", "s_NMDA", "I_NMDA", "s_GABA", "I_GABA", "s_NMDA_pre"]})
    mm_inhibitory = nest.Create("multimeter", NI, params={"record_from": ["V_m", "s_AMPA", "I_AMPA", "s_NMDA", "I_NMDA", "s_GABA", "I_GABA", "s_NMDA_pre"]})

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

    #
    # from external
    #

    nest.Connect(
        poisson_0, nonselective_pop + selective_pop1 + selective_pop2, conn_spec="all_to_all", syn_spec=exte_syn_spec
    )
    nest.Connect(poisson_0, inhibitory_pop, conn_spec="all_to_all", syn_spec=exti_syn_spec)


    #
    # from nonselective pop
    #

    conn_spec = {"rule": "fixed_indegree", "indegree": int(NE * (1 - 2 * f) * epsilon), "allow_multapses": False}
    syn_spec = nest.CollocatedSynapses(syn_spec_dep_AMPA,
                                       syn_spec_dep_NMDA)
    nest.Connect(nonselective_pop, selective_pop1 + selective_pop2, conn_spec=conn_spec, syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ee_syn_spec_AMPA,
                                       ee_syn_spec_NMDA)
    nest.Connect(nonselective_pop, nonselective_pop, conn_spec=conn_spec, syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ei_syn_spec_AMPA,
                                       ei_syn_spec_NMDA)
    nest.Connect(nonselective_pop, inhibitory_pop, conn_spec=conn_spec, syn_spec=syn_spec)

    nest.Connect(nonselective_pop, sr_nonselective)


    #
    # from selective pops
    #

    conn_spec = {"rule": "fixed_indegree", "indegree": int(NE * f * epsilon), "allow_multapses": False}
    syn_spec = nest.CollocatedSynapses(syn_spec_pot_AMPA,
                                       syn_spec_pot_NMDA)
    nest.Connect(selective_pop1, selective_pop1, conn_spec=conn_spec, syn_spec=syn_spec)
    nest.Connect(selective_pop2, selective_pop2, conn_spec=conn_spec, syn_spec=syn_spec)

    conn_spec = {"rule": "fixed_indegree", "indegree": int(NE * f * epsilon), "allow_multapses": False}
    syn_spec = nest.CollocatedSynapses(syn_spec_dep_AMPA,
                                       syn_spec_dep_NMDA)
    nest.Connect(selective_pop1, selective_pop2, conn_spec=conn_spec, syn_spec=syn_spec)
    nest.Connect(selective_pop2, selective_pop1, conn_spec=conn_spec, syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ee_syn_spec_AMPA,
                                       ee_syn_spec_NMDA)
    nest.Connect(selective_pop1, nonselective_pop, conn_spec=conn_spec, syn_spec=syn_spec)
    nest.Connect(selective_pop2, nonselective_pop, conn_spec=conn_spec, syn_spec=syn_spec)

    syn_spec = nest.CollocatedSynapses(ei_syn_spec_AMPA,
                                       ei_syn_spec_NMDA)
    nest.Connect(selective_pop1, inhibitory_pop, conn_spec=conn_spec, syn_spec=syn_spec)
    nest.Connect(selective_pop2, inhibitory_pop, conn_spec=conn_spec, syn_spec=syn_spec)

    nest.Connect(selective_pop1, sr_selective1)
    nest.Connect(selective_pop2, sr_selective2)



    #
    # from inhibitory pop
    #

    conn_spec = {"rule": "fixed_indegree", "indegree": int(NI * epsilon), "allow_multapses": False}
    nest.Connect(
        inhibitory_pop, selective_pop1 + selective_pop2, conn_spec=conn_spec, syn_spec=ie_syn_spec
    )

    conn_spec = {"rule": "fixed_indegree", "indegree": int(NI * epsilon), "allow_multapses": False}
    nest.Connect(
        inhibitory_pop, nonselective_pop, conn_spec=conn_spec, syn_spec=ie_syn_spec
    )

    nest.Connect(inhibitory_pop, inhibitory_pop, conn_spec=conn_spec, syn_spec=ii_syn_spec)

    nest.Connect(inhibitory_pop, sr_inhibitory)


    #
    # multimeters
    #

    nest.Connect(mm_selective1, selective_pop1, "one_to_one")
    nest.Connect(mm_selective2, selective_pop2, "one_to_one")
    nest.Connect(mm_nonselective, nonselective_pop, "one_to_one")
    nest.Connect(mm_inhibitory, inhibitory_pop, "one_to_one")


    #
    # Initialize conditions such that synapses from selective populations have fixed s_NMDA value
    #

    # set weight of NMDA connections to 0
    # get handle for all NMDA synapses from selective populations
    conns_s1s1 = nest.GetConnections(source=selective_pop1, target=selective_pop1)
    conns_s2s2 = nest.GetConnections(source=selective_pop2, target=selective_pop2)
    conns_s1s2 = nest.GetConnections(source=selective_pop1, target=selective_pop2)
    conns_s2s1 = nest.GetConnections(source=selective_pop2, target=selective_pop1)
    conns_si = nest.GetConnections(source=selective_pop1 + selective_pop2, target=inhibitory_pop)
    conns_sn = nest.GetConnections(source=selective_pop1 + selective_pop2, target=nonselective_pop)

    def set_NMDA_weights(conns):
        receptors = conns.get("receptor")
        orig_weights= conns.get("weight")
        new_weights= [w if receptors[i] == 1 else 0. for i, w in enumerate(orig_weights)]
        conns.set({"weight": new_weights})
        return orig_weights

    # keep old weights to reset them after reaching equilibrium
    orig_weights_s1s1 = set_NMDA_weights(conns_s1s1)
    orig_weights_s2s2 = set_NMDA_weights(conns_s2s2)
    orig_weights_s1s2 = set_NMDA_weights(conns_s1s2)
    orig_weights_s2s1 = set_NMDA_weights(conns_s2s1)
    orig_weights_si = set_NMDA_weights(conns_si)
    orig_weights_sn = set_NMDA_weights(conns_sn)

    # set constant s_NMDA current to all population assuming s1 and s2 have fixed values
    val = s1 * NE * f * w_plus * epsilon + s2 * NE * f * w_minus * epsilon
    selective_pop1.set({"s_NMDA_post_current": val * g_NMDA_ex / epop_params["tau_decay_NMDA"],
                        "s_NMDA_pre_clamp": True,
                        "s_NMDA_pre_value": s1})


    val = s1 * NE * f * w_minus * epsilon + s2 * NE * f * w_plus * epsilon
    selective_pop2.set({"s_NMDA_post_current": val * g_NMDA_ex / epop_params["tau_decay_NMDA"],
                        "s_NMDA_pre_clamp": True,
                        "s_NMDA_pre_value": s2})

    val = s1 * NE * f * epsilon * w_minus + s2 * NE * f * epsilon * w_plus
    nonselective_pop.set({"s_NMDA_post_current": val * g_NMDA_ex / epop_params["tau_decay_NMDA"]})

    val = s1 * NE * f * 1 * epsilon + s2 * NE * f * 1 * epsilon
    nonselective_pop.set({"s_NMDA_post_current": val * g_NMDA_ex / epop_params["tau_decay_NMDA"]})

    val = s1 * NE * f * 1 * epsilon + s2 * NE * f * 1 * epsilon
    inhibitory_pop.set({"s_NMDA_post_current": val * g_NMDA_in / epop_params["tau_decay_NMDA"]})

    ##################################################
    # Run simulation

    nest.Simulate(2000.0)

    # remove constant s_NMDA from selective pops
    selective_pop1.set({"s_NMDA_post_current": 0,
                        "s_NMDA_pre_clamp": False})

    selective_pop2.set({"s_NMDA_post_current": 0,
                        "s_NMDA_pre_clamp": False})

    nonselective_pop.set({"s_NMDA_post_current": 0})

    inhibitory_pop.set({"s_NMDA_post_current": 0})

    # reset weights
    conns_s1s1.set({"weight": orig_weights_s1s1})
    conns_s2s2.set({"weight": orig_weights_s2s2})
    conns_s1s2.set({"weight": orig_weights_s1s2})
    conns_s2s1.set({"weight": orig_weights_s2s1})
    conns_sn.set({"weight": orig_weights_sn})
    conns_si.set({"weight": orig_weights_si})

    nest.Simulate(4000.0)

    ##################################################
    # Collect data from simulation
    bins = np.arange(0, 60001, 10) - 0.01
    spikes_nonselective = sr_nonselective.get("events", "times")
    hist_nonselective, _ = np.histogram(spikes_nonselective, bins=bins)
    spikes_selective1 = sr_selective1.get("events", "times")
    hist_selective1, _ = np.histogram(spikes_selective1, bins=bins)
    spikes_selective2 = sr_selective2.get("events", "times")
    hist_selective2, _ = np.histogram(spikes_selective2, bins=bins)
    spikes_inhibitory = sr_inhibitory.get("events", "times")
    hist_inhibitory, _ = np.histogram(spikes_inhibitory, bins=bins)

    n = int(NE * (1 - 2 * f))
    vm_nonselective = np.array([mm_nonselective[i].get("events", "V_m") for i in range(n)])
    s_AMPA_nonselective = np.array([mm_nonselective[i].get("events", "s_AMPA") for i in range(n)])
    s_NMDA_nonselective = np.array([mm_nonselective[i].get("events", "s_NMDA") for i in range(n)])
    s_NMDA_pre_nonselective = np.array([mm_nonselective[i].get("events", "s_NMDA_pre") for i in range(n)])
    s_GABA_nonselective = np.array([mm_nonselective[i].get("events", "s_GABA") for i in range(n)])
    I_AMPA_nonselective = np.array([mm_nonselective[i].get("events", "I_AMPA") for i in range(n)])
    I_NMDA_nonselective = np.array([mm_nonselective[i].get("events", "I_NMDA") for i in range(n)])
    I_GABA_nonselective = np.array([mm_nonselective[i].get("events", "I_GABA") for i in range(n)])

    n = int(NE * f)
    vm_selective1 = np.array([mm_selective1[i].get("events", "V_m") for i in range(n)])
    s_AMPA_selective1 = np.array([mm_selective1[i].get("events", "s_AMPA") for i in range(n)])
    s_NMDA_selective1 = np.array([mm_selective1[i].get("events", "s_NMDA") for i in range(n)])
    s_NMDA_pre_selective1 = np.array([mm_selective1[i].get("events", "s_NMDA_pre") for i in range(n)])
    s_GABA_selective1 = np.array([mm_selective1[i].get("events", "s_GABA") for i in range(n)])
    I_AMPA_selective1 = np.array([mm_selective1[i].get("events", "I_AMPA") for i in range(n)])
    I_NMDA_selective1 = np.array([mm_selective1[i].get("events", "I_NMDA") for i in range(n)])
    I_GABA_selective1 = np.array([mm_selective1[i].get("events", "I_GABA") for i in range(n)])

    vm_selective2 = np.array([mm_selective2[i].get("events", "V_m") for i in range(n)])
    s_AMPA_selective2 = np.array([mm_selective2[i].get("events", "s_AMPA") for i in range(n)])
    s_NMDA_selective2 = np.array([mm_selective2[i].get("events", "s_NMDA") for i in range(n)])
    s_NMDA_pre_selective2 = np.array([mm_selective2[i].get("events", "s_NMDA_pre") for i in range(n)])
    s_GABA_selective2 = np.array([mm_selective2[i].get("events", "s_GABA") for i in range(n)])
    I_AMPA_selective2 = np.array([mm_selective2[i].get("events", "I_AMPA") for i in range(n)])
    I_NMDA_selective2 = np.array([mm_selective2[i].get("events", "I_NMDA") for i in range(n)])
    I_GABA_selective2 = np.array([mm_selective2[i].get("events", "I_GABA") for i in range(n)])

    n = NI
    vm_inhibitory = np.array([mm_inhibitory[i].get("events", "V_m") for i in range(n)])
    s_AMPA_inhibitory = np.array([mm_inhibitory[i].get("events", "s_AMPA") for i in range(n)])
    s_NMDA_inhibitory = np.array([mm_inhibitory[i].get("events", "s_NMDA") for i in range(n)])
    s_NMDA_pre_inhibitory = np.array([mm_inhibitory[i].get("events", "s_NMDA_pre") for i in range(n)])
    s_GABA_inhibitory = np.array([mm_inhibitory[i].get("events", "s_GABA") for i in range(n)])
    I_AMPA_inhibitory = np.array([mm_inhibitory[i].get("events", "I_AMPA") for i in range(n)])
    I_NMDA_inhibitory = np.array([mm_inhibitory[i].get("events", "I_NMDA") for i in range(n)])
    I_GABA_inhibitory = np.array([mm_inhibitory[i].get("events", "I_GABA") for i in range(n)])

    return {
        "hist_selective1": hist_selective1,
        "hist_selective2": hist_selective2,
        "hist_nonselective": hist_nonselective,
        "hist_inhibitory": hist_inhibitory,

        "s_AMPA_selective1": s_AMPA_selective1,
        "s_AMPA_selective2": s_AMPA_selective2,
        "s_AMPA_nonselective": s_AMPA_nonselective,
        "s_AMPA_inhibitory": s_AMPA_inhibitory,

        "I_AMPA_selective1": I_AMPA_selective1,
        "I_AMPA_selective2": I_AMPA_selective2,
        "I_AMPA_nonselective": I_AMPA_nonselective,
        "I_AMPA_inhibitory": I_AMPA_inhibitory,

        "s_NMDA_selective1": s_NMDA_selective1,
        "s_NMDA_selective2": s_NMDA_selective2,
        "s_NMDA_nonselective": s_NMDA_nonselective,
        "s_NMDA_inhibitory": s_NMDA_inhibitory,

        "I_NMDA_selective1": I_NMDA_selective1,
        "I_NMDA_selective2": I_NMDA_selective2,
        "I_NMDA_nonselective": I_NMDA_nonselective,
        "I_NMDA_inhibitory": I_NMDA_inhibitory,

        "s_GABA_selective1": s_GABA_selective1,
        "s_GABA_selective2": s_GABA_selective2,
        "s_GABA_nonselective": s_GABA_nonselective,
        "s_GABA_inhibitory": s_GABA_inhibitory,

        "I_GABA_selective1": I_GABA_selective1,
        "I_GABA_selective2": I_GABA_selective2,
        "I_GABA_nonselective": I_GABA_nonselective,
        "I_GABA_inhibitory": I_GABA_inhibitory,

        "vm_selective1": vm_selective1,
        "vm_selective2": vm_selective2,
        "vm_nonselective": vm_nonselective,
        "vm_inhibitory": vm_inhibitory,

        "s_NMDA_pre_selective1": s_NMDA_pre_selective1,
        "s_NMDA_pre_selective2": s_NMDA_pre_selective2,
        "s_NMDA_pre_nonselective": s_NMDA_pre_nonselective,
        "s_NMDA_pre_inhibitory": s_NMDA_pre_inhibitory,
   }

##

base_seed = int(os.environ["SLURM_JOB_ID"]) * 1000 + int(runner_id)
np.random.seed(base_seed)
rng = np.random.default_rng()

epsilons = np.linspace(0.9, 1.0, 11)
for i, epsilon in enumerate(epsilons):
    nest_seed = np.random.randint(low=1, high=2**32-1)
    s1 = np.random.rand()
    s2 = np.random.rand()
    s1 = 1.0
    s2 = 0.0

    res = run_sim(s1, s2, seed=nest_seed)

    with h5py.File(outfile, "a") as f:
        inner_group = f.create_group(str(i))
        inner_group.attrs["s1_init"] = s1
        inner_group.attrs["s2_init"] = s2
        inner_group.attrs["epsilon"] = epsilon
        inner_group.attrs["seed"] = nest_seed
        for key, val in res.items():
            if "hist" in key: 
                inner_group.create_dataset(key, data=val)
            else:
                inner_group.create_dataset(key + "_mean", data=val.mean(0))
#                 inner_group.create_dataset(key + "_std", data=val.std(0))

