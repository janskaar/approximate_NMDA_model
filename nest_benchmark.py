import nest
import numpy as np
import sys, os, time

runner_id = int(sys.argv[1])
scale = float(sys.argv[2])
n_threads = int(os.environ["SLURM_CPUS_PER_TASK"])

outfile = os.path.join(f"benchmarking_data_{n_threads}_threads",  f"wang_benchmark_nest_{runner_id}.csv")

dt = 0.1

##################################################
# conductances excitatory population
g_AMPA_ext_ex = 2.1              # external AMPA conductance     x
g_AMPA_ex = 0.05                 # recurrent AMPA conductance    x
g_NMDA_ex = 0.165                # recurrent GABA conductance    x
g_GABA_ex = 1.3                  # recurrent GABA conductance    x

# conductances inhibitory population
g_AMPA_ext_in = 1.62             # external AMPA conductance     x
g_AMPA_in = 0.04                 # recurrent AMPA conductance    x
g_NMDA_in = 0.13                 # recurrent GABA conductance    x
g_GABA_in = 1.0                  # recurrent GABA conductance    x 

# neuron parameters
epop_params = {
    "tau_GABA": 5.0,            # GABA decay time constant      x
    "tau_AMPA": 2.0,             # AMPA decay time consta
    "tau_decay_NMDA": 100.0,     # NMDA decay time constant      x
    "tau_rise_NMDA": 2.0,        # NMDA rise time constant       x
    "alpha": 0.5,                # NMDA parameter                x
    "conc_Mg2": 1.0,             # Magnesium concentration       x
    "g_L": 25.0,                 # leak conductance              x
    "E_L": -70.0,                # leak reversal potential       x
    "E_ex": 0.0,                 # excitatory reversal potential x
    "E_in": -70.0,               # inhibitory reversal potential x
    "V_reset": -55.0,            # reset potential               x
    "V_th": -50.0,               # threshold                     x
    "C_m": 500.0,                # membrane capacitance          x
    "t_ref": 2.0,                # refreactory period            x
    "V_m": -70.                  # initial membrane potential
}

ipop_params = {
    "tau_GABA": 5.0,            # GABA decay time constant     x
    "tau_AMPA": 2.0,             # AMPA decay time constant      x
    "tau_decay_NMDA": 100.0,     # NMDA decay time constant      x
    "tau_rise_NMDA": 2.0,        # NMDA rise time constant       x
    "alpha": 0.5,                # NMDA parameter                x
    "conc_Mg2": 1.0,             # Magnesium concentration       x
    "g_L": 20.0,                 # leak conductance              x
    "E_L": -70.0,                # leak reversal potential       x
    "E_ex": 0.0,                 # excitatory reversal potential x
    "E_in": -70.0,               # inhibitory reversal potential x
    "V_reset": -55.0,            # reset potential               x
    "V_th": -50.0,               # threshold                     x
    "C_m": 200.0,                # membrane capacitance          x
    "t_ref": 1.0,                # refreactory period            x
    "V_m": -70.                  # initial membrane potential
}


def run_sim(scale, model, seed=123):

    N = 2560
    # number of neurons in each population
    NE = int(N * 0.8 * scale)
    NI = int(N * 0.2 * scale)

    eparams = epop_params.copy()
    iparams = ipop_params.copy()


    nest.ResetKernel()
    nest.set(resolution=dt,
             print_time=False,
             rng_seed=seed,
             local_num_threads=n_threads)
    print(f"local_num_threads={nest.local_num_threads}")
    print(f"total_num_virtual_procs={nest.total_num_virtual_procs}")
    print(f"num_processes={nest.num_processes}")
    delay = 0.5

    ##################################################
    # Create neurons and devices

    excitatory_pop = nest.Create(model, NE, params=eparams)
    inhibitory_pop = nest.Create(model, NI, params=iparams)

    poisson_0 = nest.Create("poisson_generator", params={"rate": 2400.0})

    sr_excitatory = nest.Create("spike_recorder", params={"time_in_steps": True})
    sr_inhibitory = nest.Create("spike_recorder", params={"time_in_steps": True})


    ##################################################
    # Define synapse specifications

    receptor_types = nest.GetDefaults(model)["receptor_types"]

    ie_syn_spec = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_GABA_ex * 400 / NI,
        "delay": delay,
        "receptor_type": receptor_types["GABA"],
    }

    ii_syn_spec = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_GABA_in * 400 / NI,
        "delay": delay,
        "receptor_type": receptor_types["GABA"],
    }

    ei_syn_spec_AMPA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_AMPA_in * 1600 / NE,
        "delay": delay,
        "receptor_type": receptor_types["AMPA"],
    }

    ei_syn_spec_NMDA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_NMDA_in * 1600 / NE,
        "delay": delay,
        "receptor_type": receptor_types["NMDA"],
    }

    ee_syn_spec_AMPA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_AMPA_ex * 1600 / NE,
        "delay": delay,
        "receptor_type": receptor_types["AMPA"],
    }

    ee_syn_spec_NMDA = {
        "synapse_model": "static_synapse",
        "weight": 1.0 * g_NMDA_ex * 1600 / NE,
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
        poisson_0, excitatory_pop, conn_spec="all_to_all", syn_spec=exte_syn_spec
    )
    nest.Connect(
        poisson_0, inhibitory_pop, conn_spec="all_to_all", syn_spec=exti_syn_spec
    )


    # from excitatory pop
    syn_spec = nest.CollocatedSynapses(ee_syn_spec_AMPA, ee_syn_spec_NMDA)
    nest.Connect(
        excitatory_pop, excitatory_pop, conn_spec="all_to_all", syn_spec=syn_spec
    )

    syn_spec = nest.CollocatedSynapses(ei_syn_spec_AMPA, ei_syn_spec_NMDA)
    nest.Connect(
        excitatory_pop, inhibitory_pop, conn_spec="all_to_all", syn_spec=syn_spec
    )

    nest.Connect(excitatory_pop, sr_excitatory)


    # from inhibitory pop
    nest.Connect(
        inhibitory_pop, excitatory_pop, conn_spec="all_to_all", syn_spec=ie_syn_spec
    )
    nest.Connect(
        inhibitory_pop, inhibitory_pop, conn_spec="all_to_all", syn_spec=ii_syn_spec
    )

    nest.Connect(inhibitory_pop, sr_inhibitory)

    ##################################################
    tic = time.time()
    nest.Simulate(1000.0)

    ##################################################
    # Collect data from simulation
    num_ex = sr_excitatory.n_events
    num_in = sr_inhibitory.n_events

    return {"rate_ex": num_ex / NE,
            "rate_in": num_in / NI,
            "time": time.time() - tic}

if not os.path.isfile(outfile):
    with open(outfile, "w") as f:
        f.write("time_approx,time_exact,rate_ex_approx,rate_in_approx,rate_ex_exact,rate_in_exact,scale\n")

res_app = run_sim(scale, model="iaf_bw_2001", seed=runner_id+1)
res_exa = run_sim(scale, model="iaf_bw_2001_exact", seed=runner_id+1)
print(f"Execution time: {res_exa['time']} s")

with open(outfile, "a") as f:
    f.writelines(f"{res_app['time']},{res_exa['time']},{res_app['rate_ex']},{res_app['rate_in']},{res_exa['rate_ex']},{res_exa['rate_in']},{scale}\n")

