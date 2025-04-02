"""
Runs benchmark for exact and approximate model for different size networks.
"""

import nest
import numpy as np
import time

outfile = "benchmark_nest.csv"

dt = 0.1

##################################################
# Set parameter values, taken from (Brunel and Wang, 2001).

# fmt: off
# conductances excitatory population
#                                                               Verified from paper                                                                
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
    "tau_AMPA": 2.0,             # AMPA decay time consta7.419
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
# fmt: on


def run_sim(scale, model, seed=123):

    # number of neurons in each population
    NE = int(1600 * scale)
    NI = int(400 * scale)
    
    CE = NE
    CI = NI


    nest.ResetKernel()
    nest.set(resolution=dt,
             print_time=True,
             rng_seed=seed,
             total_num_virtual_procs=8)

    delay = 0.5

    ##################################################
    # Create neurons and devices

    excitatory_pop = nest.Create(model, NE, params=epop_params)
    inhibitory_pop = nest.Create(model, NI, params=ipop_params)

    poisson_0 = nest.Create("poisson_generator", params={"rate": 2400.0})

    sr_excitatory = nest.Create("spike_recorder", params={"time_in_steps": True})
    sr_inhibitory = nest.Create("spike_recorder", params={"time_in_steps": True})


    ##################################################
    # Define synapse specifications

    receptor_types = excitatory_pop[0].get("receptor_types")

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
    toc = time.time()

    num_ex = sr_excitatory.n_events
    num_in = sr_inhibitory.n_events

    return {"rate_ex": num_ex / NE,
            "rate_in": num_in / NI,
            "time": toc - tic
    }

with open(outfile, "w") as f:
    f.write("time_approx,time_exact,rate_ex_approx,rate_in_approx,rate_ex_exact,rate_in_exact,scale\n")

scales = [0.2, 0.5, 1, 2, 4, 8]
for i, scale in enumerate(scales):
    res_app = run_sim(scale, model="iaf_bw_2001", seed=i+1)
    res_exa = run_sim(scale, model="iaf_bw_2001_exact")

    with open(outfile, "a") as f:
        f.writelines(f"{res_app['time']},{res_exa['time']},{res_app['rate_ex']},{res_app['rate_in']},{res_exa['rate_ex']},{res_exa['rate_in']},{scale}\n")

