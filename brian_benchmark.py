"""
Benchmarking script for Brian2, adapted from https://brian2.readthedocs.io/en/2.7.1/examples/frompapers.Wang_2002.html . To enable multiple threads, the network must be compiled and run separately . The script therefore has to be run once for each network size.
"""

from brian2 import *
import os, time, sys
import numpy as np

outfile = "benchmark_brian.csv"

scale = 8

# Makes brian use OMP, see https://brian2.readthedocs.io/en/stable/user/computation.html#multi-threading-with-openmp
prefs.devices.cpp_standalone.openmp_threads = 8

set_device("cpp_standalone", directory="brian_benchmark_standalone", build_on_run=False)

# -----------------------------------------------------------------------------------------------
# Set up the simulation
# -----------------------------------------------------------------------------------------------

runtime = 1000 * ms  # total simulation time

# External noise inputs
rate_ext = 2400 * Hz # external Poisson rate for excitatory population

# Network parameters
N = int(2000 * scale)  # number of neurons
f_inh = 0.2  # fraction of inhibitory neurons
NE = int(N * (1.0 - f_inh))  # number of excitatory neurons (1600)
NI = int(N * f_inh)  # number of inhibitory neurons (400)

# Neuron parameters
El = -70.0 * mV  # resting potential
Vt = -50.0 * mV  # firing threshold
Vr = -55.0 * mV  # reset potential
CmE = 0.5 * nF  # membrane capacitance for pyramidal cells (excitatory neurons)
CmI = 0.2 * nF  # membrane capacitance for interneurons (inhibitory neurons)
gLeakE = 25.0 * nS  # membrane leak conductance of excitatory neurons
gLeakI = 20.0 * nS  # membrane leak conductance of inhibitory neurons
refE = 1.9 * ms  # refractory periodof excitatory neurons
refI = 0.9 * ms  # refractory period of inhibitory neurons

# Synapse parameters
V_E = 0. * mV  # reversal potential for excitatory synapses
V_I = -70. * mV  # reversal potential for inhibitory synapses
tau_AMPA = 2.0 * ms  # AMPA synapse decay
tau_NMDA_rise = 2.0 * ms  # NMDA synapse rise
tau_NMDA_decay = 100.0 * ms  # NMDA synapse decay
tau_GABA = 5.0 * ms  # GABA synapse decay
alpha = 0.5 * kHz  # saturation of NMDA channels at high presynaptic firing rates
C = 1 * mmole  # extracellular magnesium concentration

# Synaptic conductances
gextE = 2.1 * nS  # external -> excitatory neurons (AMPA)
gextI = 1.62 * nS  # external -> inhibitory neurons (AMPA)
gEEA = 0.05 * nS / NE * 1600  # excitatory -> excitatory neurons (AMPA)
gEIA = 0.04 * nS / NE * 1600  # excitatory -> inhibitory neurons (AMPA)
gEEN = 0.165 * nS / NE * 1600  # excitatory -> excitatory neurons (NMDA)
gEIN = 0.13 * nS / NE * 1600  # excitatory -> inhibitory neurons (NMDA)
gIE = 1.3 * nS / NI * 400  # inhibitory -> excitatory neurons (GABA)
gII = 1.0 * nS / NI * 400  # inhibitory -> inhibitory neurons (GABA)

# Neuron equations
# Note the "(unless refractory)" statement serves to clamp the membrane voltage during the refractory period;
# otherwise the membrane potential continues to be integrated but no spikes are emitted.
eqsE = """
   label : integer (constant)  # label for decision encoding populations
   dV/dt = (- gLeakE * (V - El) - I_AMPA - I_NMDA - I_GABA - I_AMPA_ext + I_input) / CmE : volt (unless refractory)

   I_AMPA = s_AMPA * (V - V_E) : amp
   ds_AMPA / dt = - s_AMPA / tau_AMPA : siemens

   I_NMDA = gEEN * s_NMDA_tot * (V - V_E) / ( 1 + exp(-0.062 * V/mvolt) * (C/mmole / 3.57) ) : amp
   s_NMDA_tot : 1

   I_GABA = s_GABA * (V - V_I) : amp
   ds_GABA / dt = - s_GABA / tau_GABA : siemens

   I_AMPA_ext = s_AMPA_ext * (V - V_E) : amp
   ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : siemens

   I_input : amp

   ds_NMDA / dt = - s_NMDA / tau_NMDA_decay + alpha * x * (1 - s_NMDA) : 1
   dx / dt = - x / tau_NMDA_rise : 1
"""

eqsI = """
   dV/dt = (- gLeakI * (V - El) - I_AMPA - I_NMDA - I_GABA - I_AMPA_ext) / CmI : volt (unless refractory)

   I_AMPA = s_AMPA * (V - V_E) : amp
   ds_AMPA / dt = - s_AMPA / tau_AMPA : siemens

   I_NMDA = gEIN * s_NMDA_tot * (V - V_E) / ( 1 + exp(-0.062 * V/mvolt) * (C/mmole / 3.57) ): amp
   s_NMDA_tot : 1

   I_GABA = s_GABA * (V - V_I) : amp
   ds_GABA / dt = - s_GABA / tau_GABA : siemens

   I_AMPA_ext = s_AMPA_ext * (V - V_E) : amp
   ds_AMPA_ext / dt = - s_AMPA_ext / tau_AMPA : siemens
"""


defaultclock.dt = 0.1 * ms

# Neuron populations
popE = NeuronGroup(NE, model=eqsE, threshold='V > Vt', reset='V = Vr', refractory=refE, name='popE', method="rk4")
popI = NeuronGroup(NI, model=eqsI, threshold='V > Vt', reset='V = Vr', refractory=refI, name='popI', method="rk4")
popE.label = 0

# Recurrent excitatory -> excitatory connections mediated by AMPA receptors
C_EE_AMPA = Synapses(popE, popE, 'w : siemens', on_pre='s_AMPA += w', delay=0.5 * ms, name='C_EE_AMPA', method="rk4")
C_EE_AMPA.connect()
C_EE_AMPA.w[:] = gEEA


# Recurrent excitatory -> inhibitory connections mediated by AMPA receptors
C_EI_AMPA = Synapses(popE, popI, on_pre='s_AMPA += gEIA', delay=0.5 * ms, name='C_EI_AMPA', method="rk4")
C_EI_AMPA.connect()

# Recurrent excitatory -> excitatory connections mediated by NMDA receptors
C_EE_NMDA = Synapses(popE, popE, on_pre='x_pre += 1', delay=0.5 * ms, name='C_EE_NMDA', method="rk4")
C_EE_NMDA.connect(j='i')

# Dummy population to store the summed activity of the three populations
NMDA_sum_group = NeuronGroup(1, 's : 1', name='NMDA_sum_group', method="rk4")

# Sum the activity according to the subpopulation labels
NMDA_sum = Synapses(popE, NMDA_sum_group, 's_post = s_NMDA_pre : 1 (summed)', name='NMDA_sum', method="rk4")
NMDA_sum.connect(j='label_pre')

# Propagate the summed activity to the NMDA synapses
NMDA_set_total_E = Synapses(NMDA_sum_group, popE,
                            '''w : 1 (constant)
                               s_NMDA_tot_post = w*s_pre : 1 (summed)''', name='NMDA_set_total_E', method="rk4")
NMDA_set_total_E.connect()
NMDA_set_total_E.w = 1

# Recurrent excitatory -> inhibitory connections mediated by NMDA receptors
NMDA_set_total_I = Synapses(NMDA_sum_group, popI,
                            '''s_NMDA_tot_post = s_pre : 1 (summed)''', name='NMDA_set_total_I')
NMDA_set_total_I.connect()

# Recurrent inhibitory -> excitatory connections mediated by GABA receptors
C_IE = Synapses(popI, popE, on_pre='s_GABA += gIE', delay=0.5 * ms, name='C_IE')
C_IE.connect()

# Recurrent inhibitory -> inhibitory connections mediated by GABA receptors
C_II = Synapses(popI, popI, on_pre='s_GABA += gII', delay=0.5 * ms, name='C_II')
C_II.connect()

# External inputs (fixed background firing rates)
extinputE = PoissonInput(popE, 's_AMPA_ext', 1, rate_ext, gextE)
extinputI = PoissonInput(popI, 's_AMPA_ext', 1, rate_ext, gextI)

# -----------------------------------------------------------------------------------------------
# Run the simulation
# -----------------------------------------------------------------------------------------------

# Set initial conditions
popE.s_NMDA_tot = 0
popI.s_NMDA_tot = 0
popE.V = -70 * mV
popI.V = -70 * mV


# Record population activity
RE = PopulationRateMonitor(popE)
RI = PopulationRateMonitor(popI)

if not os.path.isfile(outfile):
    with open(outfile, "w") as f:
        f.write("time_brian,n_threads,rate_ex,rate_in,scale\n")

run(runtime)
device.build(run=False)

# time run only
tic = time.time()
device.run()
toc = time.time()

print(f"Execution time: {toc - tic}")

rate_E = (RE.rate / Hz).mean()
rate_I = (RI.rate / Hz).mean()

with open(outfile, "a") as f:
    f.writelines(f"{toc - tic},{rate_E},{rate_I},{scale}\n")

