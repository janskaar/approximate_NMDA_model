# A simplified model of NMDA-receptor-mediated dynamics in leaky integrate-and-fire neurons

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs10827--025--00911--8-blue)](https://doi.org/10.1007/s10827-025-00911-8)

Supplementary code for

> Skaar, J.E.W., Haug, N. & Plesser, H.E. (2025). A simplified model of NMDA-receptor-mediated dynamics in leaky integrate-and-fire neurons. *Journal of Computational Neuroscience* 53, 475–487. https://doi.org/10.1007/s10827-025-00911-8

The paper proposes an approximation to the Wang (1999) / Brunel & Wang (2001) model of NMDA-receptor-mediated synaptic currents that can be simulated efficiently for arbitrary network connectivity and delay distributions. Both the approximate model (`iaf_bw_2001`) and the original model (`iaf_bw_2001_exact`) are included in the [NEST simulator](https://nest-simulator.readthedocs.io/) from version 3.8. This repository contains the scripts used to generate the data and figures in the paper.

<details>
<summary>Abstract</summary>

A model for NMDA-receptor-mediated synaptic currents in leaky integrate-and-fire neurons, first proposed by Wang (J Neurosci, 1999), has been widely studied in computational neuroscience. The model features a fast rise in the NMDA conductance upon spikes in a pre-synaptic neuron followed by a slow decay. In a general implementation of this model which allows for arbitrary network connectivity and delay distributions, the summed NMDA current from all neurons in a pre-synaptic population cannot be simulated in aggregated form. Simulating each synapse separately is prohibitively slow for all but small networks, which has largely limited the use of the model to fully connected networks with identical delays, for which an efficient simulation scheme exists. We propose an approximation to the original model that can be efficiently simulated for arbitrary network connectivity and delay distributions. Our results demonstrate that the approximation incurs minimal error and preserves network dynamics. We further use the approximate model to explore binary decision making in sparsely coupled networks.

</details>

## Using the model

If you only want to use the model, you do not need this repository. The models ship with NEST, and the NEST documentation describes them and provides a complete decision-making network example:

- [`iaf_bw_2001`](https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001.html) — the approximate model proposed in the paper
- [`iaf_bw_2001_exact`](https://nest-simulator.readthedocs.io/en/stable/models/iaf_bw_2001_exact.html) — the original model, with every synapse simulated explicitly
- [Decision-making network example](https://nest-simulator.readthedocs.io/en/stable/auto_examples/wang_decision_making.html) — Wang (2002) binary decision-making network using the approximate model

## Dependencies

- **NEST 3.8** ([Zenodo](https://doi.org/10.5281/zenodo.12624784)). The two neuron models used, `iaf_bw_2001` and `iaf_bw_2001_exact`, are part of the NEST distribution, so no custom build is required.
- **Brian2** 2.6–2.9, used only by the two benchmark scripts.
- **numpy, scipy, matplotlib, seaborn, pandas, h5py**.

## Reproducing the figures

The scripts fall into three groups. The first group runs on a laptop; the other two were run on the JURECA supercomputer at Jülich Supercomputing Centre and depend on SLURM.

### Self-contained figure scripts (no cluster needed)

These run their own NEST simulations and write the figure as a PDF.

| Figure | Script | Notes |
|---|---|---|
| Fig. 1 | `figure1.py` | Single post-synaptic neuron, exact vs. approximate model. |
| Fig. 2 | `figure2.py` | RMS voltage error vs. pre-synaptic population size. Uses 4 threads. |
| Fig. 3 | `figure3.py` | Full decision-making network at three coherence levels. Set the `model` variable at the top of the script to `"iaf_bw_2001"` (approximate) or `"iaf_bw_2001_exact"` (exact). Spike data are cached to a pickle file, so a second run only replots. Uses 4 threads; the exact model is slow. |

### Figures that need data generated on a cluster

| Figure | Data-generating script(s) | Figure script |
|---|---|---|
| Fig. 4 | `decision_making_varying_coherence.py` | `figure4.py` |
| Fig. 5 | `nest_benchmark.py`, `brian_benchmark.py`, `brian_benchmark_explicit.py` (via `run_benchmarks_*.job`) | `figure5.py` |
| Fig. 6 | `decision_making_sparse.py`, `decision_making_sparse_adjusted.py`, then `extract_decision_making_data.py` | `figure6.py` |

- **Fig. 4.** `decision_making_varying_coherence.py` runs 16 simulations per coherence level for both models and writes an HDF5 file into `decision_making_results/`. It must be run 25 times (with different runner IDs) to obtain the 400 simulations per coherence level used in the paper. `figure4.py` reads every file in that directory.
- **Fig. 5.** Each benchmark script is run once per network scale by the corresponding job script and appends one row to a CSV in `benchmarking_data_<n>_threads/`. `figure5.py` expects two files, `wang_benchmark_nest.csv` and `wang_benchmark_brian.csv`, in the working directory. The per-runner CSVs were concatenated into these by hand; that step is not in the repository. The unrestricted Brian2 implementation (`brian_benchmark_explicit.py`) is likewise not read by `figure5.py` directly.
- **Fig. 6.** The two `decision_making_sparse*.py` scripts write `runner_<id>.h5` files, one per runner, for runner IDs 1–8. `extract_decision_making_data.py` collects these into a pickle (toggle the two `data_dir` and output lines near the top and bottom of the script to process each dataset). Note that `figure6.py` reads `decision_making_sparse_results.pkl` and `decision_making_sparse_adjusted_results.pkl`, while the extraction script writes `decision_making_sparse.pkl` and `decision_making_sparse_adjusted.pkl`; rename the files accordingly.

### Running on a cluster

The data-generating scripts read `SLURM_CPUS_PER_TASK` (and `decision_making_varying_coherence.py` also reads `SLURM_JOB_ID`) from the environment and will fail outside a SLURM job. To run them elsewhere, set these variables manually.

The three `run_benchmarks_*.job` files are the SLURM scripts used on JURECA. They contain site-specific details (compute account, module names, a `jemalloc.sh` helper, and the path to the NEST installation) that must be adapted to your system.

The scripts are provided as they were used for the paper and have not been refactored.

## Citation

```bibtex
@article{Skaar2025,
  author  = {Skaar, Jan-Eirik Welle and Haug, Nicolai and Plesser, Hans Ekkehard},
  title   = {A simplified model of {NMDA}-receptor-mediated dynamics in leaky integrate-and-fire neurons},
  journal = {Journal of Computational Neuroscience},
  volume  = {53},
  number  = {3},
  pages   = {475--487},
  year    = {2025},
  doi     = {10.1007/s10827-025-00911-8}
}
```

## License

This code is released under the [MIT License](LICENSE).
