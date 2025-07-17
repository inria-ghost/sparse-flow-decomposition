# Efficient Sparse Flow Decomposition Methods for RNA Multi-Assembly

Source code for the paper "Efficient Sparse Flow Decomposition Methods for RNA Multi-Assembly", Mathieu Besançon, ECAI 2025 (https://arxiv.org/abs/2501.14662).

The code cannot run without the data files which were not added for a size concern but are available at [this zenodo entry](https://zenodo.org/records/10775004).
The datasets are expected in the "imperfect_flow_dataset" folder, with the same subfolder structure as in the Zenodo entry.

- *functions.jl* contains most of the logic functions, builds the different objective functions and integer optimization models. It is imported by all experiments.
- *run_xyz.jl* files correspond to an experiment running on all instances and writing result files.
- *consolidate_xyz* files gather the corresponding JSON result files and produces plots and printed table results which were used in the paper.
- *single_poisson_case.jl* corresponds to a single hard salmon instance used to display the numerical difficulties associated with the Poisson model.

Cite the code specifically using the Zenodo DOI: https://doi.org/10.5281/zenodo.16033781
