[![build](https://github.com/IMMM-SFA/statemodify/actions/workflows/build.yml/badge.svg)](https://github.com/IMMM-SFA/statemodify/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/IMMM-SFA/statemodify/branch/main/graph/badge.svg?token=csQBZMRSdp)](https://codecov.io/gh/IMMM-SFA/statemodify)
[![Static Badge](https://img.shields.io/badge/Powered%20by-MSDLIVE-blue?label=Powered%20by&color=blue)](https://statemodify.msdlive.org)
[![DOI](https://zenodo.org/badge/484620418.svg)](https://zenodo.org/doi/10.5281/zenodo.10258007)
[![pre-commit](https://github.com/IMMM-SFA/statemodify/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/IMMM-SFA/statemodify/actions/workflows/pre-commit.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06325/status.svg)](https://doi.org/10.21105/joss.06325)

# statemodify
`statemodify` is an open-source Python package that provides users with a way to easily modify [StateMod](https://github.com/OpenCDSS/cdss-app-statemod-fortran)'s input and output files to enable exploratory modeling. StateMod is written in Fortran and conducting high-performance computing enabled ensemble exploratory modeling with it requires a systematic and automated approach. Due to the model’s complexity, there are also nontrivial computational challenges in comprehensively sampling the model’s input space and managing the outputs of interest, especially for large ensembles. These challenges limit its use among researchers and broader operational users. Thus, we developed `statemodify`, a Python package and framework that allows users to easily interact with StateMod using Python exclusively. The user can implement statemodify functions to manipulate StateMod’s input files to develop alternative demand, hydrology, infrastructure, and institutional scenarios for Colorado’s West Slope basins and run these scenarios through StateMod. We also create methods to compress and extract model output into easily readable data frames and provide guidance on analysis and visualization of output in a series of Jupyter notebooks that step through the functionality of the package.

## Documentation
Full documentation and tutorials are provided [here](https://immm-sfa.github.io/statemodify).

## Installation
Install `statemodify` using pip:
```bash
pip install statemodify
```

## Online quickstarter tutorials
Take a `statemodify` for a spin in one of our no-install Jupyter quickstarters [here](https://statemodify.msdlive.org):

- **Notebook 1**:  Getting Started and using the DDM and DDR modification functions in the San Juan Subbasin.  This notebook has more general intro information on `statemodify` and shows how changes to demand and water rights can lead to changes to user shortages in the San Juan Subbasin.

- **Notebook 2**:  Using the EVA modification functions in the Gunnison Subbasin.  This notebook looks at how changes in evaporation in reservoirs in the Gunnison subbasin lead to changes to reservoir levels.

- **Notebook 3**:  Using the RES modification function in the Upper Colorado Subbasin.  This notebook looks at how changes in storage in reservoirs in the Upper Colorado subbasin lead to changes to user shortages.

- **Notebook 4**:  Using the XBM and IWR modification functions across all basins.  This notebook debuts the stationary Hidden Markov Model to generate alternative streamflow scenarios across the basins.

- **Notebook 5**:  Sampling multiple uncertainties.  This notebook demonstrates how to create a global Latin hypercube sample to consider multiple uncertainties in a basin.

## Contributing to `statemodify`
Whether you find a typo in the documentation, find a bug, or want to develop functionality that you think will make statemodify more robust, you are welcome to contribute!  See the full contribution guidelines in our [online documentation](https://immm-sfa.github.io/statemodify/reference/contributing.html).
