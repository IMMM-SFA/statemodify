[![build](https://github.com/IMMM-SFA/statemodify/actions/workflows/build.yml/badge.svg)](https://github.com/IMMM-SFA/statemodify/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/IMMM-SFA/statemodify/branch/main/graph/badge.svg?token=csQBZMRSdp)](https://codecov.io/gh/IMMM-SFA/statemodify)
[![Static Badge](https://img.shields.io/badge/Powered%20by-MSDLIVE-blue?label=Powered%20by&color=blue)](https://statemodify.msdlive.org)


# statemodify
A package to modify StateMod's input and output files for exploratory modeling

## In development
While in development, install `statemodify` using the following steps:
- Clone the repo using `git clone https://github.com/IMMM-SFA/statemodify.git`
- Navigate to the directory in which `statemodify` was cloned and run `python setup.py develop`.  Ensure your `python` points to a Python 3.8 and up instance.
- Test the install by entering into a Python prompt and running:  `import statemodify`, if no errors, you are all good.

## Functionality
### Generate a set of .ddm files using a LHS
```python
import statemodify as stm

# a dictionary to describe what you want to modify and the bounds for the LHS
setup_dict = {
    "names": ["municipal", "standard"],
    "ids": [["7200764", "7200813CH"], ["7200764_I", "7200818"]],
    "bounds": [[-1.0, 1.0], [-1.0, 1.0]]
}

output_directory = "<your desired output directory>"
scenario = "<your scenario name>"

# the number of samples you wish to generate
n_samples = 4

# seed value for reproducibility if so desired
seed_value = None

# my template file.  If none passed into the `modify_ddm` function, the default file will be used.
template_file = "<your ddm template file>"

# the field that you want to use to query and modify the data
query_field = "id"

# generate a batch of files using generated LHS
stm.modify_ddm(modify_dict=setup_dict,
               output_dir=output_directory,
               scenario=scenario,
               n_samples=n_samples,
               seed_value=seed_value,
               query_field=query_field,
               template_file=template_file)
```

### Convert output .xdd files to .parquet files
Parquet files are efficient columnar data stores that easily interoperate with pandas dataframes.
```python
from statemodify.xdd import XddConverter

# set up the converter
converter = XddConverter(
    output_path='./output',
    allow_overwrite=False,
    xdd_files='**/*.xdd',
    id_subset=None,
    parallel_jobs=4,
)

# convert the files
converter.convert()

# look for your parquet files in './output'!

```
