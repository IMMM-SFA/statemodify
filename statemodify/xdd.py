"""This module is responsible for converting StateMod output .xdd files to compressed, columnar .parquet files."""

from glob import glob
from pathlib import Path
from typing import List, Type, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


class XddConverter:
    """Convert .xdd files to columnar .parquet files."""

    def __init__(
        self,
        *,
        output_path: Union[str, Path] = "./output",
        allow_overwrite: bool = False,
        xdd_files: Union[str, Path, list[Union[str, Path]]] = "**/*.xdd",
        id_subset: Union[None, list[str]] = None,
        parallel_jobs: int = 4,
        preserve_string_dtype: bool = True
    ):
        """Convert object for transforming StateMod output .xdd files into compressed, columnar .parquet files.

        :param output_path:             Path to a folder where outputs should be written; default "./output"
        :type output_path:              str

        :param allow_overwrite:         If False, abort if files already exist in the output_path; default False
        :type allow_overwrite:          bool

        :param xdd_files:               File(s) or glob(s) to the .xdd files to convert; default "**/*.xdd"
        :type xdd_files:                List[str]

        :param id_subset:               List of structure IDs to convert, or None for all; default None
        :type id_subset:                List[str]

        :param parallel_jobs:           How many files to process in parallel; default 4
        :type parallel_jobs:            int

        :param preserve_string_dtype:   Keep string parsed data instead of casting to actual type; default True
        :type preserve_string_dtype:    bool

        :example:

        .. code-block:: python

            import statemodify as stm

            converter = stm.xdd.XddConverter(
                # path to a directory where output .parquet files should be written
                output_path="./output",
                # whether to abort if .parquet files already exist at the output_path
                allow_overwrite=False,
                # path, glob, or a list of paths/globs to the .xdd files you want to convert
                xdd_files="**/*.xdd",
                # if the output .parquet files should only contain a subset of structure
                # ids, list them here; None for all
                id_subset=None,
                # how many .xdd files to convert in paralllel; optimally you will want
                #  2-4 CPUs per parallel process
                parallel_jobs=4,
            )

            converter.convert()

            # look for your output .parquet files at the output_path!

        """
        # where to write the parquet files
        self.output_path = Path(output_path)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True, exist_ok=True)
        if not self.output_path.is_dir():
            raise OSError(f"output_path '{self.output_path}' is not a directory")
        if not allow_overwrite:
            if len(list(self.output_path.glob("*.parquet"))) > 0:
                raise OSError(
                    f"parquet files already exist in {self.output_path} but allow_overwrite is False"
                )
        else:
            for f in self.output_path.glob("*.parquet"):
                f.unlink()

        # which xdd files to convert
        if isinstance(xdd_files, list):
            self.xdd_files = [
                Path(g) for sublist in [glob(p) for p in xdd_files] for g in sublist
            ]
        else:
            self.xdd_files = [Path(g) for g in glob(xdd_files, recursive=True)]
        if len(self.xdd_files) == 0:
            raise OSError(f"no .xdd files found in {xdd_files}")

        # filter to a subset of structure IDs (None means keep all)
        self.id_subset = id_subset

        # run this many jobs in parallel
        self.parallel_jobs = parallel_jobs

        # to preserve string type for spacing set to True otherwise False for natural types
        self.preserve_string_dtype = preserve_string_dtype

        # field names for each .xdd entry
        self.fields = [
            "structure_id",
            "river_id",
            "year",
            "month",
            "demand_total",
            "demand_cu",
            "from_river_by_priority",
            "from_river_by_storage",
            "from_river_by_other",
            "from_river_by_loss",
            "from_well",
            "from_carrier_by_priority",
            "from_carrier_by_other",
            "from_carrier_by_loss",
            "carried_exchange_bypass",
            "from_soil",
            "supply_total",
            "shortage_total",
            "shortage_cu",
            "water_use_cu",
            "water_use_to_soil",
            "water_use_to_other",
            "water_use_loss",
            "station_in_out_upstream_inflow",
            "station_in_out_reach_gain",
            "station_in_out_return_flow",
            "station_in_out_well_deplete",
            "station_in_out_from_to_groundwater_storage",
            "station_balance_river_inflow",
            "station_balance_river_divert",
            "station_balance_river_by_well",
            "station_balance_river_outflow",
            "available_flow",
            "control_location",
            "control_right",
        ]

        # actual field data types
        self.field_dtypes = {
            "structure_id": str,
            "river_id": str,
            "year": np.int64,
            "month": str,
            "demand_total": np.float64,
            "demand_cu": np.float64,
            "from_river_by_priority": np.float64,
            "from_river_by_storage": np.float64,
            "from_river_by_other": np.float64,
            "from_river_by_loss": np.float64,
            "from_well": np.float64,
            "from_carrier_by_priority": np.float64,
            "from_carrier_by_other": np.float64,
            "from_carrier_by_loss": np.float64,
            "carried_exchange_bypass": np.float64,
            "from_soil": np.float64,
            "supply_total": np.float64,
            "shortage_total": np.float64,
            "shortage_cu": np.float64,
            "water_use_cu": np.float64,
            "water_use_to_soil": np.float64,
            "water_use_to_other": np.float64,
            "water_use_loss": np.float64,
            "station_in_out_upstream_inflow": np.float64,
            "station_in_out_reach_gain": np.float64,
            "station_in_out_return_flow": np.float64,
            "station_in_out_well_deplete": np.float64,
            "station_in_out_from_to_groundwater_storage": np.float64,
            "station_balance_river_inflow": np.float64,
            "station_balance_river_divert": np.float64,
            "station_balance_river_by_well": np.float64,
            "station_balance_river_outflow": np.float64,
            "available_flow": np.float64,
            "control_location": str,
            "control_right": np.float64,
        }

        # field sizes for each line of interest in the .xdd file
        self.field_sizes = np.array(
            [
                11,
                13,
                5,
                5,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                8,
                13,
                12,
            ]
        )

        # the expected line length for lines of interest
        # the line separator counts as an extra character, hence the +1
        self.expected_line_size = self.field_sizes.sum() + 1

        # the names of months expected in the data
        # one of these must be in the line of the file to be parsed
        self.months = [
            "OCT",
            "NOV",
            "DEC",
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "TOT",
        ]

    def convert(self):
        """Convert .xdd files into columnar .parquet files."""
        Parallel(n_jobs=self.parallel_jobs, backend="loky")(
            delayed(XddConverter._parse_file)(
                file=file,
                fields=self.fields,
                sizes=self.field_sizes,
                line_size=self.expected_line_size,
                months=self.months,
                output_path=self.output_path,
                id_subset=self.id_subset,
            )
            for file in tqdm(self.xdd_files)
        )

    @staticmethod
    def _parse_file(
        *,
        file: Path,
        fields: list[str],
        sizes: list[int],
        line_size: int,
        months: list[str],
        output_path: str,
        id_subset: Union[None, list[str]],
    ):
        data = []
        with open(file) as f:
            structure_name = ""
            line = f.readline()
            while line:
                if "STRUCTURE NAME" in line:
                    structure_name = line.split(":")[1].strip()
                elif len(line) == line_size:
                    if any(m in line for m in months):
                        d = [structure_name]
                        position = 0
                        for count in sizes:
                            d.append(line[position : position + count].strip())
                            position += count
                        data.append(d)
                line = f.readline()
        df = pd.DataFrame(data=data, columns=["structure_name"] + fields)
        if id_subset is not None:
            df = df[df["structure_id"].isin(id_subset)]

        if self.preserve_string_dtype is False:
            df = df.astype(self.field_dtypes)

        df.to_parquet(f"{output_path}/{Path(file).stem}.parquet") 


def convert_xdd(
    *,
    output_path: Union[str, Path] = "./output",
    allow_overwrite: bool = False,
    xdd_files: Union[str, Path, list[Union[str, Path]]] = "**/*.xdd",
    id_subset: Union[None, list[str]] = None,
    parallel_jobs: int = 4,
    preserve_string_dtype: bool = True
):
    """Convert StateMod output .xdd files to compressed, columnar .parquet files.

    Easily interoperate with pandas dataframes.

    :param output_path:         Path to a folder where outputs should be written; default "./output"
    :type output_path:          str

    :param allow_overwrite:     If False, abort if files already exist in the output_path; default False
    :type allow_overwrite:      bool

    :param xdd_files:           File(s) or glob(s) to the .xdd files to convert; default "**/*.xdd"
    :type xdd_files:            List[str]

    :param id_subset:           List of structure IDs to convert, or None for all; default None
    :type id_subset:            List[str]

    :param parallel_jobs:       How many files to process in parallel; default 4
    :type parallel_jobs:        int

    :param preserve_string_dtype:   Keep string parsed data instead of casting to actual type; default True
    :type preserve_string_dtype:    bool

    :return: None
    :rtype: None

    :example:

    .. code-block:: python

        import statemodify as stm

        stm.xdd.convert_xdd(
            # path to a directory where output .parquet files should be written
            output_path="./output",
            # whether to abort if .parquet files already exist at the output_path
            allow_overwrite=False,
            # path, glob, or a list of paths/globs to the .xdd files you want to convert
            xdd_files="**/*.xdd",
            # if the output .parquet files should only contain a subset of structure ids, list them here; None for all
            id_subset=None,
            # how many .xdd files to convert in paralllel; optimally you will want 2-4 CPUs per parallel process
            parallel_jobs=4,
        )

        # look for your output .parquet files at the output_path!

    """
    XddConverter(
        output_path=output_path,
        allow_overwrite=allow_overwrite,
        xdd_files=xdd_files,
        id_subset=id_subset,
        parallel_jobs=parallel_jobs,
    ).convert()
