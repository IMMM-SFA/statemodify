``statemodify`` Quickstarter Notebook #5 : Sampling Multiple Uncertainties
--------------------------------------------------------------------------

In the prior notebooks, we sampled just one type of uncertainty at a
time to demonstrate how a single adjustment in the selected input file
leads to a verifiable change in output so that we can demonstrate that
``statemodify`` is working as expected. It is much harder to make sense
of the relative importance of uncertain drivers unless we conduct a
formal sensitivity analysis, which lies outside of the bounds of this
tutorial. However, it is very likely that many of these uncertainties
will be present and of interest in any given future for the region.
Thus, this notebook is used to demonstrate how to do a Latin hypercube
sample simultaneously across multiple uncertainties in a given basin
using the ``modify_batch()`` function.

.. code:: ipython3

    import argparse
    import logging
    import os
    import pickle
    from string import Template
    import subprocess
    
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd 
    import statemodify as stm

.. container:: alert alert-block alert-info

   NOTE: Each simulation in this notebook is run for the length of the
   historical period (from 1909-2013). If you want to reduce the length
   of the simulation, navigate to the ``.ctl`` file and adjust the
   ``iystr`` and ``iyend`` variables. For this notebook, this file is
   located in: ``data/cm2015_StateMod/StateMod/cm2015.ctl``

.. code:: ipython3

    # statemod directory
    statemod_dir = "/usr/src/statemodify/statemod_upper_co"
    
    # root directory of statemod data for the target basin
    root_dir = os.path.join(statemod_dir, "src", "main", "fortran")
    
    # home directory of notebook instance
    home_dir = os.path.dirname(os.getcwd())
    
    # path to the statemod executable
    statemod_exe = os.path.join(root_dir, "statemod")
    
    # data directory and root name for the target basin
    data_dir = os.path.join(
        home_dir,
        "data",
        "cm2015_StateMod",
        "StateMod"
    )
    
    # directory to the target basin input files with root name for the basin
    basin_path = os.path.join(data_dir, "cm2015B")
    
    # scenarios output directory
    scenarios_dir = os.path.join(data_dir, "scenarios")
    
    #parquet output directory
    parquet_dir=os.path.join(data_dir, "parquet")
    
    
    # path to template file
    multi_template_file = os.path.join(
        home_dir,
        "data",
        "cm2015B_template_multi.rsp"
    )

Step 1: Creating a Sample Across Multiple Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we create a global Latin hypercube sample across 3
example uncertainties that we are interested in for the Upper Colorado:
evaporation at reservoirs, modification of water rights, and demands.
The form of the ``modify_batch()`` function is similiar to those
presented in the last notebooks; however, now, a problem dictionary
stores the names of the respective ``statemodify`` functions
(``modify_eva``, ``modify_ddr``, ``modify_ddm``) that will now be
applied simultaneously. The Latin hypercube sample is conducted with
respect to the ``bounds`` listed and the resulting multipliers or
additives are applied to the target IDs listed.

.. container:: alert alert-block alert-info

   NOTE: If you are interested in creating an alternative ``.ddr`` file
   that does not sample decrees, only adjusts the water rights, then you
   will need to write ``None`` in the “bounds” parameter as we do below.
   If you include bounds, then decrees as well as water rights will be
   adjusted simultaneously.

.. code:: ipython3

    import statemodify as stm
    
    # variables that apply to multiple functions
    output_dir = os.path.join(data_dir, "input_files")
    basin_name = "Upper_Colorado"
    scenario = "1"
    seed_value = 77
    
    # problem dictionary
    problem_dict = {
        "n_samples": 1,
        'num_vars': 3,
        'names': ['modify_eva', 'modify_ddr', 'modify_ddm'],
        'bounds': [
            [-0.5, 1.0],
            None,
            [0.5, 1.0]
        ],
        # additional settings for each function
        "modify_eva": {
            "seed_value": seed_value,
            "output_dir": output_dir,
            "scenario": scenario,
            "basin_name": basin_name,
            "query_field": "id",
            "ids": ["10008", "10009"]
        },
        "modify_ddr": {
            "seed_value": seed_value,
            "output_dir": output_dir,
            "scenario": scenario,
            "basin_name": basin_name,
            "query_field": "id",
            "ids": ["3600507.01", "3600507.02"],
            "admin": [1, None],
            "on_off": [1, 1]
        },
        "modify_ddm": {
            "seed_value": seed_value,
            "output_dir": output_dir,
            "scenario": scenario,
            "basin_name": basin_name,
            "query_field": "id",
            "ids": ["3600507", "3600603"]
        }
    }
    
    # run in batch
    fn_parameter_dict = stm.modify_batch(problem_dict=problem_dict)


.. parsed-literal::

    Running modify_eva
    Running modify_ddr
    Running modify_ddm


Step 2: Running a Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have developed the samples, we need to adjust our template
file to take in the additional uncertainties and then we can run our
simulation! Note that in contrast to the other notebooks, we are
changing the “EVA”, “DDM”, and “DDR” entries in the ``.rsp`` file at the
same time, running the simulation, and then extracting the shortages for
a specific user (ID: 3601008).

.. container:: alert alert-block alert-info

   NOTE In order to expedite simulations for the Upper Colorado dataset,
   make sure to turn off “Reoperation” mode. You can do so by opening
   ``/home/jovyan/data/cm2015_StateMod/StateMod/cm2015.ctl``, navigating
   to the ``ireopx`` entry and changing the value from “0” to “10”.

.. code:: ipython3

    # set realization and sample
    realization = 1
    sample = np.arange(0, 1, 1)
    
    # read RSP template
    with open(multi_template_file) as template_obj:
        
        # read in file
        template_rsp = Template(template_obj.read())
    
        for i in sample:
            
            # create scenario name
            scenario = f"S{i}_{realization}"
            
            # dictionary holding search keys and replacement values to update the template file
            d = {"EVA": f"../../input_files/cm2015B_{scenario}.eva","DDM": f"../../input_files/cm2015B_{scenario}.ddm","DDR": f"../../input_files/cm2015B_{scenario}.ddr"}
            
            # update the template
            new_rsp = template_rsp.safe_substitute(d)
            
            # construct simulated scenario directory
            simulated_scenario_dir = os.path.join(scenarios_dir, scenario)
            if not os.path.exists(simulated_scenario_dir):
                os.makedirs(simulated_scenario_dir)
                
            # target rsp file
            rsp_file = os.path.join(simulated_scenario_dir, f"cm2015B_{scenario}.rsp")
            
            # write updated rsp file
            with open(rsp_file, "w") as f1:
                f1.write(new_rsp)
            
            # construct simulated basin path
            simulated_basin_path = f"cm2015B_{scenario}"
    
            # run StateMod
            print(f"Running: {scenario}")
            os.chdir(simulated_scenario_dir)
    
            subprocess.call([statemod_exe, simulated_basin_path, "-simulate"])
            
            #Save output to parquet files 
            print('creating parquet for ' + scenario)
            
            output_directory = os.path.join(parquet_dir+ "/scenario/"+ scenario)
            
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            
            stm.xdd.convert_xdd(output_path=output_directory,allow_overwrite=False,xdd_files=scenarios_dir + "/"+ scenario + "/cm2015B_"+scenario+".xdd",id_subset=['3601008'],parallel_jobs=2)



.. parsed-literal::

    Running: S0_1
      Parse; Command line argument: 
      cm2015B_S0_1 -simulate                                                                                                         
    ________________________________________________________________________
    
            StateMod                       
            State of Colorado - Water Supply Planning Model     
    
            Version: 15.00.01
            Last revision date: 2015/10/28
    
    ________________________________________________________________________
    
      Opening log file cm2015B_S0_1.log                                                                                                                                                                                                                                                
      
      Subroutine Execut
      Subroutine Datinp
    
    ...



At the end of the simulation, the output is the file,
``cm2015B_S0_1.parquet``, which now contains the shortages for the
target ID for the length of the simulation. The user can then proceed to
do similiar analyses on water shortages that have been demonstrated in
the prior notebooks.

.. container:: alert alert-block alert-warning

   Tip: If you are interested in understanding how to apply
   ``statemodify`` functions to your own model, take a look at the
   source code found in the repository here:

   .. container::

      ::

         1.  <a href="https://github.com/IMMM-SFA/statemodify/blob/main/statemodify/batch.py">modify_batch()</a>

