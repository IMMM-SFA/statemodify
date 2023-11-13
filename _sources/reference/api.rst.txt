.. _pythonapi:

Python API
=================

**statemodify** offers a programmatic API in Python.

.. note::

  For questions or request for support, please reach out to the development team.
  Your feedback is much appreciated in evolving this API!


Input Modification
---------------------------------

statemodify.modify_eva
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_eva

statemodify.modify_single_eva
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_single_eva

statemodify.modify_ddm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_ddm

statemodify.modify_single_ddm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_single_ddm

statemodify.modify_ddr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_ddr

statemodify.modify_single_ddr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_single_ddr

statemodify.apply_on_off_modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.apply_on_off_modification

statemodify.apply_seniority_modification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.apply_seniority_modification

statemodify.modify_xbm_iwr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_xbm_iwr

statemodify.modify_xbm_iwr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_single_xbm_iwr

statemodify.get_reservoir_structure_ids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.get_reservoir_structure_ids

statemodify.modify_single_res
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_single_res

statemodify.modify_res
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_res



HMM Functions
---------------------------------

statemodify.hmm_multisite_fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.hmm_multisite_fit

statemodify.hmm_multisite_sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.hmm_multisite_sample

statemodify.get_samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.get_samples

statemodify.generate_dry_state_means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_dry_state_means

statemodify.generate_wet_state_means
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_wet_state_means

statemodify.generate_dry_covariance_matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_dry_covariance_matrix

statemodify.generate_wet_covariance_matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_wet_covariance_matrix

statemodify.generate_transition_matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_transition_matrix 

statemodify.calculate_array_monthly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.calculate_array_monthly

statemodify.calculate_array_annual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.calculate_array_annual 

statemodify.calculate_annual_sum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.calculate_annual_sum 

statemodify.calculate_annual_mean_fractions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.calculate_annual_mean_fractions

statemodify.fit_iwr_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.fit_iwr_model


statemodify.generate_hmm_inputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_hmm_inputs

statemodify.generate_flows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_flows 


statemodify.generate_modified_file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_modified_file


Output Modification
---------------------------------

statemodify.convert_xdd
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.convert_xdd

statemodify.read_xre
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.read_xre

statemodify.extract_xre_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.extract_xre_data


Sampling
---------------------------------

statemodify.build_problem_dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.build_problem_dict

statemodify.generate_samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_samples

statemodify.validate_modify_dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.validate_modify_dict

statemodify.generate_sample_iwr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_sample_iwr

statemodify.generate_sample_all_params
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_sample_all_params

Modification
---------------------------------

statemodify.set_alignment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.set_alignment

statemodify.pad_with_spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.pad_with_spaces

statemodify.add_zero_padding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.add_zero_padding

statemodify.populate_dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.populate_dict

statemodify.prep_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.prep_data

statemodify.construct_outfile_name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.construct_outfile_name

statemodify.construct_data_string
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.construct_data_string

statemodify.apply_adjustment_factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.apply_adjustment_factor

statemodify.validate_bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.validate_bounds

statemodify.Modify
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.Modify

Batch Modification
---------------------------------

statemodify.get_required_arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.get_required_arguments

statemodify.get_arguments_values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.get_arguments_values 

statemodify.generate_parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.generate_parameters

statemodify.modify_batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.modify_batch

Visualization
---------------------------------

statemodify.plot_flow_duration_curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.plot_flow_duration_curves

statemodify.plot_res_quantiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.plot_res_quantiles

statemodify.plot_reservoir_boxes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.plot_reservoir_boxes

Utilities
---------------------------------

statemodify.yaml_to_dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.yaml_to_dict


statemodify.select_template_file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.select_template_file


statemodify.select_data_specification_file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: statemodify.select_data_specification_file
