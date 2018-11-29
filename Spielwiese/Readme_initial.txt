How to use:
- Specify path of Data for each configuration in module "configuration"
- project_A_data.py: (for your own data preparation)
	- use to specify format of input data
	- use pickling module for documentation of every desired variable
- modules to extend for your own needs:
	- model_setup to define your model and loss function/optimizier
	- model_evaluation to define your plot routines, etc.
	- data_evaluation to define your plots for analysis of input data

- if use of pickling module: Check assignment of data/labels after reading from pickle (method is implemented)
- if use of logging -> specify a summarizing logfunction in a module to avoid functions full of log spam :D
