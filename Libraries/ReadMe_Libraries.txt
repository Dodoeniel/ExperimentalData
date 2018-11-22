This folder contains modules for Brake Squeal Analysis

to use brake squeal library use import statement as follows: from brakesqueal_library import <modulename>

The following adjustments can be made:
	- data_import: adjust tolerances and checks manually
	- model_setup: define method with new model
	- configuration: define new dataset

How to add a new dataset:
	Define a Configuration Object in Configurations and specifiy:
		- eedPath (relative to "main.py", path to folder with all eedFiles)
		- eecPath (relative to "main.py", path to folder with eec File + eecFilename + fileending!)
		- dataSetNumber