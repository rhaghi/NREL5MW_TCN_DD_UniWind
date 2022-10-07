# NREL5MW_TCN_DD_UniWind
The latest version of UniWind TCN DD for NAWEA2022 presnetation
The goal of this work is mapping wind time series to load time series of a NREL 5MW wind turbine. The wind is uniform over the rotor plane (shear included)
and the output is load time series out of OpenFAST at different channels. 

The main file is the `NREL5W_TCN_DDModel_UniWind_rev04.ipynb` which takes in the data, fit the model and plot. All the functions related to TCN are in `tcnae.py`.

The simulation output are convertet to CSV files. The file `Access_to_CSV_files.txt`
has the link to download the csv files. The png files are some tests. 
