# Purpose
Develop a excel database modeling solar activity, air density, altitude, and wind speed at 200-300 km altitudes with the ability to adjust alitude and potentially other variables.

# File Organization
The run files are those that predict the future atmospheric temperature by taking your inputs and finding the most similar data point from 1996-2026 within the model. To run the JB2008 Model (which predicts) you must download the jb2008_analog_predictor. To execute run_msis or run_dtm (this one needs to be in wsl) you must also have the msis_dtm_analog_predictor file.

# File Information
NRLMSIS-00 Model: Utilized pymsis library to integrate NRLMSIS 2000 model. Inputs time, altitude, latitude, longitude, F10.7, 81-day average F10.7, Ap index. Inputs are directly from the atmospheric data csv file I created. Time is taken every 6 hours of the day. Outputs air density (kg/m^3) mean of the four times, standard deviation, min/max, as well as the number of atomic oxygen atoms per m^3.

DTM-2020 Model: Utilized Fortran as the Space Weather Atmosphere Models and Indices (SWAMI) from MOWA Climatological Model (MCM) was written in Fortran. The model calls MCM which is contained inside the SWAMI library with inputs from the csv  atmospheric data I created. Same outputs as MSIS-00: air density and atomic oxygen number density.

JB2008: Utilized pyatmos library which required an older version of python (python 3.12). Pyatmos has the existing data and space weather csv did not need to be used. Outputs air density and neutral temperature( the average kinetic energy of air molecules, useful for finding the air density). 

# Master Data Sheet of Historical Data:
Contains all data collected from the models which can be used to correlate their strengths and weaknesses. (This is a large file and may take some time to load): 
https://mitprod-my.sharepoint.com/:x:/g/personal/jt23_mit_edu/IQBbncqiUT7eSJxOW7EaLaYpAdT10qaf6AKefasRNTqoDFM?e=tryhIj
