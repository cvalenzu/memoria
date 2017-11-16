#!/bin/bash
#PBS -N best_sarimas
#PBS -o train_arima_$PBS_JOBID.out
#PBS -e train_arima_$PBS_JOBID.err
#PBS -l walltime=200:00:00


#RUN BEST ARIMA
#FILES
#Memoria/data/processed/x_potency_canela1_merged.csv
#Memoria/data/processed/x_potency_monte_redondo_merged.csv
#Memoria/data/processed/x_potency_totoral_merged.csv

use gcc63 boost

# Running ARIMA
Rscript Memoria/code/r-code/generate_results_arima.R Memoria/data/processed/x_potency_canela1_merged.csv 1 1 0 0 0 0
Rscript Memoria/code/r-code/generate_results_arima.R Memoria/data/processed/x_potency_monte_redondo_merged.csv 0 1 1 0 0 0
Rscript Memoria/code/r-code/generate_results_arima.R Memoria/data/processed/x_potency_totoral_merged.csv 0 1 1 0 0 0

# Running SARIMA
Rscript Memoria/code/r-code/generate_results_arima.R Memoria/data/processed/x_potency_canela1_merged.csv 1 1 2 1 0 1
Rscript Memoria/code/r-code/generate_results_arima.R Memoria/data/processed/x_potency_monte_redondo_merged.csv 2 1 2 1 0 1
Rscript Memoria/code/r-code/generate_results_arima.R Memoria/data/processed/x_potency_totoral_merged.csv 3 1 1 1 0 1
