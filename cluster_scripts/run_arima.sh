#!/bin/bash
#PBS -N arima-train
#PBS -o train_arima.out
#PBS -e train_arima.err
#PBS -m bea
#PBS -M camilo.valenzuela@alumnos.usm.cl
#PBS -l walltime=64:00:00

#FILES
#Memoria/data/processed/x_potency_canela1_merged.csv
#Memoria/data/processed/x_potency_monte_redondo_merged.csv
#Memoria/data/processed/x_potency_totoral_merged.csv

for p in $(seq 0 5); do
  for q in $(seq 0 1); do
    for d in $(seq 0 5); do
      Rscript Memoria/code/r-code/arima.R Memoria/data/processed/x_potency_canela1_merged.csv
    done
  done
done
