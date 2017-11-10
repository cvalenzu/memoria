#!/bin/bash
#PBS -N esn_training_one
#PBS -o esn_one_$PBS_JOBID.out
#PBS -e esn_one_$PBS_JOBID.err
#PBS -l walltime=200:00:00
echo "Loading Anaconda"
use gcc63 boost anaconda3
echo "Running Script"
cd Memoria/code/python-code

files=(../../data/processed/x_potency_canela1_merged.csv ../../data/processed/x_potency_monte_redondo_merged.csv ../../data/processed/x_potency_totoral_merged.csv)

lags=(1 6 12 24)
for file in ${files[@]};do
  echo "Processing $file"
  for lag in ${lags[@]};do
    python one_out.py $file --inputs $lag
  done
done
