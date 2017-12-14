#!/bin/bash
#PBS -N esn_training_multi
#PBS -o esn_multi_$PBS_JOBID.out
#PBS -e esn_multi_$PBS_JOBID.err
#PBS -l walltime=200:00:00
echo "Loading Anaconda"
use gcc63 boost anaconda3
echo "Running Script"
cd Memoria/code/python-code

files=(../../data/windowed/X_canela1.csv ../../data/windowed/X_monte.csv ../../data/windowed/X_totoral.csv)

lags=(1 6 12 24)
for file in ${files[@]};do
  echo "Processing $file"
  for lag in ${lags[@]};do
    python multi_out.py $file --inputs $lag
  done
done
