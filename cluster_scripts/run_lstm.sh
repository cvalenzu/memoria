#!/bin/bash
#PBS -N lstm-train
#PBS -o lstm_train_$PBS_JOBID.out
#PBS -e lstm_train_$PBS_JOBID.err
#PBS -l walltime=200:00:00

use gcc63 boost anaconda3

files=(../../data/windowed/X_canela1.csv ../../data/windowed/X_monte.csv ../../data/windowed/X_totoral.csv)

lags=(6 12 24)
outputs=(12)

cd Memoria/code/python-code/
for file in ${files[@]};do
  echo "Processing $file"
  for output in ${outputs[@]}; do
    for lag in ${lags[@]};do
      python lstm.py  $file --inputs $lag --outputs $output --epochs 10
    done
  done
done
