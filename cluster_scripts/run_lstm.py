#!/bin/bash
#PBS -N lstm-train
#PBS -o lstm_train_$PBS_JOBID.out
#PBS -e lstm_train_$PBS_JOBID.err
#PBS -l walltime=64:00:00

use gcc63 boost anaconda3
cd Memoria/code/python-code/
python lstm.py  
