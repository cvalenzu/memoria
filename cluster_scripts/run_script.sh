#!/bin/bash
#PBS -N esn_training1
#PBS -o script_out
#PBS -e script_err
#PBS -l walltime=100:00:00
echo "Loading Anaconda"
use gcc63 boost anaconda3
echo "Running Script"
cd Memoria/code/python-code
python multi_out.py
