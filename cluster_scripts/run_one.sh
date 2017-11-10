#!/bin/bash
#PBS -N esn_training2
#PBS -o script_out_%j.log
#PBS -e script_err_%j.log
#PBS -l walltime=100:00:00
echo "Loading Anaconda"
use gcc63 boost anaconda3
echo "Running Script"
cd Memoria/code/python-code
python one_out.py
