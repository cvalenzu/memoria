#!/bin/bash
#PBS -N lstm-train
#PBS -o lstm_arima.out
#PBS -e lstm_arima.err
#PBS -m bea
#PBS -M camilo.valenzuela@alumnos.usm.cl
#PBS -l walltime=64:00:00
#PBS -q gpuk

use anaconda3
cd Memoria/code/python-code/
python lstm.py  
