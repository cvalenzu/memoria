#!/bin/bash
#PBS -N arima-train
#PBS -o train_arima.out
#PBS -e train_arima.err
#PBS -m bea
#PBS -M camilo.valenzuela@alumnos.usm.cl
#PBS -l walltime=64:00:00

Rscript Memoria/code/r-code/arima.R
