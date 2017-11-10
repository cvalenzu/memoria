#!/bin/bash
#PBS -N arima-train
#PBS -o train_arima_$PBS_JOBID.out
#PBS -e train_arima_$PBS_JOBID.err
#PBS -m bea
#PBS -M camilo.valenzuela@alumnos.usm.cl
#PBS -l walltime=64:00:00

#FILES
#Memoria/data/processed/x_potency_canela1_merged.csv
#Memoria/data/processed/x_potency_monte_redondo_merged.csv
#Memoria/data/processed/x_potency_totoral_merged.csv

use gcc63 boost
files=(Memoria/data/processed/x_potency_canela1_merged.csv Memoria/data/processed/x_potency_monte_redondo_merged.csv Memoria/data/processed/x_potency_totoral_merged.csv)

max_params=5
max_diff=1

for file in ${files[@]}; do
	for D in $(seq 0 0); do
  		for P in $(seq 0 $max_params); do
    			for Q in $(seq 0 $max_params); do
				for d in $(seq 1 $max_diff); do
					for p in $(seq 0 $max_params); do
						for q in $(seq 0 $max_params); do
								Rscript Memoria/code/r-code/arima.R $file $p $d $q $P $D $Q &
						done
					done
				done
				wait
   			done
  		done
	done
done
