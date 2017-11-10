#!/bin/bash
#PBS -N arima-train
#PBS -o train_arima_$PBS_JOBID.out
#PBS -e train_arima_$PBS_JOBID.err
#PBS -l walltime=200:00:00

#FILES
#Memoria/data/processed/x_potency_canela1_merged.csv
#Memoria/data/processed/x_potency_monte_redondo_merged.csv
#Memoria/data/processed/x_potency_totoral_merged.csv

use gcc63 boost
files=(Memoria/data/processed/x_potency_canela1_merged.csv Memoria/data/processed/x_potency_monte_redondo_merged.csv Memoria/data/processed/x_potency_totoral_merged.csv)

max_params=5
max_diff=1
d=1

for file in ${files[@]}; do
	for D in $(seq 0 $max_diff); do
  		for P in $(seq 0 $max_params); do
    			for Q in $(seq 0 $max_params); do
#						for d in $(seq 1 $max_diff); do
							for p in $(seq 0 $max_params); do
								for q in $(seq 0 $max_params); do
									if [ $(($p + $q)) -le 4 ]; then
										if [ $(( $P + $Q)) -le 4  ]; then
										Rscript Memoria/code/r-code/arima.R $file $p $d $q $P $D $Q &
										fi
									fi
								done
#							done
						done
						wait
   				done
  		done
	done
done
