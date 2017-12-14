#!/bin/bash
#PBS -N arima-train
#PBS -o train_arima_$PBS_JOBID.out
#PBS -e train_arima_$PBS_JOBID.err
#PBS -l walltime=200:00:00

use gcc63 boost
files=(Memoria/data/windowed/X_canela1.csv Memoria/data/windowed/X_totoral.csv Memoria/data/windowed/X_monte.csv)

max_params=5
max_diff=1
d=1


mkdir -p results/arima
cd results/arima


rm args.dat 2>/dev/null
for file in ${files[@]}; do
	for D in $(seq 0 $max_diff); do
  		for P in $(seq 0 $max_params); do
    			for Q in $(seq 0 $max_params); do
#						for d in $(seq 1 $max_diff); do
							for p in $(seq 0 $max_params); do
								for q in $(seq 0 $max_params); do
									if [ $(($p + $q)) -le 4 ]; then
										if [ $(( $P + $Q)) -le 4  ]; then
											echo "../../Memoria/code/r-code/arima.R ../../$file $p $d $q $P $D $Q" >> args.dat
											#Rscript ../../Memoria/code/r-code/arima.R ../../$file $p $d $q $P $D $Q &
										fi
									fi
								done
#							done
						done
#						wait
   				done
  		done
	done
done

<args.dat xargs -L1 -P20 Rscript

echo "Finish"
