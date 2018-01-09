mkdir -p results/arfima
cd results/arfima

echo "Canela"
Rscript ../../Memoria/code/r-code/arfima.R ../../Memoria/data/windowed/X_canela1.csv &
echo "Canela Done"

echo "Totoral"
Rscript ../../Memoria/code/r-code/arfima.R ../../Memoria/data/windowed/X_totoral.csv &
echo "Totoral Done"

echo "Monte Redondo"
Rscript ../../Memoria/code/r-code/arfima.R ../../Memoria/data/windowed/X_monte.csv &
echo "Monte Redondo Done"

wait
