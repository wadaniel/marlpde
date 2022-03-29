IC='turbulence'
run='3'
NEX=200000
NDNS=1024
N=32
NA=32
dt=0.0001
noise=0.0
nu=0.02
iex=0.1
seed=42


mkdir -p ./runs/
cp launcher.sh "./runs/launcher${run}.sh"
python run-vracer-burger.py --ic $IC --NDNS $NDNS --run $run --NE $NEX --N $N --NA $NA --dt $dt --nu $nu --iex $iex --noise $noise --seed $seed --dforce
python run-vracer-burger.py --ic $IC --NDNS $NDNS --run $run --NE $NEX --N $N --NA $NA --dt $dt --nu $nu --iex $iex --noise $noise --seed $seed --dforce --test
python -m korali.rlview --dir "_result_${IC}_${N}_${NA}_${dt}_${nu}_${noise}_${seed}_${run}" --out "vracer${run}.png"

