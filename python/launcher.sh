IC='sinus'
run='1'
NEX=1000
N=32
NA=32
dt=0.001
noise=0.0
nu=0.02
iex=0.1
seed=42

python run-vracer-burger.py --ic $IC --run $run --NE $NEX --N $N --NA $NA --dt $dt --nu $nu --iex $iex --noise $noise --seed $seed
python run-vracer-burger.py --ic $IC --run $run --NE $NEX --N $N --NA $NA --dt $dt --nu $nu --iex $iex --noise $noise --seed $seed --test
python -m korali.rlview --dir "_result_${IC}_${N}_${NA}_${dt}_${nu}_${noise}_${seed}_${run}" --out "vracer${run}.png"

