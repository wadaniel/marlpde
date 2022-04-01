IC='turbulence'
run='15'
NEX=1000000
N=16
NA=16
NDNS=512
dt=0.001
noise=0.0
nu=0.02
iex=0.1
seed=42
esteps=500

mkdir -p ./runs/

cp jax_launcher.sh "./runs/jax_launcher${run}.sh"

git diff > "./runs/jax_gitdiff${run}.txt"

python run-vracer-burger-jax.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \

python run-vracer-burger-jax.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --test

python -m korali.rlview --dir "_result_jax_${run}" --out "vracer_jax${run}.png"
