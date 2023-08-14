#!/bin/bash
export OMP_NUM_THREADS=4

run=102
IC='forced'
L=100
T=2
NEX=1000000
N=64
NA=64
NDNS=1024
dt=0.01
stepper=1
noise=0.001
nu=0.02
iex=0.1
seed=42
tf=50
nt=100
esteps=100
version=0
width=64

pushd .
cd ~/projects/korali
bname=`git rev-parse --abbrev-ref HEAD`
echo "[Korali] On branch ${bname}"
if [ $bname != "safe-rl" ]; then
    echo "[Korali] Please install branch safe-rl"
    echo "[Korali] exit.."
    popd
    exit
fi
popd

launchname="${0##*/}"
cp $launchname "./burger_force_launcher_${run}.sh"

git diff > "./gitdiff_force_burger_${run}.txt"

pushd .
cd ..

python3 run-vracer-burger.py --ic $IC --run $run --L $L --T $T --NE $NEX \
    --N $N --NA $NA --dt $dt --stepper $stepper --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width  \
    --specreward --forcing

python3 run-vracer-burger.py --ic $IC --run $run --L $L --T $T --NE $NEX \
    --N $N --NA $NA --dt $dt --stepper $stepper --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --specreward --forcing \
    --test

python3 -m korali.rlview --dir "_result_${run}" --out "./plots/vracer${run}.png"


popd
