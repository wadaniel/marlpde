#!/bin/bash

export OMP_NUM_THREADS=4

run=11
nagents=2
noise=0.1
iex=0.001
version=1
mar="Cooperation"

IC='sinus'
NEX=1000000
N=32
NA=32
width=256
NDNS=512
dt=0.001
nu=0.02
seed=42
tf=50
nt=20
esteps=500

pushd .
cd ~/projects/korali
bname=`git rev-parse --abbrev-ref HEAD`
echo "[Korali] On branch ${bname}"
if [ $bname != "MARL-new-safe-rl" ]; then
    echo "[Korali] Please install branch MARL-new-safe-rl"
    echo "[Korali] exit.."
    popd
    exit
fi
popd

launchname="${0##*/}"
cp $launchname "./burger_marl_launcher_${run}.sh"

git diff > "./gitdiff_marl_burger_${run}.txt"

pushd .
cd ..

python3 run-vracer-burger-marl.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --nagents $nagents --mar $mar

python3 run-vracer-burger-marl.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --nagents $nagents --mar $mar \
    --test

python3 -m korali.rlview --dir "_result_vracer_marl_${run}" --out "vracer_marl_${run}.png"


popd
