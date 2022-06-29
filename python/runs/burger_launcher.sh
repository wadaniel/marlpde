#!/bin/bash

IC='sinus'
run=10
NEX=1000000
N=32
NA=32
NDNS=1024
dt=0.001
noise=0.1
nu=0.02
iex=0.1
seed=42
tf=50
nt=25
esteps=500
version=0
width=256

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
cp $launchname "./burger_launcher_${run}.sh"

git diff > "./gitdiff_${run}.txt"

pushd .
cd ..

python3 run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width

python3 run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --test

python3 -m korali.rlview --dir "_result_${run}" --out "vracer${run}.png"


popd
