#!/bin/bash

IC='sinus'
run=20
NEX=1000000
N=32
NA=32
NDNS=512
dt=0.001
noise=0.1
nu=0.02
iex=0.1
seed=42
tf=50
nt=20
esteps=500
version=1
width=512

launchname="${0##*/}"
cp $launchname "./burger_launcher_${run}.sh"

git diff > "./gitdiff_${run}.txt"

pushd .
cd ..

python3 run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --nunoise

python3 run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --nunoise \
    --test

python3 -m korali.rlview --dir "_result_${IC}_${run}" --out "vracer${run}.png"


popd
