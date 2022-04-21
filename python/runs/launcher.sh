#!/bin/bash

IC='turbulence'
run=43
NEX=3000000
N=32
NA=32
NDNS=512
dt=0.001
noise=0.1
nu=0.02
iex=0.1
seed=42
tf=200
nt=10
esteps=500

launchname="${0##*/}"
cp $launchname "./launcher${run}.sh"

git diff > "./gitdiff${run}.txt"

pushd .
cd ..

python run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt \
    --forcing --nunoise

python run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt \
    --forcing --nunoise
    --test

python -m korali.rlview --dir "_result_${IC}_${run}" --out "vracer${run}.png"


popd
