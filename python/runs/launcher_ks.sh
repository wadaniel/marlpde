#!/bin/bash

run=2
NEX=500000
N=32
NA=32
NDNS=2048
dt=0.1
nu=1.
iex=0.0001
seed=42
esteps=500
tf=100
nt=1


launchname="${0##*/}"
cp $launchname "./launcher_ks${run}.sh"

git diff > "./gitdiff${run}.txt"

pushd .
cd ..

python run-vracer-ks.py --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt \
    --dforce

python run-vracer-ks.py --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt \
    --dforce \
    --test

python -m korali.rlview --dir "_result_ks_${run}" --out "vracer_ks${run}.png"


popd
