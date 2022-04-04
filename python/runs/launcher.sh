#!/bin/bash

IC='turbulence'
run='18'
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

cp launcher.sh "./launcher${run}.sh"

git diff > "./gitdiff${run}.txt"

pushd .
cd ..

python run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --specreward

python run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --specreward \
    --test

python -m korali.rlview --dir "_result_${IC}_${run}" --out "vracer${run}.png"

popd
