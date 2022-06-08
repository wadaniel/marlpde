#!/bin/bash

version=1
IC='box'
run=1
NEX=5000000
N=32
NA=32
dt=0.01
tend=10.0
noise=0.0
nu=0.1
iex=0.01
seed=42
esteps=500

launchname="${0##*/}"
cp $launchname "./diffusion_launcher${run}.sh"

git diff > "./diffusion_gitdiff${run}.txt"

pushd .
cd ..

python3 run-vracer-diffusion.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --tend $tend --nu $nu \
    --iex $iex --seed $seed \
    --episodelength $esteps --version $version --tnoise

python3 run-vracer-diffusion.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --tend $tend --nu $nu \
    --iex $iex --seed $seed \
    --episodelength $esteps --version $version --tnoise \
    --test

python3 -m korali.rlview --dir "_result_diffusion_${run}" --out "vracer_diffusion_${run}.png"


popd
