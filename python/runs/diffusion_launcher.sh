#!/bin/bash

version=1
IC='box'
run=8
NEX=2000000
N=32
NA=32
dt=0.1
tend=10.0
noise=0.01
nu=0.1
iex=0.1
seed=42
esteps=100

launchname="${0##*/}"
cp $launchname "./diffusion_launcher${run}.sh"

git diff > "./gitdiff${run}.txt"

pushd .
cd ..

python run-vracer-diffusion.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --tend $tend --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --version $version

python -m korali.rlview --dir "_result_diffusion_${run}" --out "vracer_diffusion_${run}.png"


popd
