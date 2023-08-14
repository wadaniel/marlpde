#!/bin/bash

run=1281
N=128
numAgents=1
noise=0.5
episodelength=500
iex=0.5
exp=1000000

dir=/scratch/wadaniel/advection_simple/run${run}/
mkdir ${dir} -p

launchname="${0##*/}"
cp $launchname "${dir}/advection_simple_launcher_${run}.sh"
git diff > "${dir}/advection_simple_gitdiff_${run}.txt"

cd ..
pushd .

cp run-vracer-advection-simple.py ${dir}
cp -r _model/ ${dir}

cd ${dir}

export OMP_NUM_THREADS=4
python run-vracer-advection-simple.py --N ${N} --episodelen ${episodelength} \
    --numAgents ${numAgents} --noise ${noise} --exp ${exp} --run ${run} --iex ${iex} \

python run-vracer-advection-simple.py --N ${N} --episodelen ${episodelength} \
    --numAgents ${numAgents} --noise ${noise} --exp ${exp} --run ${run} --iex ${iex} \
    --test

popd

python3 -m korali.rlview --dir "${dir}/_result_advection_simple_${run}" --out "vracer_advection_simple_${run}.png" \
    --showCI 0.8 --showObservations
