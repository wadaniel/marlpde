#!/bin/bash

run=2
N=32
numAgents=32
noise=0.1
episodelength=100
iex=0.5
exp=1000000
IC="one"
force="sincos"

dir=/scratch/wadaniel/laplace/run${run}/
mkdir ${dir} -p

launchname="${0##*/}"
cp $launchname "${dir}/laplace_launcher_${run}.sh"
git diff > "${dir}/laplace_gitdiff_${run}.txt"

cd ..
pushd .

cp run-vracer-laplace.py ${dir}
cp -r _model/ ${dir}

cd ${dir}

export OMP_NUM_THREADS=4
python run-vracer-laplace.py --N ${N} --episodelen ${episodelength} \
    --numAgents ${numAgents} --noise ${noise} --exp ${exp} --run ${run} --iex ${iex} \
    --ic ${IC} --force ${force}

python run-vracer-laplace.py --N ${N} --episodelen ${episodelength} \
    --numAgents ${numAgents} --noise ${noise} --exp ${exp} --run ${run} --iex ${iex} \
    --ic ${IC} --force ${force} \
    --test

popd

python3 -m korali.rlview --dir "${dir}/_result_laplace_${run}" --out "vracer_laplace_${run}.png" \
    --showCI 0.8 --showObservations
