#!/bin/bash
ic="sinus"
run=32
version=0
N=32
numAgents=32
nu=0.1
noise=0.5
episodelength=500
exp=1000000

dir=/scratch/wadaniel/diffusion_direct/run${run}/
mkdir ${dir} -p

launchname="${0##*/}"
cp $launchname "${dir}/diffusion_direct_launcher_${run}.sh"
git diff > "${dir}/diffusion_direct_gitdiff_${run}.txt"

cd ..
pushd .

cp run-vracer-diffusion-direct.py ${dir}
cp -r _model/ ${dir}

cd ${dir}

export OMP_NUM_THREADS=8
python3 run-vracer-diffusion-direct.py --ic ${ic} --version ${version} \
    --N ${N} --numAgents ${numAgents} \
    --nu ${nu} --noise ${noise} \
    --episodelength ${episodelength} \
    --exp ${exp} --run ${run}

python3 run-vracer-diffusion-direct.py --ic ${ic} --version ${version} \
    --N ${N} --numAgents ${numAgents} \
    --nu ${nu} --noise ${noise} \
    --episodelength ${episodelength} \
    --exp ${exp} --run ${run} \
    --test

popd

python3 -m korali.rlview --dir "${dir}/_result_diffusion_direct_${run}" --out "vracer_diffusion_direct_${run}.png" \
    --showCI 0.8 --showObservations
