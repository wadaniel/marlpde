#!/bin/bash
ic="sinus"
run=1288
version=0
N=128
numAgents=8
nu=0.1
noise=0.5
episodelength=500
exp=1000000

dir=/scratch/wadaniel/diffusion_simple/run${run}/
mkdir ${dir} -p

launchname="${0##*/}"
cp $launchname "${dir}/diffusion_simple_launcher_${run}.sh"
git diff > "${dir}/diffusion_simple_gitdiff_${run}.txt"

cd ..
pushd .

cp run-vracer-diffusion-simple.py ${dir}
cp -r _model/ ${dir}

cd ${dir}

export OMP_NUM_THREADS=8
python3 run-vracer-diffusion-simple.py --ic ${ic} --version ${version} \
    --N ${N} --numAgents ${numAgents} \
    --nu ${nu} --noise ${noise} \
    --episodelength ${episodelength} \
    --exp ${exp} --run ${run}

python3 run-vracer-diffusion-simple.py --ic ${ic} --version ${version} \
    --N ${N} --numAgents ${numAgents} \
    --nu ${nu} --noise ${noise} \
    --episodelength ${episodelength} \
    --exp ${exp} --run ${run} \
    --test

popd

python3 -m korali.rlview --dir "${dir}/_result_diffusion_simple_${run}" --out "vracer_diffusion_simple_${run}.png" \
    --showCI 0.8 --showObservations
