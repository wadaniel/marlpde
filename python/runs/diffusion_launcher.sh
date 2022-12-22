#!/bin/bash

IC='sinus'
version=0
noise=0
run=1

dir=/scratch/wadaniel/diffusion/run${run}/

launchname="${0##*/}"
cp $launchname "${dir}/diffusion_launcher_${run}.sh"
git diff > "${run}/diffusion_gitdiff_${run}.txt"

pushd .
cd ..

mkdir ${dir}

cp run-vracer-diffsion.py ${dir}
cp -r _model/ ${dir}

cd ${dir}

python3 run-vracer-diffusion.py --IC ${IC} --version ${version} --noise ${noise} --run ${run}

popd

python3 -m korali.rlview --dir "${dir}/_result_diffusion_${run}" --out "vracer_diffusion_${run}.png"
