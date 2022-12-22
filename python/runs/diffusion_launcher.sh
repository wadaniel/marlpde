#!/bin/bash
export OMP_THREAD_NUM=4

ic="gaussian"
version=0
noise=0
run=1
exp=10000000

dir=/scratch/wadaniel/diffusion/run${run}/
mkdir ${dir} -p

launchname="${0##*/}"
cp $launchname "${dir}/diffusion_launcher_${run}.sh"
git diff > "${dir}/diffusion_gitdiff_${run}.txt"

cd ..
pushd .

cp run-vracer-diffusion.py ${dir}
cp -r _model/ ${dir}

cd ${dir}

python3 run-vracer-diffusion.py --ic ${ic} --version ${version} --noise ${noise} \
    --exp ${exp} --run ${run} 

popd

python3 -m korali.rlview --dir "${dir}/_result_diffusion_${run}" --out "vracer_diffusion_${run}.png"
