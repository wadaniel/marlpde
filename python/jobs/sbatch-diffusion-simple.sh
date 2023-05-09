#!/bin/bash
ic="sinus"
run=0
version=0
N=128
numAgents=1
nu=0.1
noise=0.5
episodelength=500
exp=1000000

RUNPATH=${SCRATCH}/diffusion_simple/${run}/
mkdir -p ${RUNPATH}

cd ..
pushd .

cp run-vracer-diffusion-simple.py ${RUNPATH}
cp -r _model/ ${RUNPATH}

cd ${RUNPATH}

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=diffusion_simple
#SBATCH --output=diffusion_simple_${run}_%j.out
#SBATCH --error=diffusion_simple_${run}_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s1160                                                         
#SBATCH --mail-user=wadaniel@ethz.ch                                            
#SBATCH --mail-type=END                                                         
#SBATCH --mail-type=FAIL 

export OMP_NUM_THREADS=12
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

python3 -m korali.rlview --dir "${RUNPATH}/_result_diffusion_simple_${run}" --out "vracer_diffusion_simple_${run}.png" \
    --showCI 0.8 --showObservations

popd
EOF

#module purge
#module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

#chmod 755 run.sbatch
#sbatch run.sbatch
