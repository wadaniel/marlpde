#!/bin/bash
run=3
N=32
numAgents=32
noise=0.1
episodelength=100
iex=0.5
exp=1000000
IC="cos"
force="sin"

RUNPATH=${SCRATCH}/laplace/${run}/
mkdir -p ${RUNPATH}

cd ..
pushd .

cp run-vracer-laplace.py ${RUNPATH}
cp -r _model/ ${RUNPATH}

cd ${RUNPATH}

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=laplace_simple
#SBATCH --output=laplace_${run}_%j.out
#SBATCH --error=laplace_${run}_err_%j.out
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
python run-vracer-laplace.py --N ${N} --episodelen ${episodelength} \
    --numAgents ${numAgents} --noise ${noise} --exp ${exp} --run ${run} --iex ${iex} \
    --ic ${IC} --force ${force}

python run-vracer-laplace.py --N ${N} --episodelen ${episodelength} \
    --numAgents ${numAgents} --noise ${noise} --exp ${exp} --run ${run} --iex ${iex} \
    --ic ${IC} --force ${force} \
    --test

python3 -m korali.rlview --dir "${dir}/_result_laplace_${run}" --out "vracer_laplace_${run}.png" \
    --showCI 0.8 --showObservations

popd
EOF

module purge
module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

chmod 755 run.sbatch
sbatch run.sbatch
