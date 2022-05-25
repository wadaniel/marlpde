#!/bin/bash -l

version=1
width=512
IC='box'
RUN=3
NEX=10000000
N=32
NA=32
dt=0.01
tend=10.0
noise=0.1
nu=0.1
iex=0.1
seed=42
esteps=500
ENV="diffusion"

git diff > "./gitdiff${RUN}.txt"

RUNPATH=${SCRATCH}/marlpde/$ENV/$RUN/
mkdir -p $RUNPATH

cd ..

cp run-vracer-${ENV}.py $RUNPATH
cp -r _model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=pde_${ENV}
#SBATCH --output=pde_${ENV}_${RUN}_%j.out
#SBATCH --error=pde_${ENV}_${RUN}_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

python3 run-vracer-diffusion.py --ic $IC --run $RUN --NE $NEX \
    --N $N --NA $NA --dt $dt --tend $tend --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --width $width --version $version --tnoise

python3 -m korali.rlview --dir "_result_diffusion_${RUN}" --out "vracer_diffusion_${RUN}.png"

popd
EOF

module purge
module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

chmod 755 run.sbatch
sbatch run.sbatch
