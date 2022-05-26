#!/bin/bash -l

RUN=7
ENV="burger"
IC="turbulence"
NEX=5000000
N=32
NA=32
NDNS=512
dt=0.001
noise=1.
nu=0.02
iex=0.1
seed=42
tf=50
nt=20
esteps=500
version=0
width=256

git diff > "./gitdiff_burger_${RUN}.txt"

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

python run-vracer-burger.py --ic $IC --run $RUN --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --nunoise --specreward

python run-vracer-burger.py --ic $IC --run $RUN --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --nunoise --specreward \
    --test

python -m korali.rlview --dir "_result_${IC}_${RUN}" --out "vracer${RUN}.png"


popd
EOF

module purge
module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

chmod 755 run.sbatch
sbatch run.sbatch
