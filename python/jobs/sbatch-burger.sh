#!/bin/bash -l

RUN=1
ENV="burger"
IC="sinus"
NEX=1000000
N=32
NA=32
NDNS=512
dt=0.001
noise=0.1
nu=0.02
iex=0.1
seed=42
tf=50
nt=20
esteps=500
version=1
width=256

pushd .

cd ~/projects/korali
bname=`git branch --show-current`
echo $bname
if [ $bname != "safe-rl" ]; then
    echo "Installing safe-rl branch"
    git checkout safe-rl
    pip3 uninstall korali -y; 
    rm -rf ~/projects/korali/build; 
    mkdir -p ~/projects/korali/build; 
    meson .. -Dmpi=true -Donednn=true -Dtest=false -Dbuildtype=release --prefix=~/.local; 
    ninja install;
fi

popd

git diff > "./gitdiff_burger_${RUN}.txt"

RUNPATH=${SCRATCH}/rlpde/$ENV/$RUN/
mkdir -p $RUNPATH

cd ..

cp run-vracer-${ENV}.py $RUNPATH
cp -r _model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=RL_${ENV}
#SBATCH --output=RL_${ENV}_${RUN}_%j.out
#SBATCH --error=RL_${ENV}_${RUN}_err_%j.out
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
    --tf $tf --nt $nt --version $version --width $width

python run-vracer-burger.py --ic $IC --run $RUN --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width
    --test

python -m korali.rlview --dir "_result_${IC}_${RUN}" --out "vracer${RUN}.png"

popd
EOF

module purge
module load daint-gpu gcc GSL/2.7-CrayGNU-21.09 cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60

chmod 755 run.sbatch
sbatch run.sbatch
