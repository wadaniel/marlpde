#!/bin/bash
export OMP_NUM_THREADS=8

run=1
nagents=4
noise=0.001
iex=0.1
version=0
mar="Cooperation"
#mar="Individual"

#IC='turbulence'
IC='sinus'
NEX=10000000
N=32
NA=32
width=64
NDNS=1024
dt=0.001
nu=0.02
seed=42
tf=50
nt=25
esteps=500

pushd .
cd ~/projects/korali
bname=`git rev-parse --abbrev-ref HEAD`
echo "[Korali] On branch ${bname}"
if [ $bname != "MARL-new-safe-rl" ]; then
    echo "[Korali] Please install branch MARL-new-safe-rl"
    echo "[Korali] exit.."
    popd
    exit
fi
popd

launchname="${0##*/}"
cp $launchname "./burger_marl_launcher_${run}.sh"

git diff > "./gitdiff_marl_burger_${run}.txt"

pushd .
cd ..

python3 run-vracer-burger-marl.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width --dforce \
    --nagents $nagents --mar $mar

python3 run-vracer-burger-marl.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width --dforce \
    --nagents $nagents --mar $mar \
    --test

python3 -m korali.rlview --dir "_result_vracer_marl_${run}" --out "vracer_marl_${run}.png"


popd
