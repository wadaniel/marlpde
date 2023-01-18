#!/bin/bash

# CONFIGS
export OMP_NUM_THREADS=4

IC='sinus'
# (sth not working imo DW, probably params need to be changed 9.11)
#IC='turbulence'
run=2
#NEX=1000000
NEX=200000
N=32
NA=32
NDNS=512
dt=0.001
noise=0.01
nu=0.02
iex=0.1
seed=42
tf=50
nt=100
esteps=500
version=0
width=64
stepper=20
ndns=20

# check branch
pushd .
cd ~/projects/korali
bname=`git rev-parse --abbrev-ref HEAD`
echo "[Korali] On branch ${bname}"
if [ $bname != "safe-rl" ]; then
    echo "[Korali] Please install branch safe-rl"
    echo "[Korali] exit.."
    popd
    exit
fi
popd

launchname="${0##*/}"

# store scripts
cp $launchname "./burger_launcher_${run}.sh"
git diff > "./gitdiff_${run}.txt"

# backup
resdir="./../_result_${run}"
mkdir $resdir
cp $launchname "${resdir}/burger_launcher_${run}.sh"
git diff > "${resdir}/gitdiff_${run}.txt"

pushd .
cd ..

# launch
python3 run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu --stepper $stepper \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width  \
    --dforce \ # --specreward
     --ndns $ndns

python3 run-vracer-burger.py --ic $IC --run $run --NE $NEX \
    --N $N --NA $NA --dt $dt --nu $nu --stepper $stepper \
    --iex $iex --noise $noise --seed $seed \
    --episodelength $esteps --NDNS $NDNS \
    --tf $tf --nt $nt --version $version --width $width \
    --dforce \ # --specreward
     --ndns $ndns \
    --test

python3 -m korali.rlview --dir "_result_${run}" --out "vracer${run}.png"

popd
