# Test

export NUMEXP=1000000
export NN=512
export IEX=0.01
export NOISE=1.0
export NUMACT=32
export IC='turbulence'
export N=64


for env in burger
do 
    for run in {0..2}
    do
        export ENV=$env
        export RUN=$run
        export SEED=$run

        ./sbatch-vracer.sh
        ./sbatch-coupled-vracer.sh
    done
done
