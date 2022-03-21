# Test

export NUMEXP=5000000
export NN=512
export IEX=0.1
export NOISE=0.001
export NUMACT=32
export IC='turbulence'
export N=64

s=1

for env in burger
do 
    for run in {16..20} 
    do
        export ENV=$env
        export RUN=$run
        export SEED=$s

        ./sbatch-vracer.sh
        #./sbatch-coupled-vracer.sh

        s=$(( $s + 1 ))
    done
done
