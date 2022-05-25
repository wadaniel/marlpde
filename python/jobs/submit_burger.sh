# Test

export NUMEXP=2000000
export NN=512
export IEX=0.1
export NOISE=0.0
export NUMACT=32
export IC='turbulence'
export N=64

s=1

for env in burger
do 
    for run in {20..20} 
    #for run in {11..15} #0.01
    #for run in {16..20} #0.001
    do
        export ENV=$env
        export RUN=$run
        export SEED=$s

        ./sbatch-burger.sh

        s=$(( $s + 1 ))
    done
done
