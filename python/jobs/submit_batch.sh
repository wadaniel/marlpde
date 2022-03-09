# Test

export RUN=10
export NUMEXP=1000000
export NN=256


for env in burger diffusion
do 
    for seed in {0..10}
    do
        for a in 32
        do
            export ENV=$env
            export IC='turbulence'
            export N=64
            export NUMACT=$a
            export SEED=$seed

            ./sbatch-vracer.sh
        done
    done
done

exit

for env in advection
do 
    for a in 32
    do
        export ENV=$env
        export IC='sinus'
        export N=64
        export NUMACT=$a
        
        ./sbatch-vracer.sh
    done
done
