# Test


for env in burger diffusion
do 
    for n in 1 8 32
    do
        export ENV=$env
        export IC='box'
        export N=32
        export NUMACT=$n
        export RUN=0

        ./sbatch-vracer.sh
    done

    for n in 1 8
    do
        export ENV=$env
        export IC='box'
        export N=8
        export NUMACT=$n
        export RUN=0
        
        ./sbatch-vracer.sh
    done

done


for env in advection
do 
    for n in 1 8 32
    do
        export ENV=$env
        export IC='sinus'
        export N=32
        export NUMACT=$n
        export RUN=0
        
        ./sbatch-vracer.sh
    done

    for n in 1 8
    do
        export ENV=$env
        export IC='sinus'
        export N=8
        export NUMACT=$n
        export RUN=0
        
        ./sbatch-vracer.sh
    done

done
