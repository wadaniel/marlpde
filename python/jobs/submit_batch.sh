# Test


export RUN=2


for env in burger #diffusion
do 
    for a in 8 32
    do
        export ENV=$env
        export IC='box'
        export N=32
        export NUMACT=$a

        ./sbatch-vracer.sh
    done

    for a in 8 16
    do
        export ENV=$env
        export IC='box'
        export N=16
        export NUMACT=$a
        
        ./sbatch-vracer.sh
    done

done

exit

for env in advection
do 
    for a in 1 8 32
    do
        export ENV=$env
        export IC='sinus'
        export N=32
        export NUMACT=$a
        
        ./sbatch-vracer.sh
    done

    for a in 1 8
    do
        export ENV=$env
        export IC='sinus'
        export N=8
        export NUMACT=$a
        
        ./sbatch-vracer.sh
    done

done
