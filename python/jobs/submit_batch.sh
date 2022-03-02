# Test
for env in advection
#for env in advection burger diffusion
do 
    export N=32
    export IC='sinus'
    export RUN=0
    
    for n in 1
    #for n in 1 8 32
    do
        export ENV=$env
        export NUMACT=$n
        ./sbatch-vracer.sh
    done
done
