# Test

# run 5: noisy, 0.1 exp noise
# run 6: noisy, 0.01 exp noise
export RUN=5
export NUMEXP=1000000
export NN=256


for env in burger #diffusion
do 
    for a in 32
    do
        export ENV=$env
        export IC='turbulence'
        export N=64
        export NUMACT=$a

        ./sbatch-vracer.sh
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
