# Test

for env in burger
do 
    for run in {21..21} 
    do
        export ENV=$env
        export RUN=$run
        export SEED=$s

        ./sbatch-burger.sh

        s=$(( $s + 1 ))
    done
done
