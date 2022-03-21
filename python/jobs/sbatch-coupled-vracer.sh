#!/bin/bash -l

echo "IEX:"                 $IEX
echo "N:"                   $N
echo "NN:"                  $NN
echo "NUMACT:"              $NUMACT
echo "NEXP:"                $NUMEXP

echo "Environment:"         $ENV
echo "IC:"                  $IC
echo "NOISE:"               $NOISE
echo "SEED:"                $SEED
echo "RUN:"					$RUN

RUNPATH=${SCRATCH}/marlpde_coupled/$ENV/$RUN/$IC
mkdir -p $RUNPATH

cd ..

cp run-vracer-coupled-${ENV}.py $RUNPATH
cp -r _model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=pde_${ENV}
#SBATCH --output=pde_${ENV}_${NUMACT}_${SEED}_${RUN}_%j.out
#SBATCH --error=pde_${ENV}_${NUMACT}_${SEED}_${RUN}_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
python3 run-vracer-coupled-${ENV}.py --N $N --numactions $NUMACT --numexp $NUMEXP --width $NN --ic $IC --iex $IEX --noise $NOISE --seed $SEED --run $RUN

resdir=\$(ls -d _result*)
python3 -m korali.rlview --dir \$resdir --out vracer.png

python3 run-vracer-coupled-${ENV}.py --N $N --numactions $NUMACT --numexp $NUMEXP --width $NN --ic $IC --iex $IEX --noise $NOISE --seed $SEED --run $RUN --test

EOF

chmod 755 run.sbatch
sbatch run.sbatch
