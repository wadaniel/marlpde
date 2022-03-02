#!/bin/bash -l

# Read Settings file
# source settings.sh

echo "Environment:"         $ENV
echo "IC:"                  $IC
echo "N:"                   $N
echo "NUMACT:"              $NUMACT
echo "RUN:"					$RUN

RUNPATH=${SCRATCH}/marlpde/$ENV/$IC/$N/$NUMACT/$RUN/
mkdir -p $RUNPATH

cd ..

cp run-vracer-${ENV}.py $RUNPATH
cp -r _model/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=pde_${ENV}
#SBATCH --output=pde_${ENV}_${N}_${NUMACT}_${RUN}_%j.out
#SBATCH --error=pde_${ENV}_${N}_${NUMACT}_${RUN}_err_%j.out
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
python3 run-vracer-${ENV}.py --N "$N" --numactions "$NUMACT" --numexp 10000 --ic $IC --run $RUN

resdir=\$(ls -d _result*)
python3 -m korali.rlview --dir \$resdir --out vracer.png
EOF

chmod 755 run.sbatch
sbatch run.sbatch
