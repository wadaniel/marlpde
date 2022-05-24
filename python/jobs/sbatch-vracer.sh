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

RUNPATH=${SCRATCH}/marlpde/$ENV/$RUN/
mkdir -p $RUNPATH

cd ..

cp run-vracer-${ENV}.py $RUNPATH
cp -r _model/ $RUNPATH
cp -r runs/ $RUNPATH

cd $RUNPATH

cat > run.sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=pde_${ENV}
#SBATCH --output=pde_${ENV}_${RUN}_%j.out
#SBATCH --error=pde_${ENV}_${RUN}_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s929

cd runs/
bash burger_launcher.sh

EOF

chmod 755 run.sbatch
sbatch run.sbatch
