#! /bin/bash
#
#SBATCH -N 1 -n 20                 # 1 tasks on 1 node
#SBATCH --time=12:00:00           # time limits: 20 min
#SBATCH --error myJob.err         # std-error file
#SBATCH --output myJob.out        # std-output file
#SBATCH --account=OGS_test2528  # account number
#SBATCH --partition=g100_usr_prod # partition to be used
##SBATCH --partition=g100_all_serial # partition to be used


#

cd $SLURM_SUBMIT_DIR

date

mpirun -np 20 python3  MODEL_1D_ENS_FAST.py EW  Rnd
mpirun -np 20 python3  MODEL_1D_ENS_FAST.py KPZ  Rnd
#mpirun -np 20 python prova_gather.py

date
