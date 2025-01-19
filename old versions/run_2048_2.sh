#!/bin/bash
#
# -= Resources =-
#
#SBATCH --job-name=Slurm-Test
#SBATCH --account=users
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=mid
#SBATCH --gres=gpu:1
#SBATCH --time=1200
#SBATCH --output=%j-slurm.out

# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=foo@bar.com

INPUT_FILE="main.py"


######### DON'T DELETE BELOW THIS LINE ########################################
source /etc/profile.d/zzz_cta.sh
echo "source /etc/profile.d/zzz_cta.sh"
######### DON'T DELETE ABOW THIS LINE #########################################

# MODULES LOAD...
echo "Load Anaconda..."
module load anaconda3-5.3.1

echo ""
echo "============================== ENVIRONMENT VARIABLES ==============================="
env
echo "===================================================================================="
echo ""

echo "======================================================================================"
# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo "======================================================================================"


echo "Running Keras command..."
echo "===================================================================================="
python $INPUT_FILE
RET=$?

echo ""
echo "===================================================================================="
echo "Solver exited with return code: $RET"
exit $RET
