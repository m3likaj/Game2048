#!/bin/bash
#
# -= Resources =-
#
#SBATCH --job-name=2048_Train        # Job name
#SBATCH --account=users             # Account name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks per node
#SBATCH --partition=mid           # Partition/queue name
#SBATCH --gres=gpu:1                # Number of GPUs required
#SBATCH --time=1200                  # Maximum runtime (in minutes)
#SBATCH --output=%j-2048.out        # Output file name (%j = job ID)

#SBATCH --mail-type=ALL           
#SBATCH --mail-user=m3likaj@gmail.com

INPUT_FILE="main.py"                # Your Python script

######### DON'T DELETE BELOW THIS LINE ########################################
source /etc/profile.d/zzz_cta.sh
echo "source /etc/profile.d/zzz_cta.sh"
######### DON'T DELETE ABOVE THIS LINE ########################################

# MODULES LOAD...
echo "Load Python and TensorFlow..."
module load python-3.8.11-gcc-10.2.0-faynju3   # Load Python 3.8
module load py-numpy-1.21.2-gcc-10.2.0-5psmui7 # Load NumPy
module load tensorflow/gpu/2                   # Load TensorFlow (GPU version)
module load cuda-11.1.1-gcc-10.2.0-pbxul2t     # Load CUDA
module load cudnn-8.0.4.30-11.1-gcc-10.2.0-tltdvgm # Load cuDNN

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


echo "Running Python script..."
echo "===================================================================================="
python $INPUT_FILE
RET=$?

echo ""
echo "===================================================================================="
echo "Script exited with return code: $RET"
exit $RET
