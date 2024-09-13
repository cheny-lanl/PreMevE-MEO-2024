#!/bin/bash -l

#SBATCH --job-name=val
#SBATCH --output=%x.log
#SBATCH --error=%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=volta-x86
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --nodelist=cn660,cn661,cn662,cn663,cn664,cn665,cn666,cn667
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --mail-user=cheny@lanl.gov
#SBATCH --mail-type=ALL

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $SLURM_JOB_NAME"
echo "Job ID : $SLURM_JOB_ID"
echo "=========================================================="
cat /etc/redhat-release

MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
MASTERPORT=6000

echo "Master Node: $MASTER"
echo "Slave Node(s): $SLAVES"

module purge
module load cuda/11.3.1
module load miniconda3/py39_4.10.3
source activate premev2024

export OMP_NUM_THREADS=1
cd ~/PreMevE_MEO/Space_code/src

# Test command is add --resume_auto --only_test to the corresponding training command.
srun python -u main.py --output_dir ~/PreMevE_MEO/Space_code/ckpt \
 --dist_url tcp://$MASTER:$MASTERPORT --world_size $SLURM_NTASKS \
 --anno-path ~/PreMevE_MEO/Space_code/dataset/2023_05_25 \
 --model FluexNet_meta \
 --ckpt_name best --no-save_window --resume_auto --only_test
