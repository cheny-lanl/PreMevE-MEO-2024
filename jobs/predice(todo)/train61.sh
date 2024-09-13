#!/bin/bash -l

#SBATCH --job-name=test61
#SBATCH --output=%x.log
#SBATCH --error=%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=volta-x86
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --nodelist=cn660,cn661,cn662,cn663,cn664,cn665,cn666,cn667
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --mail-user=ynf@lanl.gov
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
source activate ynf

export OMP_NUM_THREADS=1
cd ~/Space/src

srun python -u main_fineturn_x.py --output_dir ~/result/space \
 --dist_url tcp://$MASTER:$MASTERPORT --world_size $SLURM_NTASKS \
 --anno-path /vast/home/ynf/dataset/space/2023_05_25  \
 --batch_size 32 --epochs 100 \
 --model FluexNet_meta \
 --warmup_epochs 40 \
 --accum_iter 2 \
 --blr 1e-3 --weight_decay 0.05 \
 --embed_dim 256 --latent_dim 256 --input_size 72 \
 --depth 2  \
 --patch_size 6 9 --drop_path 0.1 --no-interp --token -5 --log --norm std --l4w 0.01 --meta_pose --meta_feature --drop_fluxe 1 2 3 4 --geo --input_geo --old_geo_input \
 --predict 24 \
 --ckpt_name test121 \

echo '**************************************************'

srun python -u main_fineturn_x.py --output_dir ~/result/space \
 --dist_url tcp://$MASTER:$MASTERPORT --world_size $SLURM_NTASKS \
 --anno-path /vast/home/ynf/dataset/space/2023_05_25  \
 --batch_size 32 --epochs 100 \
 --model FluexNet_meta \
 --warmup_epochs 40 \
 --accum_iter 2 \
 --blr 1e-3 --weight_decay 0.05 \
 --embed_dim 256 --latent_dim 256 --input_size 72 \
 --depth 2  \
 --patch_size 6 9 --drop_path 0.1 --no-interp --token -5 --log --norm std --l4w 0.01 --meta_pose --meta_feature --drop_fluxe 1 2 3 4 --geo --input_geo --old_geo_input \
 --predict 48 \
 --ckpt_name test122 \


srun python -u main_fineturn_x.py --output_dir ~/result/space \
 --dist_url tcp://$MASTER:$MASTERPORT --world_size $SLURM_NTASKS \
 --anno-path /vast/home/ynf/dataset/space/2023_05_25  \
 --batch_size 32 --epochs 100 \
 --model FluexNet_meta \
 --warmup_epochs 40 \
 --accum_iter 2 \
 --blr 1e-3 --weight_decay 0.05 \
 --embed_dim 256 --latent_dim 256 --input_size 72 \
 --depth 2  \
 --patch_size 6 9 --drop_path 0.1 --no-interp --token -5 --log --norm std --l4w 0.01 --meta_pose --meta_feature --drop_fluxe 1 2 3 4 --geo --input_geo --old_geo_input \
 --predict 24 \
 --ckpt_name test121 --resume_auto --only_test \

echo '**************************************************'

srun python -u main_fineturn_x.py --output_dir ~/result/space \
 --dist_url tcp://$MASTER:$MASTERPORT --world_size $SLURM_NTASKS \
 --anno-path /vast/home/ynf/dataset/space/2023_05_25  \
 --batch_size 32 --epochs 300 \
 --model FluexNet_meta \
 --warmup_epochs 40 \
 --accum_iter 2 \
 --blr 1e-3 --weight_decay 0.05 \
 --embed_dim 256 --latent_dim 256 --input_size 72 \
 --depth 2  \
 --patch_size 6 9 --drop_path 0.1 --no-interp --token -5 --log --norm std --l4w 0.01 --meta_pose --meta_feature --drop_fluxe 1 2 3 4 --geo --input_geo --old_geo_input \
 --predict 48 \
 --ckpt_name test122 --resume_auto --only_test \
