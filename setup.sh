#!/bin/bash -l

#SBATCH --job-name=setup
#SBATCH --output=%x.log
#SBATCH --error=%x.err
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=volta-x86
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --qos=debug
#SBATCH --time=1:00:00
#SBATCH --mail-user=cheny@lanl.gov
#SBATCH --mail-type=ALL

# Purge existing modules
module purge

# Load required modules
module load cuda/11.3.1
module load miniconda3/py39_4.10.3

# Create a new conda environment named premev2024
conda create --name premev2024 python=3.9 -y

# Activate the environment
source activate premev2024

# Install required packages
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
conda install -c conda-forge timm
conda install numpy scikit-learn scipy pandas -y
conda install tensorboard

#conda install timm numpy scikit-learn scipy pandas -y

echo "Environment premev2024 setup completed."
