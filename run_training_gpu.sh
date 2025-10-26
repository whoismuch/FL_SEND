#!/bin/bash
#SBATCH --job-name=send_training_gpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=ampere  # Using ampere partition with A100 GPUs
#SBATCH --output=~/FL_SEND/24oct/training_gpu_%j.out
#SBATCH --error=~/FL_SEND/24oct/training_gpu_%j.err

# Activate conda environment
source ~/.bashrc
conda activate flsend_clean

# Print environment info
echo "=== Job Environment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM"
echo "GPU: $SLURM_GPUS_ON_NODE"
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# Check GPU
echo "=== GPU Check ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
nvidia-smi || echo "nvidia-smi not available"

echo "=== Starting Training ==="

# Change to working directory
cd ~/FL_SEND/24oct/FL_SEND

# Run training
python SEND_PSE_AMI.py --test_size 75 --epochs 5 --compute_der_during_training

echo "=== Training Complete ==="

