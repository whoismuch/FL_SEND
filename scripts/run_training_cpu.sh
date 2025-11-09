#!/bin/bash
#SBATCH --job-name=send_training_cpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=training_cpu_%j.out
#SBATCH --error=training_cpu_%j.err
# Remove partition specification to use any available CPU nodes

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
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=== Starting Training on CPU ==="

# Change to working directory
cd ~/FL_SEND/24oct/FL_SEND

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run training
python src/SEND_PSE_AMI.py --test_size 75 --epochs 5 --compute_der_during_training

echo "=== Training Complete ==="

