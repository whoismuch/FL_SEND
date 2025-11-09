#!/bin/bash
#SBATCH --job-name=send_training_any
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=~/FL_SEND/24oct/training_any_%j.out
#SBATCH --error=~/FL_SEND/24oct/training_any_%j.err

# Try to get ANY available resources
#SBATCH --constraint=ANY  # Remove GPU requirement, run on any CPU

source ~/.bashrc
conda activate flsend_clean

# Change to working directory
cd ~/FL_SEND/24oct/FL_SEND

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=== Job started ==="
echo "Node: $SLURM_NODELIST"
python src/SEND_PSE_AMI.py --test_size 75 --epochs 5 --compute_der_during_training
