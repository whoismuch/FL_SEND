#!/bin/bash
#SBATCH --job-name=send_training_full
#SBATCH --time=2-00:00:00  # 2 days (max for ampere partition) - may need multiple runs for 100 epochs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G  # More memory for full dataset
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal  # Using pascal partition (infinite timelimit, 6 idle nodes available)
#SBATCH --exclude=pascal-node01.l3s.intra,pascal-node03.l3s.intra  # Exclude unavailable nodes
#SBATCH --output=training_full_%j.out
#SBATCH --error=training_full_%j.err

# Initialize conda - try multiple common locations
if [ -f ~/.conda/etc/profile.d/conda.sh ]; then
    source ~/.conda/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
else
    # Try to find conda.sh
    CONDA_SH=$(find ~ -name "conda.sh" 2>/dev/null | head -1)
    if [ -n "$CONDA_SH" ]; then
        source "$CONDA_SH"
    else
        # Last resort: source bashrc
        source ~/.bashrc
    fi
fi

# Initialize conda shell hook (needed for conda activate to work)
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Find conda environment path directly (more reliable than conda activate)
CONDA_ENV_PATH=""
for possible_base in ~/.conda ~/anaconda3 ~/miniconda3; do
    if [ -d "$possible_base/envs/flsend_clean" ] && [ -f "$possible_base/envs/flsend_clean/bin/python" ]; then
        CONDA_ENV_PATH="$possible_base/envs/flsend_clean"
        break
    fi
done

# If not found, try to find it
if [ -z "$CONDA_ENV_PATH" ]; then
    CONDA_ENV_PATH=$(find ~ -type d -path "*/envs/flsend_clean/bin" -exec dirname {} \; 2>/dev/null | head -1)
fi

# Use direct path to Python if found
if [ -n "$CONDA_ENV_PATH" ] && [ -f "$CONDA_ENV_PATH/bin/python" ]; then
    echo "Found conda environment at: $CONDA_ENV_PATH"
    export PATH="$CONDA_ENV_PATH/bin:$PATH"
    export CONDA_DEFAULT_ENV=flsend_clean
elif conda env list 2>/dev/null | grep -q "flsend_clean"; then
    echo "Activating existing conda environment: flsend_clean"
    conda activate flsend_clean
else
    echo "ERROR: Environment flsend_clean not found on GPU node!"
    echo "Please ensure it exists. You may need to create it on login node first."
    exit 1
fi

# Verify environment is activated
echo "Active conda environment: ${CONDA_DEFAULT_ENV:-not set}"
echo "Python path: $(which python || echo 'NOT FOUND - WILL FAIL')"

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

echo "=== Starting Training on FULL DATASET with 100 EPOCHS ==="

# Change to working directory
cd ~/FL_SEND/6nov/FL_SEND

# Run training on full dataset with 100 epochs
# Note: --test_size not specified means using ALL available data
# Memory optimizations (if your SEND_PSE_AMI.py supports these arguments):
#   --chunk_size 250: Process 250 samples at a time (reduces peak memory)
#   --batch_size 2: Smaller batch size (reduces memory during training)
# If arguments are not recognized, remove --chunk_size and --batch_size lines below
python SEND_PSE_AMI.py --epochs 100 --compute_der_during_training --chunk_size 250 --batch_size 2

echo "=== Training Complete ==="

