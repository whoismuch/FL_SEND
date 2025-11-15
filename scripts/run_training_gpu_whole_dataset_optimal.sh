#!/bin/bash
#SBATCH --job-name=send_training_optimal
#SBATCH --time=30-00:00:00  # 30 days (max for pascal partition)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G  # Maximum available memory (will auto-limit sequence length to fit)
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal  # Using pascal partition (infinite timelimit, 6 idle nodes available)
#SBATCH --output=training_optimal_%j.out
#SBATCH --error=training_optimal_%j.err

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

echo "=== Starting OPTIMAL Training on FULL DATASET with 100 EPOCHS ==="
echo "=== OPTIMIZATIONS: All performance optimizations enabled ==="

# Change to working directory
cd ~/FL_SEND/28nov/FL_SEND

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run training on full dataset with 100 epochs
# Note: --test_size not specified means using ALL available data
# OPTIMAL Performance settings:
#   --chunk_size 250: Process 250 samples at a time (reduces peak memory)
#   --batch_size 4: Balanced batch size (good GPU utilization, lower OOM risk than 8)
#     If you have OOM errors, reduce to 2. If GPU utilization < 80%, increase to 8
#   --max_memory_gb 64: Auto-calculate max_sequence_length to fit in 64 GB
#   NO --compute_der_during_training: DER computation disabled for 30-50% faster training
#     DER is still computed on validation set after each epoch
# 
# Expected performance improvements:
#   - Mixed Precision Training (AMP): 2-3x speedup
#   - DataLoader optimization (num_workers, pin_memory): 2-4x speedup for data loading
#   - cuDNN benchmark: 5-15% speedup for convolutions
#   - No DER during training: 30-50% speedup
#   - Larger batch size: 2-4x speedup (if memory allows)
# 
# Total expected speedup: 4-7x compared to original implementation
# PYTHONUNBUFFERED=1 ensures all print/log statements appear immediately in logs
PYTHONUNBUFFERED=1 python src/SEND_PSE_AMI.py --epochs 100 --chunk_size 250 --batch_size 4 --max_memory_gb 64

echo "=== Training Complete ==="

