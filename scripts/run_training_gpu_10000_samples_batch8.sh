#!/bin/bash
#SBATCH --job-name=send_training_10k_b8
#SBATCH --time=10-00:00:00  # 10 days (should be enough for 10000 samples)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G  # Maximum available memory (will auto-limit sequence length to fit)
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal  # Using pascal partition (infinite timelimit, 6 idle nodes available)
#SBATCH --output=training_10000_samples_gpu_batch8_%j.out
#SBATCH --error=training_10000_samples_gpu_batch8_%j.err

# Enable debugging and ensure output is not buffered
set -x
# Disable output buffering
export PYTHONUNBUFFERED=1

# Debug: Output to .out file immediately (stdout goes to .out file)
echo "=== SLURM Script Started ==="
echo "Script: $0"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not set}"
echo ""

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

# Print environment info (output to stdout for .out file)
echo "=== Job Environment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM"
echo "GPU: $SLURM_GPUS_ON_NODE"
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Check GPU
echo "=== GPU Check ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
nvidia-smi 2>&1 || echo "nvidia-smi not available"

echo "=== Starting Training on 10000 SAMPLES with 100 EPOCHS ==="

# Change to working directory
# Update this path to match your server's directory structure
# Common paths: ~/FL_SEND/28nov/FL_SEND or ~/FL_SEND/24oct/FL_SEND
# If script is run from project root, use current directory
if [ -f "src/SEND_PSE_AMI.py" ]; then
    # Already in project root
    WORK_DIR=$(pwd)
else
    # Try to find project root or use common path
    WORK_DIR="${FL_SEND_WORK_DIR:-$HOME/FL_SEND/17dec_2/FL_SEND}"
    if [ ! -f "$WORK_DIR/src/SEND_PSE_AMI.py" ]; then
        echo "WARNING: Could not find SEND_PSE_AMI.py at $WORK_DIR"
        echo "Please set FL_SEND_WORK_DIR environment variable or update WORK_DIR in script"
    fi
fi

cd "$WORK_DIR" || {
    echo "ERROR: Failed to change to directory $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

# Verify we're in the right place
if [ ! -f "src/SEND_PSE_AMI.py" ]; then
    echo "ERROR: SEND_PSE_AMI.py not found in $(pwd)/src/"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/training_10000_samples_gpu_batch8_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo "To view logs in real-time, run in another terminal: tail -f $LOG_FILE"
echo "Starting Python training script..."
echo ""

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run training and save all output to log file
# 2>&1 redirects stderr to stdout, so both go to the log file
# Using --test_size 10000 to limit dataset to 10000 samples
# Memory optimizations:
#   --chunk_size 250: Process 250 samples at a time (reduces peak memory)
#   --batch_size 8: Batch size for training (larger batch for better GPU utilization)
#   --max_memory_gb 64: Auto-calculate max_sequence_length to fit in 64 GB
# Performance note: --compute_der_during_training significantly slows down training (30-50% slower)
#   Consider removing this flag for faster training - DER is still computed on validation set
# PYTHONUNBUFFERED=1 ensures all print/log statements appear immediately in logs
PYTHONUNBUFFERED=1 python src/SEND_PSE_AMI.py \
  --test_size 10000 \
  --epochs 50 \
  --hidden_dim 256 \
  --num_speech_encoder_layers 4 \
  --num_post_net_layers 3 \
  --num_transformer_layers 2 \
  --batch_size 8 \
  --chunk_size 250 \
  --max_memory_gb 64 \
  > "$LOG_FILE" 2>&1

TRAIN_EXIT_CODE=$?
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== Training Complete ==="
else
    echo "=== Training Failed with exit code $TRAIN_EXIT_CODE ==="
fi
echo "Logs saved to: $LOG_FILE"
echo "Script finished at: $(date)"
exit $TRAIN_EXIT_CODE
