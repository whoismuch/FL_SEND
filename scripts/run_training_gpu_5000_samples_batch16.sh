#!/bin/bash
#SBATCH --job-name=send_training_5000_b16
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal  # Using pascal partition (ampere is currently unavailable)
#SBATCH --output=training_5000_samples_gpu_batch16_%j.out
#SBATCH --error=training_5000_samples_gpu_batch16_%j.err

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

echo "=== Starting Training ==="

# Change to working directory
# Update this path to match your server's directory structure
# Common paths: ~/FL_SEND/28nov/FL_SEND or ~/FL_SEND/24oct/FL_SEND
# If script is run from project root, use current directory
if [ -f "src/SEND_PSE_AMI.py" ]; then
    # Already in project root
    WORK_DIR=$(pwd)
else
    # Try to find project root or use common path
    WORK_DIR="${FL_SEND_WORK_DIR:-$HOME/FL_SEND/17dec/FL_SEND}"
    if [ ! -f "$WORK_DIR/src/SEND_PSE_AMI.py" ]; then
        echo "WARNING: Could not find SEND_PSE_AMI.py at $WORK_DIR"
        echo "Please set FL_SEND_WORK_DIR environment variable or update WORK_DIR in script"
    fi
fi

cd "$WORK_DIR"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/training_5000_samples_gpu_batch16_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo "To view logs in real-time, run in another terminal: tail -f $LOG_FILE"
echo ""

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run training and save all output to log file
# 2>&1 redirects stderr to stdout, so both go to the log file
# Using --test_size 5000 to limit dataset to 5000 samples (faster training for testing)
# PYTHONUNBUFFERED=1 ensures all print/log statements appear immediately in logs
PYTHONUNBUFFERED=1 python src/SEND_PSE_AMI.py \
  --test_size 5000 \
  --epochs 1 \
  --hidden_dim 256 \
  --num_speech_encoder_layers 4 \
  --num_post_net_layers 3 \
  --num_transformer_layers 2 \
  --batch_size 16 \
  > "$LOG_FILE" 2>&1

echo ""
echo "=== Training Complete ==="
echo "Logs saved to: $LOG_FILE"

