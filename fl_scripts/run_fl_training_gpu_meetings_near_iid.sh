#!/bin/bash
#SBATCH --job-name=fl_send_gpu_meetings_near_iid
#SBATCH --time=10-00:00:00  # 10 days
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal
#SBATCH --exclude=pascal-node03.l3s.intra
#SBATCH --output=fl_training_gpu_meetings_near_iid_%j.out
#SBATCH --error=fl_training_gpu_meetings_near_iid_%j.err

# Enable debugging and ensure output is not buffered
set -x
# Disable output buffering
export PYTHONUNBUFFERED=1

# Ray memory configuration
export RAY_memory_usage_threshold=0.90
export RAY_memory_monitor_refresh_ms=1000
export RAY_object_store_memory=10000000000
export RAY_spill_objects_to_disk=1

# Debug: Output to .out file immediately
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
    CONDA_SH=$(find ~ -name "conda.sh" 2>/dev/null | head -1)
    if [ -n "$CONDA_SH" ]; then
        source "$CONDA_SH"
    else
        source ~/.bashrc
    fi
fi

# Initialize conda shell hook
eval "$(conda shell.bash hook)" 2>/dev/null || true

# Find conda environment path
CONDA_ENV_PATH=""
for possible_base in ~/.conda ~/anaconda3 ~/miniconda3; do
    if [ -d "$possible_base/envs/flsend_clean" ] && [ -f "$possible_base/envs/flsend_clean/bin/python" ]; then
        CONDA_ENV_PATH="$possible_base/envs/flsend_clean"
        break
    fi
done

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
    echo "ERROR: Environment flsend_clean not found!"
    exit 1
fi

# Verify environment
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
echo "Python version: $(python --version 2>&1)"

# Check GPU
echo "=== GPU Check ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
nvidia-smi 2>&1 || echo "nvidia-smi not available"

echo "=== Starting Federated Learning Training with Meeting-based Subsampling (Near-IID) ==="
echo "Configuration:"
echo "  - Device: GPU"
echo "  - Training meetings: 40"
echo "  - Validation meetings: 10"
echo "  - Test meetings: 10"
echo "  - Number of clients: 2"
echo "  - Clients per round: 2 (parallel)"
echo "  - Epochs per round: 1"
echo "  - Number of rounds: 40"
echo "  - Seed: 42"
echo "  - Distribution: Stratified (near-IID by domain)"
echo ""

# Change to working directory
if [ -f "src/FL_SEND_PSE_AMI.py" ]; then
    WORK_DIR=$(pwd)
else
    WORK_DIR="${FL_SEND_WORK_DIR:-$HOME/FL_SEND/17dec_2/FL_SEND}"
    if [ ! -f "$WORK_DIR/src/FL_SEND_PSE_AMI.py" ]; then
        echo "WARNING: Could not find FL_SEND_PSE_AMI.py at $WORK_DIR"
        echo "Please set FL_SEND_WORK_DIR environment variable or update WORK_DIR in script"
    fi
fi

cd "$WORK_DIR" || {
    echo "ERROR: Failed to change to directory $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

# Verify we're in the right place
if [ ! -f "src/FL_SEND_PSE_AMI.py" ]; then
    echo "ERROR: FL_SEND_PSE_AMI.py not found in $(pwd)/src/"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/fl_training_gpu_meetings_near_iid_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo "To view logs in real-time, run in another terminal: tail -f $LOG_FILE"
echo "Starting Python federated learning training script..."
echo ""

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run federated learning training with meeting-based subsampling
# NEW: Using --subset_*_meetings instead of --test_size for better near-IID distribution
# This avoids domain skew by selecting complete meetings instead of cutting segments
# Memory optimizations:
#   --chunk_size 250: Process 250 samples at a time (reduces peak memory)
#   --batch_size 8: Batch size for training (good GPU utilization)
#   --max_sequence_length 1000: Fixed sequence length (SEND uses 100-2000 frames)
# Federated Learning parameters:
#   --num_clients 2: Number of federated clients
#   --clients_per_round 2: Run all clients in parallel
#   --num_rounds 40: Number of federated learning rounds (30-50 recommended)
#   --epochs 1: Number of local training epochs per round (1-2 for FL)
# Meeting-based subsampling (NEW):
#   --subset_train_meetings 40: Use 40 meetings for training
#   --subset_val_meetings 10: Use 10 meetings for validation
#   --subset_test_meetings 10: Use 10 meetings for testing
#   --seed 42: Random seed for reproducibility
# Model architecture parameters (for faster training):
#   --hidden_dim 256: Reduced hidden dimension
#   --num_speech_encoder_layers 4: Reduced number of layers
#   --num_post_net_layers 3: Reduced number of layers
#   --num_transformer_layers 2: Reduced number of layers
# Early stopping parameters:
#   --early_stopping_patience 10: Stop if no improvement for 10 rounds
#   --early_stopping_min_delta 0.001: Minimum change to qualify as improvement
PYTHONUNBUFFERED=1 python src/FL_SEND_PSE_AMI.py \
  --subset_train_meetings 20 \
  --subset_val_meetings 5 \
  --subset_test_meetings 5 \
  --seed 42 \
  --epochs 1 \
  --num_rounds 25 \
  --num_clients 2 \
  --clients_per_round 2 \
  --hidden_dim 256 \
  --num_speech_encoder_layers 4 \
  --num_post_net_layers 3 \
  --num_transformer_layers 2 \
  --batch_size 8 \
  --chunk_size 250 \
  --max_sequence_length 1000 \
  --early_stopping_patience 10 \
  --early_stopping_min_delta 0.001 \
  > "$LOG_FILE" 2>&1

TRAIN_EXIT_CODE=$?
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== Federated Learning Training Complete ==="
    echo ""
    echo "Verification checklist:"
    echo "  1. Check logs for 'Stratified partitioning' message"
    echo "  2. Verify domain distribution is balanced across clients"
    echo "  3. Check that train/val/test have exactly 40/10/10 meetings"
    echo "  4. Verify DER is more stable (should not jump 50-80%)"
else
    echo "=== Federated Learning Training Failed with exit code $TRAIN_EXIT_CODE ==="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check for OOM errors in logs"
    echo "  2. Try reducing --batch_size or --chunk_size"
    echo "  3. Check available memory: free -h"
fi
echo "Logs saved to: $LOG_FILE"
echo "Script finished at: $(date)"
exit $TRAIN_EXIT_CODE

