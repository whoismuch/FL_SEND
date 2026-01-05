#!/bin/bash
#SBATCH --job-name=fl_send_quick_test
#SBATCH --time=02:00:00  # 2 hours should be enough for quick test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal
#SBATCH --exclude=pascal-node03.l3s.intra
#SBATCH --output=fl_training_quick_test_%j.out
#SBATCH --error=fl_training_quick_test_%j.err

# Enable debugging and ensure output is not buffered
set -x
# Disable output buffering
export PYTHONUNBUFFERED=1

# Ray memory configuration to prevent OOM
export RAY_memory_usage_threshold=0.90
export RAY_memory_monitor_refresh_ms=1000
export RAY_object_store_memory=10000000000
export RAY_spill_objects_to_disk=1

# Debug: Output to .out file immediately
echo "=== SLURM Script Started (Quick Test) ==="
echo "Script: $0"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not set}"
echo ""

# Initialize conda
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

eval "$(conda shell.bash hook)" 2>/dev/null || true

# Find conda environment
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
echo "Python path: $(which python || echo 'NOT FOUND')"

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

echo "=== Starting Quick Federated Learning Test ==="
echo "Configuration:"
echo "  - test_size: 10000 samples"
echo "  - num_clients: 1"
echo "  - num_rounds: 1"
echo "  - epochs: 5 (local training epochs per round)"
echo "  - batch_size: 8"
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
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/fl_training_quick_test_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo "To view logs in real-time, run in another terminal: tail -f $LOG_FILE"
echo "Starting Python federated learning training script..."
echo ""

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run federated learning training with quick test parameters
# Quick test configuration:
#   --test_size 10000: Use 10000 samples
#   --num_clients 1: Single client (no aggregation needed, but tests FL pipeline)
#   --num_rounds 1: Single round
#   --epochs 5: 5 local training epochs to see weight changes
#   --batch_size 8: Standard batch size
#   --chunk_size 250: Process 250 samples at a time
#   --max_sequence_length 1000: Fixed sequence length
#   --lr 1e-4: Learning rate (can be passed via config)
PYTHONUNBUFFERED=1 python src/FL_SEND_PSE_AMI.py \
  --test_size 10000 \
  --num_clients 1 \
  --num_rounds 1 \
  --epochs 5 \
  --hidden_dim 256 \
  --num_speech_encoder_layers 4 \
  --num_post_net_layers 3 \
  --num_transformer_layers 2 \
  --batch_size 8 \
  --chunk_size 250 \
  --max_sequence_length 1000 \
  > "$LOG_FILE" 2>&1

TRAIN_EXIT_CODE=$?
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== Quick Test Complete ==="
    echo "Check the logs for debugging output:"
    echo "  - L2 norm differences (should be > 0 after training)"
    echo "  - Gradient norms (should be > 0 during training)"
    echo "  - Weight change confirmations"
    echo "  - FedAvg verification (if applicable)"
else
    echo "=== Quick Test Failed with exit code $TRAIN_EXIT_CODE ==="
fi
echo "Logs saved to: $LOG_FILE"
echo "Script finished at: $(date)"
exit $TRAIN_EXIT_CODE

