#!/bin/bash
#SBATCH --job-name=fl_send_cpu_parallel_test
#SBATCH --time=10-00:00:00  # 10 days
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # For 2 clients × 4 CPUs each
#SBATCH --mem=16G  # Reduced memory for smaller test (2 clients, 1000 samples)
#SBATCH --partition=pascal  # Using pascal partition
#SBATCH --exclude=pascal-node03.l3s.intra  # Exclude node with TaskProlog configuration issue
#SBATCH --output=fl_training_cpu_parallel_test_%j.out
#SBATCH --error=fl_training_cpu_parallel_test_%j.err

# Enable debugging and ensure output is not buffered
set -x
# Disable output buffering
export PYTHONUNBUFFERED=1

# Ray memory configuration for CPU parallel mode
# Lower memory threshold for CPU mode
export RAY_memory_usage_threshold=0.85
# Check memory every second
export RAY_memory_monitor_refresh_ms=1000
# Set object store memory limit (5GB for CPU mode - reduced from 10GB)
export RAY_object_store_memory=5000000000
# Enable object spilling to disk when memory is low
export RAY_spill_objects_to_disk=1

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
    echo "ERROR: Environment flsend_clean not found on node!"
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
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

# Verify CPU mode (no GPU check needed)
echo "=== CPU Mode Verification ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', 'cuda' if torch.cuda.is_available() else 'cpu')" 2>&1

echo "=== Starting PARALLEL CPU Federated Learning Training ==="
echo "Configuration:"
echo "  - Device: CPU (forced)"
echo "  - Parallel clients per round: 2 (all clients in parallel)"
echo "  - CPUs per client: 4"
echo "  - Total clients: 2"
echo "  - Test size: 1000 samples"
echo ""

# Change to working directory
# Update this path to match your server's directory structure
# Common paths: ~/FL_SEND/28nov/FL_SEND or ~/FL_SEND/24oct/FL_SEND
# If script is run from project root, use current directory
if [ -f "src/FL_SEND_PSE_AMI.py" ]; then
    # Already in project root
    WORK_DIR=$(pwd)
else
    # Try to find project root or use common path
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
LOG_FILE="logs/fl_training_cpu_parallel_test_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo "To view logs in real-time, run in another terminal: tail -f $LOG_FILE"
echo "Starting Python federated learning training script (CPU parallel mode)..."
echo ""

# Add src to PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run federated learning training with CPU parallel mode
# Key changes from GPU version:
#   1. --device cpu: Forces CPU-only mode (disables CUDA)
#   2. --clients_per_round 2: Enables parallel execution (all 2 clients per round)
#   3. --num_cpus_per_client 4: Allocates 4 CPUs per client
#   4. Reduced test_size: 1000 samples (small test for verification)
#   5. Reduced max_sequence_length: 1000 (memory optimization)
# Memory optimizations (already applied in code):
#   - Variable-length features (no client-level padding)
#   - Per-sample meeting_ids (not per-frame)
#   - Float32 features (not float64)
#   - Explicit cleanup after fit/eval
PYTHONUNBUFFERED=1 python src/FL_SEND_PSE_AMI.py \
  --device cpu \
  --test_size 1000 \
  --epochs 2 \
  --num_rounds 5 \
  --num_clients 2 \
  --clients_per_round 2 \
  --num_cpus_per_client 4 \
  --hidden_dim 256 \
  --num_speech_encoder_layers 4 \
  --num_post_net_layers 3 \
  --num_transformer_layers 2 \
  --batch_size 4 \
  --chunk_size 250 \
  --max_sequence_length 1000 \
  --lr 1e-4 \
  > "$LOG_FILE" 2>&1

TRAIN_EXIT_CODE=$?
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== Federated Learning Training Complete ==="
    echo ""
    echo "Verification checklist:"
    echo "  1. Check logs for 'PARALLEL MODE' message"
    echo "  2. Verify multiple clients started fit simultaneously"
    echo "  3. Check for 'FORCING CPU mode (CUDA disabled)' message"
    echo "  4. Verify variable-length sequence logs"
    echo ""
    echo "To check logs:"
    echo "  tail -f $LOG_FILE"
    echo "  grep 'PARALLEL MODE' $LOG_FILE"
    echo "  grep 'CLIENT.*fit started' $LOG_FILE"
else
    echo "=== Federated Learning Training Failed with exit code $TRAIN_EXIT_CODE ==="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check for OOM errors in logs"
    echo "  2. Try reducing --clients_per_round to 1 (sequential mode)"
    echo "  3. Try reducing --test_size or --max_sequence_length"
    echo "  4. Check available memory: free -h"
fi
echo "Logs saved to: $LOG_FILE"
echo "Script finished at: $(date)"
exit $TRAIN_EXIT_CODE

