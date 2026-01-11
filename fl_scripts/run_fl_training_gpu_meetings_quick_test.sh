#!/bin/bash
#SBATCH --job-name=fl_send_gpu_meetings_test
#SBATCH --time=2-00:00:00  # 2 days for quick test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal
#SBATCH --exclude=pascal-node03.l3s.intra
#SBATCH --output=fl_training_gpu_meetings_quick_test_%j.out
#SBATCH --error=fl_training_gpu_meetings_quick_test_%j.err

# Enable debugging and ensure output is not buffered
set -x
export PYTHONUNBUFFERED=1

# Ray memory configuration
export RAY_memory_usage_threshold=0.90
export RAY_memory_monitor_refresh_ms=1000
export RAY_object_store_memory=5000000000
export RAY_spill_objects_to_disk=1

echo "=== SLURM Script Started ==="
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

echo "Active conda environment: ${CONDA_DEFAULT_ENV:-not set}"
echo "Python path: $(which python || echo 'NOT FOUND')"

echo "=== Job Environment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS_ON_NODE"

# Check GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>&1

echo "=== Quick Test: Federated Learning with Meeting-based Subsampling ==="
echo "Configuration:"
echo "  - Training meetings: 40"
echo "  - Validation meetings: 10"
echo "  - Test meetings: 10"
echo "  - Number of clients: 2"
echo "  - Clients per round: 2"
echo "  - Epochs per round: 1"
echo "  - Number of rounds: 10 (quick test)"
echo "  - Seed: 42"
echo ""

# Change to working directory
if [ -f "src/FL_SEND_PSE_AMI.py" ]; then
    WORK_DIR=$(pwd)
else
    WORK_DIR="${FL_SEND_WORK_DIR:-$HOME/FL_SEND/17dec_2/FL_SEND}"
fi

cd "$WORK_DIR" || {
    echo "ERROR: Failed to change to directory $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

if [ ! -f "src/FL_SEND_PSE_AMI.py" ]; then
    echo "ERROR: FL_SEND_PSE_AMI.py not found!"
    exit 1
fi

mkdir -p logs
LOG_FILE="logs/fl_training_gpu_meetings_quick_test_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Quick test with fewer rounds
PYTHONUNBUFFERED=1 python src/FL_SEND_PSE_AMI.py \
  --subset_train_meetings 25 \
  --subset_val_meetings 5 \
  --subset_test_meetings 5 \
  --seed 42 \
  --epochs 1 \
  --num_rounds 10 \
  --num_clients 2 \
  --clients_per_round 2 \
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
    echo "Check logs for:"
    echo "  - Stratified partitioning message"
    echo "  - Domain distribution per client"
    echo "  - Meeting counts (40/10/10)"
else
    echo "=== Quick Test Failed with exit code $TRAIN_EXIT_CODE ==="
fi
echo "Logs saved to: $LOG_FILE"
echo "Script finished at: $(date)"
exit $TRAIN_EXIT_CODE

