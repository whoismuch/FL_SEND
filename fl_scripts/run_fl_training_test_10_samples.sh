#!/bin/bash
#SBATCH --job-name=fl_send_test_10
#SBATCH --time=1-00:00:00  # 1 day (should be enough for 10 samples test)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal
#SBATCH --exclude=pascal-node03.l3s.intra
#SBATCH --output=fl_training_test_10_samples_%j.out
#SBATCH --error=fl_training_test_10_samples_%j.err

# Enable debugging and ensure output is not buffered
set -x
export PYTHONUNBUFFERED=1

echo "=== SLURM Test Script Started ==="
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
echo "Python: $(which python)"
echo "Python version: $(python --version 2>&1)"

echo "=== GPU Check ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>&1
nvidia-smi 2>&1 || echo "nvidia-smi not available"

echo "=== Starting Federated Learning Test on 10 SAMPLES ==="

# Change to working directory
if [ -f "src/FL_SEND_PSE_AMI.py" ]; then
    WORK_DIR=$(pwd)
else
    WORK_DIR="${FL_SEND_WORK_DIR:-$HOME/FL_SEND/17dec_2/FL_SEND}"
    if [ ! -f "$WORK_DIR/src/FL_SEND_PSE_AMI.py" ]; then
        echo "WARNING: Could not find FL_SEND_PSE_AMI.py at $WORK_DIR"
    fi
fi

cd "$WORK_DIR" || {
    echo "ERROR: Failed to change to directory $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

if [ ! -f "src/FL_SEND_PSE_AMI.py" ]; then
    echo "ERROR: FL_SEND_PSE_AMI.py not found in $(pwd)/src/"
    exit 1
fi

mkdir -p logs
LOG_FILE="logs/fl_training_test_10_samples_$(date +%Y%m%d_%H%M%S).log"

echo "Logs will be saved to: $LOG_FILE"
echo "Starting Python federated learning test script..."
echo ""

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Test run with minimal parameters:
# - 10 samples for quick test
# - 2 clients
# - 2 rounds
# - 1 epoch per round
# - Small model for fast testing
PYTHONUNBUFFERED=1 python src/FL_SEND_PSE_AMI.py \
  --test_size 10 \
  --epochs 1 \
  --num_rounds 2 \
  --num_clients 2 \
  --hidden_dim 128 \
  --num_speech_encoder_layers 2 \
  --num_post_net_layers 2 \
  --num_transformer_layers 1 \
  --batch_size 4 \
  --chunk_size 50 \
  --max_sequence_length 500 \
  > "$LOG_FILE" 2>&1

TRAIN_EXIT_CODE=$?
echo ""
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "=== Federated Learning Test Complete ==="
else
    echo "=== Federated Learning Test Failed with exit code $TRAIN_EXIT_CODE ==="
fi
echo "Logs saved to: $LOG_FILE"
echo "Script finished at: $(date)"
exit $TRAIN_EXIT_CODE

