#!/bin/bash

# Script to run training with logs saved to file
# Usage: ./run_training_with_logs.sh

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate log filename with timestamp
LOG_FILE="logs/training_5000_samples_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training..."
echo "Logs will be saved to: $LOG_FILE"
echo "To view logs in real-time, run in another terminal: tail -f $LOG_FILE"
echo ""

# Run training and save all output to log file
# 2>&1 redirects stderr to stdout, so both go to the log file
# Using --test_size 5000 to limit dataset to 5000 samples (faster training for testing)
python src/SEND_PSE_AMI.py \
  --test_size 5000 \
  --epochs 1 \
  --hidden_dim 256 \
  --num_speech_encoder_layers 4 \
  --num_post_net_layers 3 \
  --num_transformer_layers 2 \
  --batch_size 4 \
  > "$LOG_FILE" 2>&1

echo ""
echo "Training completed. Logs saved to: $LOG_FILE"

