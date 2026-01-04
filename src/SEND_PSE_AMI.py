import os
import logging
import re
import sys
import builtins
import warnings

# Disable numba debug output (IR - Intermediate Representation)
# This prevents verbose compilation details from appearing in logs
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled
os.environ['NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING'] = '1'  # Disable highlighting

# Suppress multiprocessing resource tracker warnings about leaked semaphores
# These warnings are harmless and occur when multiprocessing workers are not explicitly closed
# The warnings don't affect functionality and are common with libraries like librosa, numba, etc.
# Use comprehensive filtering to catch all variations of the warning
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*leaked semaphore.*')
# Also set environment variable to suppress at OS level
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'

# Additional suppression: intercept stderr to filter out resource_tracker warnings
# This is needed because some warnings bypass the warnings module
_original_stderr_write = sys.stderr.write
def _filtered_stderr_write(s):
    if 'resource_tracker' in s and 'leaked semaphore' in s:
        return  # Suppress the warning
    return _original_stderr_write(s)
sys.stderr.write = _filtered_stderr_write

# Try to configure numba to suppress debug output
try:
    import numba
    # Disable numba's verbose output
    numba.config.DISABLE_JIT = False  # Keep JIT enabled
    # Suppress numba warnings and debug info
    import warnings
    warnings.filterwarnings('ignore', category=numba.NumbaWarning)
except ImportError:
    pass  # numba not installed yet, will be imported later via librosa

# Configure root logger to output ALL logs to stdout with proper formatting
# This ensures all modules (data_processing, dataset_statistics, etc.) use the same configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create stdout handler for INFO and below (goes to .out file)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)  # Accept all levels, filter in handler
stdout_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                     datefmt='%Y-%m-%d %H:%M:%S')

# Filter: only INFO and DEBUG go to stdout, but exclude numba IR output
def stdout_filter(record):
    # Exclude numba IR (Intermediate Representation) debug output
    # Numba outputs verbose compilation details that clutter logs
    message = record.getMessage()
    
    # Skip messages that look like numba IR output
    numba_ir_indicators = [
        'label 0:',
        'label ',
        ' = arg(',
        ' = const(',
        ' = getitem(',
        ' = unary(',
        ' = binary_subscr',
        ' = compare_op',
        'branch ',
        'jump ',
        'return ',
        ' = call ',
        ' = load_global',
        ' = load_attr',
    ]
    
    # Check if message contains numba IR indicators
    if any(indicator in message for indicator in numba_ir_indicators):
        return False  # Filter out this message
    
    return record.levelno <= logging.INFO

stdout_handler.addFilter(stdout_filter)
stdout_handler.setFormatter(stdout_formatter)
root_logger.addHandler(stdout_handler)

# Create stderr handler for WARNING and above (goes to .err file)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)  # Only WARNING and above
stderr_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
stderr_handler.setFormatter(stderr_formatter)
root_logger.addHandler(stderr_handler)

# Get logger for this module
logger = logging.getLogger(__name__)

# Disable DEBUG logging for numba to avoid IR (Intermediate Representation) output
# Numba uses its own logger and outputs verbose compilation details at DEBUG level
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)  # Only show WARNING and above from numba

# Also disable DEBUG for librosa's numba-compiled functions
librosa_logger = logging.getLogger('librosa')
librosa_logger.setLevel(logging.WARNING)

# Override builtins.print globally to use logger for better log visibility
# This ensures all modules (including dataset_statistics) use the logger
_original_print = builtins.print
def print(*args, **kwargs):
    """Override builtins.print to use logger.info for better log visibility."""
    # Remove 'file' and 'flush' kwargs if present, as logger handles this
    kwargs.pop('file', None)
    kwargs.pop('flush', None)
    message = ' '.join(str(arg) for arg in args)
    # Use root logger to ensure it goes through our handlers
    root_logger.info(message)
    sys.stdout.flush()  # Ensure immediate output

# Replace builtins.print with our version
builtins.print = print

import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import librosa
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader, Dataset, random_split
# Removed federated learning imports
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from speechbrain.inference.speaker import EncoderClassifier
from datasets import load_dataset
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd
from data_processing import (
    split_data_for_clients, 
    process_training_data,
    process_validation_data,
    extract_features, 
    simulate_overlapping_speech, 
    group_by_meeting,
    prepare_data_loaders,
    power_set_encoding,
    calculate_der,
    compute_speaker_embeddings,
    OverlappingSpeechDataset
)
from dataset_statistics import (
    print_meeting_statistics,
    print_dataset_overview,
    print_grouping_results,
    print_experiment_config,
    print_training_progress,
    print_final_results,
    analyze_speaker_distribution,
    print_data_loading_info,
    print_power_set_encoder_examples,
    print_send_model_statistics,
    print_client_split_statistics
)
import time
import argparse




# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class PowerSetEncoder:
    """Power Set Encoding for overlapping speech diarization with limited overlap."""
    def __init__(self, max_speakers: int = 4, max_overlap: int = None):
        self.max_speakers = max_speakers
        self.max_overlap = max_overlap if max_overlap is not None else max_speakers
        
        # Calculate number of classes using C(K,N) formula
        from math import comb
        self.num_classes = sum(comb(max_speakers, k) for k in range(self.max_overlap + 1))
        
        # Create mapping from speaker combinations to class indices
        self._create_mapping()
        
    def _create_mapping(self):
        """Create mapping from speaker combinations to class indices."""
        self.combination_to_class = {}
        self.class_to_combination = {}
        
        class_idx = 0
        # Generate all combinations with up to max_overlap speakers
        for k in range(self.max_overlap + 1):
            from itertools import combinations
            for combo in combinations(range(self.max_speakers), k):
                self.combination_to_class[combo] = class_idx
                self.class_to_combination[class_idx] = list(combo)
                class_idx += 1
    
    def encode(self, speaker_labels: List[int]) -> int:
        """Encode speaker labels (as speaker IDs) into a single integer using power set encoding."""
        if any(label >= self.max_speakers or label < 0 for label in speaker_labels):
            raise ValueError(f"Speaker ID in labels exceeds max_speakers ({self.max_speakers}) or is negative.")
        
        if len(speaker_labels) > self.max_overlap:
            raise ValueError(f"Number of speakers ({len(speaker_labels)}) exceeds max_overlap ({self.max_overlap}).")
        
        # Convert to tuple for mapping lookup
        combo = tuple(sorted(speaker_labels))
        return self.combination_to_class[combo]
    
    def decode(self, encoded_value: int) -> List[int]:
        """Decode an encoded value back into a list of active speaker IDs."""
        if encoded_value < 0 or encoded_value >= self.num_classes:
            raise ValueError(f"Encoded value {encoded_value} is out of range [0, {self.num_classes-1}]")
        
        return self.class_to_combination[encoded_value].copy()

class FSMNLayer(nn.Module):
    """Feedforward Sequential Memory Network layer - VECTORIZED VERSION."""
    def __init__(self, input_dim: int, hidden_dim: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        if input_dim != hidden_dim:
            self.input_lin = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_lin = None
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.memory = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        if self.input_lin is not None:
            x = self.input_lin(x)
        # Now x shape: (batch_size, seq_len, hidden_dim)
        h = self.linear(x)  # (batch_size, seq_len, hidden_dim)
        
        # VECTORIZED: Replace Python loop with efficient cumsum-based approach
        batch_size, seq_len, hidden_dim = x.shape
        
        if seq_len == 0:
            return h
        
        # Pre-allocate memory tensor
        memory = torch.zeros_like(h)
        
        # Use cumulative sum for efficient sliding window averages
        # For position i, we need mean of x[:, max(0, i-stride):i+1]
        # Strategy: Use cumsum and subtract to get window sums, then divide by window size
        
        # Compute cumulative sum along sequence dimension
        cumsum = torch.cumsum(x, dim=1)  # (batch_size, seq_len, hidden_dim)
        
        # For each position i, window is [max(0, i-stride), i+1)
        # Window size = min(stride + 1, i + 1)
        indices = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        window_sizes = torch.clamp(indices + 1, max=self.stride + 1)  # (seq_len,)
        window_sizes = window_sizes.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        # For positions where i >= stride, we need to subtract cumsum at position (i - stride)
        # For positions where i < stride, we just use cumsum[i]
        if self.stride > 0 and seq_len > self.stride:
            # Pad cumsum with zeros at the beginning for easier indexing
            # pad_cumsum[0] = 0, pad_cumsum[1:] = cumsum
            pad_cumsum = F.pad(cumsum, (0, 0, 1, 0), mode='constant', value=0.0)
            # Get start positions: max(0, i - stride) -> i - stride for i >= stride, 0 otherwise
            start_positions = torch.clamp(torch.arange(seq_len, device=x.device) - self.stride, min=0)
            # Index pad_cumsum: pad_cumsum[:, start_positions, :]
            # pad_cumsum shape: (batch_size, seq_len+1, hidden_dim)
            # start_positions shape: (seq_len,)
            # Simple indexing: pad_cumsum[:, start_positions, :] works but needs proper shape
            # Use advanced indexing: for each batch, get cumsum at start_positions
            start_cumsum = pad_cumsum[:, start_positions]  # (batch_size, seq_len, hidden_dim)
            window_sums = cumsum - start_cumsum  # (batch_size, seq_len, hidden_dim)
        else:
            # All windows start at 0
            window_sums = cumsum
        
        # Compute window means
        window_means = window_sums / window_sizes  # (batch_size, seq_len, hidden_dim)
        
        # Apply memory linear layer
        memory = self.memory(window_means)
        
        return h + memory

class SENDModel(nn.Module):
    """Speaker Embedding-aware Neural Diarization model with Power-Set Encoding."""
    def __init__(self, input_dim: int = 80, hidden_dim: int = 512, num_classes: int = 16, dropout_p: float = 0.1,
                 num_speech_encoder_layers: int = 8, num_post_net_layers: int = 6, num_transformer_layers: int = 4):
        super().__init__()
        # Speech Encoder (FSMN) - configurable number of layers
        # Added LayerNorm for better training stability (prevents gradient explosion)
        self.speech_encoder = nn.ModuleList([
            nn.Sequential(
                FSMNLayer(input_dim if i == 0 else hidden_dim, hidden_dim, stride=2**i),
                nn.LayerNorm(hidden_dim),  # Normalize across features (not batch/sequence)
                nn.Dropout(dropout_p)
            ) for i in range(num_speech_encoder_layers)
        ])
        # Speaker Encoder (MLP) with LayerNorm and Dropout after each activation
        # LayerNorm stabilizes activations and helps prevent gradient explosion
        self.speaker_encoder = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_p)
        )
        # CI Scorer (Context-Independent)
        self.ci_scorer = nn.Linear(hidden_dim, 1)
        # CD Scorer (Context-Dependent) - configurable number of layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_p,
            batch_first=True
        )
        self.cd_scorer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # Post-Net (FSMN) with LayerNorm and Dropout after each layer - configurable number of layers
        # Added LayerNorm for better training stability
        self.post_net = nn.ModuleList([
            nn.Sequential(
                FSMNLayer(hidden_dim, hidden_dim, stride=2**i),
                nn.LayerNorm(hidden_dim),  # Normalize across features
                nn.Dropout(dropout_p)
            ) for i in range(num_post_net_layers)
        ])
        # Final classification with LayerNorm for stability
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )
        # Adapter for combining CI and CD scores
        self.combine_adapter = None

    def forward(self, x: torch.Tensor, speaker_embeddings: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        # speaker_embeddings shape: (batch_size, num_speakers, 192)
        batch_size, seq_len, _ = x.shape
        num_speakers = speaker_embeddings.size(1)
        # Process audio features through Speech Encoder (with Dropout)
        for fsmn_dropout in self.speech_encoder:
            x = fsmn_dropout(x)
        # Process speaker embeddings (with Dropout)
        speaker_features = self.speaker_encoder(speaker_embeddings)  # (batch_size, num_speakers, hidden_dim)
        # CI Scoring - VECTORIZED: Replace Python loop with batch matrix multiplication
        # x: (batch_size, seq_len, hidden_dim)
        # speaker_features: (batch_size, num_speakers, hidden_dim)
        # We want: (batch_size, seq_len, num_speakers) - dot product for each speaker at each time step
        # Use batch matrix multiplication: x @ speaker_features.transpose(-2, -1)
        ci_scores = torch.bmm(x, speaker_features.transpose(1, 2))  # (batch_size, seq_len, num_speakers)
        # CD Scoring
        cd_scores = self.cd_scorer(x)  # (batch_size, seq_len, hidden_dim)
        # Combine CI and CD scores
        # ci_scores: (batch_size, seq_len, num_speakers)
        # cd_scores: (batch_size, seq_len, hidden_dim)
        # Concatenate along feature dimension: (batch_size, seq_len, num_speakers + hidden_dim)
        combined = torch.cat([
            ci_scores,
            cd_scores
        ], dim=2)  # (batch_size, seq_len, num_speakers + hidden_dim)
        # Create adapter if it doesn't exist or dimensions have changed
        if self.combine_adapter is None or self.combine_adapter.in_features != combined.size(-1):
            self.combine_adapter = nn.Linear(combined.size(-1), self.post_net[0][0].input_dim).to(combined.device)
        # Adapt dimensions before Post-Net
        combined = self.combine_adapter(combined)
        # Process through Post-Net (with Dropout)
        for fsmn_dropout in self.post_net:
            combined = fsmn_dropout(combined)
        # Final classification
        out = self.classifier(combined)
        return out


# Centralized training functions
def train_model(model, train_loader, val_loader, device, power_set_encoder, epochs=50, compute_der_during_training=False, progress_log_file=None, early_stopping_patience=5, early_stopping_min_delta=0.001, debug_mode=False, debug_max_batches=200, learning_rate=3e-4):
    """Train the SEND model in centralized manner with early stopping.
    
    Args:
        debug_mode: If True, limit training to debug_max_batches and print detailed stats
        debug_max_batches: Maximum number of batches to process in debug mode (default: 200)
        learning_rate: Learning rate for optimizer (default: 3e-4, reduced from 1e-3 for stability)
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding labels (-100)
    
    # Print learning rate information
    print(f"✅ Using learning rate: {learning_rate}")
    if learning_rate > 1e-3:
        print(f"⚠️  WARNING: Learning rate {learning_rate} is high. Consider using 3e-4 or lower for stability.")
    
    # Enable Mixed Precision Training for faster GPU computation (2-3x speedup)
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("✅ Mixed Precision Training (AMP) ENABLED - will speed up training significantly")
    else:
        print("⚠️  Mixed Precision Training (AMP) DISABLED - CUDA not available")
    
    # Enable cuDNN benchmark for faster convolutions (only if input sizes are constant)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("✅ cuDNN benchmark ENABLED - will optimize convolution operations")
        # Set float32 matmul precision for stability (PyTorch 2.x)
        try:
            torch.set_float32_matmul_precision("high")
            print("✅ Float32 matmul precision set to 'high' for stability")
        except AttributeError:
            # PyTorch < 2.0 doesn't have this function
            pass
        # Enable TF32 for faster matmul on Ampere+ GPUs (if available)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            print("✅ TF32 enabled for faster matmul operations (Ampere+ GPUs)")
        except AttributeError:
            pass
    
    epoch_metrics = []
    
    # Early stopping variables - now using DER instead of loss
    best_val_der = float('inf')  # DER: lower is better
    best_val_loss = float('inf')  # Keep for logging
    patience_counter = 0
    best_model_state = None
    
    # Initialize progress logging
    if progress_log_file:
        with open(progress_log_file, 'w') as f:
            f.write(f"{'EPOCH':<6} {'LOSS':<12} {'DER':<10} {'ACCURACY':<10} {'VAL_LOSS':<12} {'VAL_DER':<10} {'TIMESTAMP':<10}\n")
            f.write("="*80 + "\n")
    
    for epoch in range(epochs):
        print(f" Starting epoch {epoch+1}/{epochs}")
        train_loss = 0.0
        batch_losses = []
        nan_batches = 0  # Track batches with NaN/Inf loss
        # Group predictions by meeting_id for proper DER calculation
        pred_by_rec = defaultdict(list)
        lab_by_rec = defaultdict(list)
            
        total_batches = len(train_loader)
        # Debug mode: limit batches
        max_batches = debug_max_batches if debug_mode else total_batches
        if debug_mode:
            print(f"[DEBUG MODE] Limiting training to {max_batches} batches per epoch")
        print(f" Epoch {epoch+1}/{epochs}: Processing {min(max_batches, total_batches)} batches...")
        
        for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(train_loader):
            if debug_mode and batch_idx >= max_batches:
                print(f"[DEBUG MODE] Reached max_batches={max_batches}, stopping epoch early")
                break
            # TIMING: Measure batch processing time (only for first 2 batches)
            if batch_idx < 2:
                batch_start_time = time.time()
                load_time = batch_start_time  # Approximate load time (will be refined)
            
            if batch_idx == 0:
                print(f" First batch in epoch {epoch+1}")
            
            # Load data to GPU (non-blocking for overlap with computation)
            load_start = time.time() if batch_idx < 2 else None
            features, speaker_embeddings, labels = features.to(device, non_blocking=True), speaker_embeddings.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            speaker_embeddings = speaker_embeddings.float()
            load_time = (time.time() - load_start) if load_start else 0.0
            
            # Check input tensors for NaN/Inf before forward pass
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.warning(f"Batch {batch_idx}: NaN/Inf detected in features. Skipping batch.")
                logger.warning(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}, has_nan={torch.isnan(features).any().item()}, has_inf={torch.isinf(features).any().item()}")
                continue
            if torch.isnan(speaker_embeddings).any() or torch.isinf(speaker_embeddings).any():
                logger.warning(f"Batch {batch_idx}: NaN/Inf detected in speaker_embeddings. Skipping batch.")
                logger.warning(f"  Speaker embeddings stats: min={speaker_embeddings.min().item():.4f}, max={speaker_embeddings.max().item():.4f}, mean={speaker_embeddings.mean().item():.4f}")
                continue
            
            optimizer.zero_grad(set_to_none=True)
            did_step = False  # Track if scaler.step() was executed
            
            # Forward pass
            forward_start = time.time() if batch_idx < 2 else None
            # Use Mixed Precision Training if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features, speaker_embeddings)
                    batch_size, seq_len, num_classes = outputs.shape
                    outputs = outputs.reshape(-1, num_classes)
                    labels_flat = labels.reshape(-1)
                    
                    # Check for valid frames: skip batch if all labels are padding (-100)
                    valid = (labels_flat != -100)
                    if valid.sum() == 0:
                        logger.warning(f"Batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                        continue
                    
                    # Validate labels: ignore_index=-100, others in [0, num_classes-1]
                    invalid_labels = (labels_flat >= num_classes) | ((labels_flat < 0) & (labels_flat != -100))
                    if invalid_labels.any():
                        invalid_count = invalid_labels.sum().item()
                        invalid_values = labels_flat[invalid_labels].unique().cpu().numpy()
                        logger.error(f"Batch {batch_idx}: Found {invalid_count} invalid labels! Invalid values: {invalid_values}, num_classes={num_classes}")
                        logger.error(f"  Label stats: min={labels_flat.min().item()}, max={labels_flat.max().item()}, unique={torch.unique(labels_flat).cpu().numpy()}")
                        logger.warning(f"  Skipping batch {batch_idx} due to invalid labels")
                        continue
                    
                    loss = criterion(outputs, labels_flat)
                    
                    # Check for NaN/Inf in loss or outputs (early detection)
                    if not torch.isfinite(loss):
                        nan_batches += 1
                        logger.warning(f"Batch {batch_idx}: NaN/Inf loss detected! Loss: {loss.item()}")
                        logger.warning(f"  Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                        logger.warning(f"  Logits stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                        logger.warning(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
                        logger.warning(f"  Labels stats: min={labels_flat.min().item()}, max={labels_flat.max().item()}, unique={torch.unique(labels_flat).cpu().numpy()[:10]}")
                        # Get label distribution for debugging
                        unique_labels_tensor, counts = torch.unique(labels_flat, return_counts=True)
                        label_dist = {int(l): int(c) for l, c in zip(unique_labels_tensor.cpu().numpy(), counts.cpu().numpy())}
                        logger.warning(f"  Label distribution: {label_dist}")
                        # Skip batch: no backward, no step, no update
                        continue
                
                # Backward pass
                backward_start = time.time() if batch_idx < 2 else None
                scaler.scale(loss).backward()
                # Gradient clipping to prevent gradient explosion (fixes NaN loss issue)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                did_step = True  # Mark that step was executed
                scaler.update()  # Update scaler only after step was executed
                backward_time = (time.time() - backward_start) if backward_start else 0.0
            else:
                outputs = model(features, speaker_embeddings)
                batch_size, seq_len, num_classes = outputs.shape
                outputs = outputs.reshape(-1, num_classes)
                labels_flat = labels.reshape(-1)
                
                # Check for valid frames: skip batch if all labels are padding (-100)
                valid = (labels_flat != -100)
                if valid.sum() == 0:
                    logger.warning(f"Batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                    continue
                
                # Validate labels: ignore_index=-100, others in [0, num_classes-1]
                invalid_labels = (labels_flat >= num_classes) | ((labels_flat < 0) & (labels_flat != -100))
                if invalid_labels.any():
                    invalid_count = invalid_labels.sum().item()
                    invalid_values = labels_flat[invalid_labels].unique().cpu().numpy()
                    logger.error(f"Batch {batch_idx}: Found {invalid_count} invalid labels! Invalid values: {invalid_values}, num_classes={num_classes}")
                    logger.error(f"  Label stats: min={labels_flat.min().item()}, max={labels_flat.max().item()}, unique={torch.unique(labels_flat).cpu().numpy()}")
                    logger.warning(f"  Skipping batch {batch_idx} due to invalid labels")
                    continue
                
                loss = criterion(outputs, labels_flat)
                
                # Check for NaN/Inf in loss or outputs (early detection)
                if not torch.isfinite(loss):
                    nan_batches += 1
                    logger.warning(f"Batch {batch_idx}: NaN/Inf loss detected! Loss: {loss.item()}")
                    logger.warning(f"  Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                    logger.warning(f"  Logits stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                    logger.warning(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
                    logger.warning(f"  Labels stats: min={labels_flat.min().item()}, max={labels_flat.max().item()}, unique={torch.unique(labels_flat).cpu().numpy()[:10]}")
                    # Get label distribution for debugging
                    unique_labels_tensor, counts = torch.unique(labels_flat, return_counts=True)
                    label_dist = {int(l): int(c) for l, c in zip(unique_labels_tensor.cpu().numpy(), counts.cpu().numpy())}
                    logger.warning(f"  Label distribution: {label_dist}")
                    # Skip batch: no backward, no step
                    continue
                
                # Backward pass
                backward_start = time.time() if batch_idx < 2 else None
                loss.backward()
                # Gradient clipping to prevent gradient explosion (fixes NaN loss issue)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                backward_time = (time.time() - backward_start) if backward_start else 0.0
            
            forward_time = (time.time() - forward_start) if forward_start else 0.0
            total_batch_time = (time.time() - batch_start_time) if batch_idx < 2 else 0.0
            
            # Check loss value before adding (additional safety check)
            loss_value = loss.item()
            if np.isnan(loss_value) or np.isinf(loss_value):
                logger.warning(f"Batch {batch_idx}: NaN/Inf loss value detected after backward pass! Loss: {loss_value}. Skipping batch.")
                continue
            
            # Log batch statistics (for first few batches and periodically)
            if batch_idx < 3 or batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx}: loss={loss_value:.4f}, features_shape={features.shape}, labels_unique={torch.unique(labels_flat).cpu().numpy()[:10]}")
            
            train_loss += loss_value
            batch_losses.append(loss_value)
            
            # TIMING: Log timing for first 2 batches
            if batch_idx < 2:
                print(f" ⏱️  Batch {batch_idx} timing: load={load_time:.4f}s, forward={forward_time:.4f}s, backward={backward_time:.4f}s, total={total_batch_time:.4f}s")
            
            # DER COMPUTATION: COMPLETELY DISABLED during training for speed
            # Only compute predictions if explicitly requested (for debugging)
            # NO CPU↔GPU copies during training loop - all operations stay on GPU
            if compute_der_during_training:
                # Only compute if explicitly enabled (not recommended for speed)
                predictions = torch.argmax(outputs, dim=-1)
                # NOTE: CPU copies only happen if DER is explicitly enabled
                predictions_np = predictions.cpu().numpy()
                labels_np = labels_flat.cpu().numpy()
                meeting_ids_flat = np.concatenate(meeting_ids, axis=0)
                for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                    if meeting_id is not None:
                        pred_by_rec[meeting_id].append(pred)
                        lab_by_rec[meeting_id].append(label)
            
            # Progress indicator for batches
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                print(f" Epoch {epoch+1} Progress: {progress:.1f}% ({batch_idx+1}/{total_batches}) - Loss: {loss.item():.4f}")
            
            # Sanity-check: Analyze label distribution (especially for first few batches or in debug mode)
            if batch_idx < 5 or debug_mode:  # Check first 5 batches or all batches in debug mode
                unique_labels = torch.unique(labels_flat).cpu().numpy()
                num_padding = (labels_flat == -100).sum().item()
                total_frames = labels_flat.numel()
                padding_ratio = num_padding / total_frames if total_frames > 0 else 0.0
                num_valid_labels = total_frames - num_padding
                valid_unique_labels = unique_labels[unique_labels != -100]
                
                # Get label distribution
                unique_labels_tensor, counts = torch.unique(labels_flat, return_counts=True)
                label_dist = {int(l): int(c) for l, c in zip(unique_labels_tensor.cpu().numpy(), counts.cpu().numpy())}
                
                print(f"Train batch {batch_idx} sanity-check:")
                print(f"  - Labels shape: {labels_flat.shape}")
                print(f"  - Unique labels: {unique_labels}")
                print(f"  - Valid unique labels (excluding -100): {valid_unique_labels}")
                print(f"  - Label distribution: {label_dist}")
                print(f"  - Padding ratio: {padding_ratio:.2%} ({num_padding}/{total_frames} frames)")
                print(f"  - Valid frames: {num_valid_labels}")
                print(f"  - Loss: {loss_value:.4f}, outputs_range=[{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
                
                if padding_ratio > 0.9:
                    logger.warning(f"⚠️  WARNING: Batch {batch_idx} has >90% padding ({padding_ratio:.2%})! This may indicate data pipeline issues.")
                
                if compute_der_during_training or debug_mode:
                    predictions = torch.argmax(outputs, dim=-1)
                    unique_preds = torch.unique(predictions).cpu().numpy()
                    unique_preds_tensor, pred_counts = torch.unique(predictions, return_counts=True)
                    pred_dist = {int(p): int(c) for p, c in zip(unique_preds_tensor.cpu().numpy(), pred_counts.cpu().numpy())}
                    print(f"  - Unique predictions: {unique_preds}")
                    print(f"  - Prediction distribution: {pred_dist}")
            
        # Calculate DER per recording and aggregate (only if requested)
        ders = {}
        if compute_der_during_training:
            print(f" Computing DER for {len(pred_by_rec)} recordings...")
            for i, rec_id in enumerate(pred_by_rec):
                if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                    print(f" Processing recording {i+1}/{len(pred_by_rec)}: {rec_id}")
                    # Get speaker_id_list from the dataset
                    speaker_id_list = train_loader.dataset.get_speaker_id_list() if hasattr(train_loader.dataset, 'get_speaker_id_list') else None
                    ders[rec_id] = calculate_der(
                        pred_by_rec[rec_id],
                        lab_by_rec[rec_id],
                        power_set_encoder,
                        speaker_id_list=speaker_id_list,
                        debug=False,
                        frame_shift=0.01,
                        uri=rec_id
                    )
                    print(f" Recording {rec_id} DER: {ders[rec_id]:.4f}")
        else:
            print(f" Skipping DER computation during training for speed (set compute_der_during_training=True to enable)")
        
        # Log NaN/Inf batch statistics
        if nan_batches > 0:
            logger.warning(f"Epoch {epoch+1}: {nan_batches} batches had NaN/Inf loss, {len(batch_losses)} valid batches remaining")
        
        # Metrics per epoch
        mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
        # Calculate accuracy across all frames (only if DER computation was enabled)
        if compute_der_during_training:
            all_predictions = []
            all_labels = []
            for rec_id in pred_by_rec:
                all_predictions.extend(pred_by_rec[rec_id])
                all_labels.extend(lab_by_rec[rec_id])
            acc = (np.array(all_predictions) == np.array(all_labels)).mean() if all_labels else float('nan')
        else:
            acc = float('nan')  # Accuracy not computed when DER is disabled
        # Average DER across recordings (only if computed)
        der = np.mean(list(ders.values())) if ders else float('nan')
        if compute_der_during_training:
            print(f"[DEBUG] Epoch {epoch+1}/{epochs} unique labels: {np.unique(all_labels) if all_labels else 'EMPTY'}")
            print(f"[DEBUG] Epoch {epoch+1}/{epochs} unique predictions: {np.unique(all_predictions) if all_predictions else 'EMPTY'}")
        
        # CAPS progress output
        der_display = f"{der:.4f}" if compute_der_during_training and not np.isnan(der) else "SKIPPED"
        acc_display = f"{acc:.4f}" if not np.isnan(acc) else "N/A"
        loss_display = f"{mean_loss:.4f}" if not np.isnan(mean_loss) else "N/A"
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs} COMPLETED")
        print(f"LOSS: {loss_display}")
        print(f"DER:  {der_display}")
        print(f"ACC:  {acc_display}")
        print(f"TIME: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}\n")
        
        print(f" Epoch {epoch+1}/{epochs} summary: min_loss={min(batch_losses) if batch_losses else 'nan'}, max_loss={max(batch_losses) if batch_losses else 'nan'}, mean_loss={mean_loss}, acc={acc}, DER={der if compute_der_during_training else 'skipped'}")
        
        # Validation after each epoch (DER enabled for early stopping)
        print(f" Running validation for epoch {epoch+1}...")
        val_loss, val_der, _, _ = evaluate_model(model, val_loader, device, power_set_encoder, compute_der=True)
        
        # Early stopping logic based on DER (lower is better)
        # Skip if val_der is NaN/Inf, fallback to loss if DER not available
        if np.isnan(val_der) or np.isinf(val_der):
            # Fallback to loss-based early stopping if DER is invalid
            logger.warning(f"Validation DER is NaN/Inf at epoch {epoch+1}, falling back to loss-based early stopping")
            if np.isnan(val_loss) or np.isinf(val_loss):
                logger.warning(f"Validation loss is also NaN/Inf at epoch {epoch+1}, skipping early stopping check")
                patience_counter += 1
                print(f" ⚠️  Invalid validation metrics (NaN/Inf) - No improvement for {patience_counter}/{early_stopping_patience} epochs")
            else:
                improvement = best_val_loss - val_loss
                if improvement > early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    print(f" ✅ Validation improved (loss)! New best val_loss: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f" ⚠️  No improvement (loss) for {patience_counter}/{early_stopping_patience} epochs")
        else:
            # Primary: DER-based early stopping (lower DER is better)
            improvement = best_val_der - val_der  # Positive improvement means DER decreased
            if improvement > early_stopping_min_delta:
                best_val_der = val_der
                best_val_loss = val_loss  # Also track best loss for logging
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f" ✅ Validation improved (DER)! New best val_der: {val_der:.4f}, val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f" ⚠️  No improvement (DER) for {patience_counter}/{early_stopping_patience} epochs (best DER: {best_val_der:.4f}, current: {val_der:.4f})")
        
        # CAPS progress output with validation metrics
        val_der_display = f"{val_der:.4f}" if not np.isnan(val_der) else "N/A"
        val_loss_display = f"{val_loss:.4f}" if not np.isnan(val_loss) else "N/A"
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs} COMPLETED")
        print(f"TRAIN LOSS: {loss_display}")
        print(f"TRAIN DER:  {der_display}")
        print(f"TRAIN ACC:  {acc_display}")
        print(f"VAL LOSS:   {val_loss_display}")
        # Show (BEST) marker if this is the best DER so far
        best_marker = ""
        if not np.isnan(val_der) and not np.isnan(best_val_der) and abs(val_der - best_val_der) < 1e-6:
            best_marker = "  (BEST)"
        print(f"VAL DER:    {val_der_display}{best_marker}")
        print(f"TIME:       {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Log to file with validation metrics
        if progress_log_file:
            with open(progress_log_file, 'a') as f:
                f.write(f"{epoch+1:<6} {loss_display:<12} {der_display:<10} {acc_display:<10} {val_loss_display:<12} {val_der_display:<10} {datetime.now().strftime('%H:%M:%S'):<10}\n")
        
        # Collect metrics for this epoch
        epoch_metrics.append({
            "train_loss": float(mean_loss),
            "acc": float(acc) if not np.isnan(acc) else None,
            "der": float(der) if not np.isnan(der) and compute_der_during_training else None,
            "val_loss": float(val_loss) if not np.isnan(val_loss) else None,
            "val_der": float(val_der) if not np.isnan(val_der) else None,
        })
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f" 🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f" Best validation DER: {best_val_der:.4f}, Best validation loss: {best_val_loss:.4f}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f" ✅ Restored best model (val_der: {best_val_der:.4f}, val_loss: {best_val_loss:.4f})")
    
    return epoch_metrics

def evaluate_model(model, val_loader, device, power_set_encoder, compute_der=False):
    """Evaluate the SEND model.
    
    Args:
        model: SEND model to evaluate
        val_loader: Validation data loader
        device: Device to run on
        power_set_encoder: Power set encoder for DER calculation
        compute_der: If False (default), skip DER computation for speed. Set True only when needed.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding labels (-100)
    val_loss = 0.0
    batch_losses = []
    nan_batches = 0  # Track batches with NaN/Inf loss
    # Group predictions by meeting_id for proper DER calculation (only if compute_der=True)
    pred_by_rec = defaultdict(list)
    lab_by_rec = defaultdict(list)
    
    # Use Mixed Precision for evaluation if available
    use_amp = torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(val_loader):
            if batch_idx == 0:
                print(f" First batch in evaluation")
            features, speaker_embeddings, labels = features.to(device, non_blocking=True), speaker_embeddings.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            speaker_embeddings = speaker_embeddings.float()
            
            # Use Mixed Precision if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(features, speaker_embeddings)
                    batch_size, seq_len, num_classes = outputs.shape
                    outputs = outputs.reshape(-1, num_classes)
                    labels_flat = labels.reshape(-1)
                    
                    # Check for valid frames: skip batch if all labels are padding (-100)
                    valid = (labels_flat != -100)
                    if valid.sum() == 0:
                        logger.warning(f"Eval batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                        continue
                    
                    loss = criterion(outputs, labels_flat)
            else:
                outputs = model(features, speaker_embeddings)
                batch_size, seq_len, num_classes = outputs.shape
                outputs = outputs.reshape(-1, num_classes)
                labels_flat = labels.reshape(-1)
                
                # Check for valid frames: skip batch if all labels are padding (-100)
                valid = (labels_flat != -100)
                if valid.sum() == 0:
                    logger.warning(f"Eval batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                    continue
                
                loss = criterion(outputs, labels_flat)

            # Check for NaN/Inf in loss using torch.isfinite (more robust)
            if not torch.isfinite(loss):
                nan_batches += 1
                loss_value = loss.item()
                logger.warning(f"Eval batch {batch_idx}: NaN/Inf loss detected! Loss: {loss_value}")
                logger.warning(f"  Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                # Skip this batch but continue validation
                continue

            loss_value = loss.item()
            val_loss += loss_value
            batch_losses.append(loss_value)
            
            # DER COMPUTATION: Only if explicitly enabled (disabled by default for speed)
            if compute_der:
                predictions = torch.argmax(outputs, dim=-1)
                # CPU copies only when DER is needed
                predictions_np = predictions.cpu().numpy()
                labels_np = labels_flat.cpu().numpy()
                meeting_ids_flat = np.concatenate(meeting_ids, axis=0)
                for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                    if meeting_id is not None:
                        pred_by_rec[meeting_id].append(pred)
                        lab_by_rec[meeting_id].append(label)
            
            # Sanity-check: Analyze label distribution (especially for first few batches)
            if batch_idx < 5:  # Check first 5 batches
                unique_labels = torch.unique(labels_flat).cpu().numpy()
                num_padding = (labels_flat == -100).sum().item()
                total_frames = labels_flat.numel()
                padding_ratio = num_padding / total_frames if total_frames > 0 else 0.0
                num_valid_labels = total_frames - num_padding
                valid_unique_labels = unique_labels[unique_labels != -100]
                
                print(f"Eval batch {batch_idx} sanity-check:")
                print(f"  - Labels shape: {labels_flat.shape}")
                print(f"  - Unique labels: {unique_labels}")
                print(f"  - Valid unique labels (excluding -100): {valid_unique_labels}")
                print(f"  - Padding ratio: {padding_ratio:.2%} ({num_padding}/{total_frames} frames)")
                print(f"  - Valid frames: {num_valid_labels}")
                
                if padding_ratio > 0.9:
                    logger.warning(f"⚠️  WARNING: Batch {batch_idx} has >90% padding ({padding_ratio:.2%})! This may indicate data pipeline issues.")
                
                if compute_der:
                    predictions = torch.argmax(outputs, dim=-1)
                    unique_preds = torch.unique(predictions).cpu().numpy()
                    print(f"  - Unique predictions: {unique_preds}")
    
    # Log NaN/Inf batch statistics
    if nan_batches > 0:
        logger.warning(f"Validation: {nan_batches} batches had NaN/Inf loss, {len(batch_losses)} valid batches remaining")
    
    # Calculate DER per recording and aggregate (only if compute_der=True)
    ders = {}
    if compute_der:
        print(f" Computing DER for {len(pred_by_rec)} recordings in evaluation...")
        for i, rec_id in enumerate(pred_by_rec):
            if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                print(f" Processing recording {i+1}/{len(pred_by_rec)}: {rec_id}")
                speaker_id_list = val_loader.dataset.get_speaker_id_list() if hasattr(val_loader.dataset, 'get_speaker_id_list') else None
                ders[rec_id] = calculate_der(
                    pred_by_rec[rec_id],
                    lab_by_rec[rec_id],
                    power_set_encoder,
                    speaker_id_list=speaker_id_list,
                    debug=False,
                    frame_shift=0.01,
                    uri=rec_id
                )
                print(f" Recording {rec_id} DER: {ders[rec_id]:.4f}")
    else:
        # This should not happen in normal flow since DER is now always computed for validation
        # But keep this branch for backward compatibility
        print(f" ⚠️  NOTE: DER computation was skipped (this is unexpected in normal flow)")
    
    # Check if batch_losses is empty (all batches were NaN/Inf or skipped)
    if not batch_losses:
        logger.error("No valid validation batches (all losses were NaN/Inf or skipped)")
        if compute_der and not ders:
            logger.error("No valid DER records computed (all batches were skipped)")
            return float('nan'), float('nan'), pred_by_rec, lab_by_rec
        else:
            # DER might still be valid even if losses are all NaN
            der = np.mean(list(ders.values())) if ders else float('nan')
            return float('nan'), der, pred_by_rec, lab_by_rec
    
    # Print summary only if batch_losses is not empty
    print(f" Eval summary: min_loss={min(batch_losses):.4f}, max_loss={max(batch_losses):.4f}, mean_loss={np.mean(batch_losses):.4f}")
    # Average DER across recordings (only if computed)
    der = np.mean(list(ders.values())) if ders else float('nan')
    mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
    
    return mean_loss, der, pred_by_rec, lab_by_rec


def main():
    # Start timing
    start_time = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Centralized Learning for Overlapping Speech Diarization")
    parser.add_argument('--test_size', type=int, default=None, help='Number of dataset records to use for training (if not specified, use all available data)')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs for training (early stopping will stop earlier if no improvement)')
    parser.add_argument('--compute_der_during_training', action='store_true', help='Compute DER during training (slower but provides more metrics)')
    parser.add_argument('--early_stopping_patience', type=int, default=10, help='Number of epochs to wait before early stopping (default: 10, increased from 5 for more stable training)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum improvement required to reset patience counter')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of samples to process at once during dataset creation (smaller = less memory, default: 500)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (smaller = less memory, default: 4)')
    parser.add_argument('--max_sequence_length', type=int, default=1000, help='Maximum sequence length to truncate longer sequences (default: 1000). Typical SEND uses 100-2000 frames, not 16k+. Use 1000-1500 for stable training.')
    parser.add_argument('--max_memory_gb', type=float, default=64.0, help='Maximum memory to use in GB (default: 64.0). Auto-calculation is DISABLED - use --max_sequence_length directly.')
    # Model simplification parameters for speed testing
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for model (default: 512, use 256 for faster training)')
    parser.add_argument('--num_speech_encoder_layers', type=int, default=8, help='Number of speech encoder layers (default: 8, use 4 for faster training)')
    parser.add_argument('--num_post_net_layers', type=int, default=6, help='Number of post-net layers (default: 6, use 3 for faster training)')
    parser.add_argument('--num_transformer_layers', type=int, default=4, help='Number of transformer layers in CD scorer (default: 4, use 2 for faster training)')
    parser.add_argument('--enable_persistent_workers', action='store_true', help='Enable persistent_workers for faster data loading (disabled by default to avoid semaphore leaks)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4, reduced from 1e-3 for stability)')
    args = parser.parse_args()

    # Assign arguments to variables
    test_size = args.test_size
    epochs = args.epochs
    compute_der_during_training = args.compute_der_during_training
    early_stopping_patience = args.early_stopping_patience
    early_stopping_min_delta = args.early_stopping_min_delta
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    max_sequence_length = args.max_sequence_length
    max_memory_gb = args.max_memory_gb
    # Model simplification parameters
    hidden_dim = args.hidden_dim
    num_speech_encoder_layers = args.num_speech_encoder_layers
    num_post_net_layers = args.num_post_net_layers
    num_transformer_layers = args.num_transformer_layers
    enable_persistent_workers = args.enable_persistent_workers
    learning_rate = args.lr
    
    # Determine if we're using all data or a subset
    use_all_data = test_size is None

    logger.info("="*80)
    logger.info("MAIN STARTED")
    logger.info(f"MAIN: Starting main()")
    sys.stdout.flush()  # Ensure output is written immediately
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"MAIN: Using device: {device}")
        
        # Debug GPU information
        if torch.cuda.is_available():
            print(f"MAIN: CUDA available: True")
            print(f"MAIN: CUDA device count: {torch.cuda.device_count()}")
            print(f"MAIN: Current CUDA device: {torch.cuda.current_device()}")
            print(f"MAIN: CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"MAIN: CUDA available: False - Using CPU")
        
        # Initialize speaker encoder
        print(f"MAIN: Initializing speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": device}
        ).to(device)
        print(f"MAIN: Speaker encoder initialized successfully")
        
        # Load and preprocess data
        print_data_loading_info("AMI")
        dataset = load_dataset("edinburghcstr/ami", "ihm")
        print(f"MAIN: Dataset loaded successfully")
        
        # Determine dataset sizes
        if use_all_data:
            train_size = len(dataset["train"])
            val_size = len(dataset["validation"])
            test_size = len(dataset["test"])
            print(f"MAIN: Using ALL data - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        else:
            train_size = test_size
            val_size = round(test_size/0.7*0.3)
            test_size = round(test_size/0.7*0.3)
            print(f"MAIN: Using SUBSET - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        print_dataset_overview("AMI", len(dataset["train"]), train_size)
        
        # Group data by meeting ID for all splits
        print(f"MAIN: Grouping data by meeting ID...")
        grouped_train = group_by_meeting(dataset["train"].select(range(train_size)))
        grouped_validation = group_by_meeting(dataset["validation"].select(range(val_size)))
        grouped_test = group_by_meeting(dataset["test"].select(range(test_size)))
        
        print_grouping_results(grouped_train, grouped_validation, grouped_test)
        
        # Print statistics for each meeting
        print_meeting_statistics(grouped_train, grouped_validation, grouped_test)
        
        # PSE/SEND Configuration: Fixed N and K (as per original paper)
        N = 5  # Maximum number of target speakers per recording
        K = 4  # Maximum simultaneous overlap (2-4 as per paper)
        
        # Initialize Power Set Encoder with fixed N and K
        print(f"MAIN: Initializing Power Set Encoder with max_speakers={N}, max_overlap={K}")
        power_set_encoder = PowerSetEncoder(max_speakers=N, max_overlap=K)

        # Calculate number of classes using C(K,N) formula
        num_classes = power_set_encoder.num_classes
        print(f"MAIN: PSE Configuration: N={N} (max speakers per recording), K={K} (max overlap)")
        print(f"MAIN: Number of classes using C(K,N) = Σ(k=0 to {K}) C({N},k) = {num_classes}")
        
        # Print PowerSetEncoder examples and statistics
        print_power_set_encoder_examples(power_set_encoder)
        
        # FIXED: Use fixed max_sequence_length instead of dangerous auto-calculation
        # Auto-calculation was producing values like 16361, which is:
        # 1. Too large for 1080 Ti GPU to handle efficiently
        # 2. Not needed for SEND (typical T is hundreds to ~2000 frames, not 16k+)
        # 3. Causes memory issues and slow training
        # 
        # Default is now 1000 (can be overridden via --max_sequence_length)
        # For SEND, recommended range is 1000-1500 frames
        if max_sequence_length is None:
            # This should not happen with default=1000, but keep as safety fallback
            max_sequence_length = 1000
            print(f"MAIN: WARNING: max_sequence_length was None, using safe default: {max_sequence_length}")
        else:
            print(f"MAIN: Using max_sequence_length={max_sequence_length}")
            if max_sequence_length > 2000:
                print(f"MAIN: ⚠️  WARNING: max_sequence_length={max_sequence_length} is very large!")
                print(f"MAIN:    SEND typically uses 100-2000 frames. Consider using 1000-1500 for stable training.")
                print(f"MAIN:    Large sequences will be slow on 1080 Ti and may cause memory issues.")
        
        # Prepare data loaders for training and evaluation
        train_loader, val_loader, test_loader = prepare_data_loaders(
            grouped_train, grouped_validation, grouped_test, speaker_encoder, power_set_encoder, 
            batch_size=batch_size, N=N, chunk_size=chunk_size, max_sequence_length=max_sequence_length,
            enable_persistent_workers=enable_persistent_workers
        ) 
        
        # Print experiment configuration
        print_experiment_config(1, 1, epochs, train_size)  # Single centralized training
        print(f"MAIN: Early stopping patience: {early_stopping_patience} epochs")
        print(f"MAIN: Early stopping min delta: {early_stopping_min_delta}")

        
        # Get all unique speakers for speaker embedding computation
        # speaker_ids = set()
        # for grouped in [grouped_train, grouped_validation, grouped_test]:
        #     for samples in grouped.values():
        #         for sample in samples:
        #             speaker_ids.add(sample["speaker_id"])
        # all_speaker_ids = sorted(list(speaker_ids))
        # speaker_id_list = all_speaker_ids[:N]  # Limit to N slots for PSE consistency
        # print(f"MAIN: Detected {len(all_speaker_ids)} unique speakers in dataset: {all_speaker_ids}")
        # print(f"MAIN: Using first {len(speaker_id_list)} speakers for PSE slots: {speaker_id_list}")
        # print(f"MAIN: Note: PSE uses fixed N={N} slots per recording, not all {len(all_speaker_ids)} speakers")
        
        # Analyze speaker distribution
        analyze_speaker_distribution(grouped_train)
        
        # Create and train model with configurable architecture
        print(f"MAIN: Creating SEND model...")
        print(f"MAIN: Model architecture: hidden_dim={hidden_dim}, speech_encoder_layers={num_speech_encoder_layers}, post_net_layers={num_post_net_layers}, transformer_layers={num_transformer_layers}")
        model = SENDModel(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_speech_encoder_layers=num_speech_encoder_layers,
            num_post_net_layers=num_post_net_layers,
            num_transformer_layers=num_transformer_layers
        ).to(device)
        
        # Print SENDModel statistics
        print_send_model_statistics(model)
        
        # Use centralized training data loaders directly
        print(f"MAIN: Using centralized training data loaders...")
        
        # Calculate and display actual training samples information
        total_training_samples = len(train_loader.dataset)
        total_validation_samples = len(val_loader.dataset)
        total_test_samples = len(test_loader.dataset)
            
            # Calculate actual frames from the dataset
        total_training_frames = 0
        if total_training_samples > 0:
                # Get actual frame count from first sample
                first_sample = train_loader.dataset[0]
                if isinstance(first_sample, tuple) and len(first_sample) > 0:
                    feature = first_sample[0]  # First element should be features
                    if hasattr(feature, 'shape') and len(feature.shape) > 0:
                        frames_per_sample = feature.shape[0]
                        total_training_frames = total_training_samples * frames_per_sample
                    else:
                        total_training_frames = total_training_samples * 100  # Fallback estimate
                else:
                    total_training_frames = total_training_samples * 100  # Fallback estimate
        
        print(f"MAIN: Training samples: {total_training_samples}")
        print(f"MAIN: Validation samples: {total_validation_samples}")
        print(f"MAIN: Test samples: {total_test_samples}")
        print(f"MAIN: Total training frames: {total_training_frames}")
        
        # Additional information about data distribution
        if total_training_samples > 0:
            avg_frames_per_sample = total_training_frames / total_training_samples
            print(f"MAIN: Average frames per sample: {avg_frames_per_sample:.1f}")
            if use_all_data:
                print(f"MAIN: Using ALL available data from AMI dataset")
            else:
                print(f"MAIN: Note: test_size={train_size} refers to number of dataset records selected for training")
            print(f"MAIN: Each meeting recording contains multiple audio segments, each segment becomes multiple training samples")
            print(f"MAIN: Each training sample contains multiple frames (time steps) for sequence learning")
        
        # Create experiment directories early
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dt_str_human = datetime.now().strftime("%Y-%m-%d-%H-%M")
        if use_all_data:
            exp_tag = f"exp_all_{epochs}epochs_centralized_{dt_str_human}"
        else:
            exp_tag = f"exp_{train_size}size_{epochs}epochs_centralized_{dt_str_human}"
        
        # Artifact directories for logs and plots
        artifact_logs_dir = os.path.join("experiments", "centralized_out_artifacts", "logs", exp_tag)
        artifact_plots_dir = os.path.join("experiments", "centralized_out_artifacts", "plots", exp_tag)
        os.makedirs(artifact_logs_dir, exist_ok=True)
        os.makedirs(artifact_plots_dir, exist_ok=True)
        
        print(f"MAIN: Experiment tag: {exp_tag}")
        print(f"MAIN: Logs directory: {artifact_logs_dir}")
        print(f"MAIN: Plots directory: {artifact_plots_dir}")
        
        # Compute speaker embeddings for train set
        print(f"MAIN: Computing speaker embeddings for train set...")
        speaker_to_embedding = compute_speaker_embeddings(grouped_train, speaker_encoder)
        
        # === CENTRALIZED TRAINING ===
        print("\n==================== STARTING CENTRALIZED TRAINING ====================\n")
        
        # Train the model
        print(f"MAIN: DER computation during training: {'ENABLED' if compute_der_during_training else 'DISABLED (faster training)'}")
        print(f"MAIN: DER computation during validation: ALWAYS ENABLED (required for early stopping based on DER)")
        print(f"MAIN: Early stopping: Based on VAL DER (lower is better), patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        
        # Create progress log file path
        progress_log_file = os.path.join(artifact_logs_dir, "training_progress.txt")
        print(f"MAIN: Progress will be logged to: {progress_log_file}")
        
        training_metrics = train_model(model, train_loader, val_loader, device, power_set_encoder, epochs, compute_der_during_training, progress_log_file, early_stopping_patience, early_stopping_min_delta, learning_rate=learning_rate)
        
        # Evaluate on validation set
        print(f"MAIN: Evaluating on validation set...")
        val_loss, val_der, val_pred_by_rec, val_lab_by_rec = evaluate_model(model, val_loader, device, power_set_encoder)
        
        print(f"MAIN: Training completed successfully")
        print(f"MAIN: Final validation loss: {val_loss:.4f}")
        print(f"MAIN: Final validation DER: {val_der:.4f}")
        
        # Store training metrics for plotting
        epoch_metrics = training_metrics
        
        # Final evaluation on test set with trained model
        print("\n==================== TESTING STARTED (TRAINED MODEL) ====================\n")
        model.eval()
        test_loss = 0.0
        # Group predictions by meeting_id for proper DER calculation
        pred_by_rec = defaultdict(list)
        lab_by_rec = defaultdict(list)
        
        # Debug: Check test dataset
        print(f"[FINAL TEST DEBUG] Test dataset size: {len(test_loader.dataset)}")
        print(f"[FINAL TEST DEBUG] Test dataset speaker_id_list: {test_loader.dataset.get_speaker_id_list()}")
        
        # Debug: Check model state before testing
        model_params_before = [p.clone() for p in model.parameters()]
        print(f"[FINAL TEST DEBUG] Model has {len(model_params_before)} parameters")
        
        with torch.no_grad():
            for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(test_loader):
                print(f"[FINAL TEST DEBUG] Processing batch {batch_idx}")
                features, speaker_embeddings, labels = (
                    features.to(device),
                    speaker_embeddings.to(device),
                    labels.to(device)
                )
                outputs = model(features, speaker_embeddings)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                
                # Check for valid frames: skip batch if all labels are padding (-100)
                valid = (labels != -100)
                if valid.sum() == 0:
                    logger.warning(f"Test batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                    continue
                
                loss = nn.CrossEntropyLoss(ignore_index=-100)(outputs, labels)
                test_loss += loss.item()
                predictions = torch.argmax(outputs, dim=-1)
                
                # Debug: Check predictions
                unique_preds = torch.unique(predictions).cpu().numpy()
                unique_labels = torch.unique(labels).cpu().numpy()
                print(f"[FINAL TEST DEBUG] Batch {batch_idx}: unique predictions: {unique_preds}")
                print(f"[FINAL TEST DEBUG] Batch {batch_idx}: unique labels: {unique_labels}")
                print(f"[FINAL TEST DEBUG] Batch {batch_idx}: loss: {loss.item():.4f}")
                
                # Debug: Check if predictions are deterministic
                if batch_idx == 0:
                    first_pred = predictions[0].item()
                    print(f"[FINAL TEST DEBUG] First prediction: {first_pred}")
                
                # Group predictions by meeting_id
                predictions_np = predictions.cpu().numpy()
                labels_np = labels.cpu().numpy()
                # meeting_ids: List[np.ndarray] (each with length = max_len of batch)
                meeting_ids_flat = np.concatenate(meeting_ids, axis=0)  # => shape: [batch_size*seq_len]
                
                for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                    if meeting_id is not None:  # Skip padded frames
                        pred_by_rec[meeting_id].append(pred)
                        lab_by_rec[meeting_id].append(label)
        
        # Calculate DER per recording and aggregate
        ders = {}
        # Get speaker_id_list from test dataset for consistency
        test_speaker_id_list = test_loader.dataset.get_speaker_id_list() if hasattr(test_loader.dataset, 'get_speaker_id_list') else None
        print(f"[FINAL TEST] Using speaker_id_list from test dataset: {test_speaker_id_list}")
        
        # Debug: Check predictions distribution
        for rec_id in pred_by_rec:
            if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                unique_preds = np.unique(pred_by_rec[rec_id])
                unique_labels = np.unique(lab_by_rec[rec_id])
                print(f"[FINAL TEST DEBUG] Recording {rec_id}:")
                print(f"  - Unique predictions: {unique_preds}")
                print(f"  - Unique labels: {unique_labels}")
                print(f"  - Prediction distribution: {np.bincount(pred_by_rec[rec_id])}")
                print(f"  - Label distribution: {np.bincount(lab_by_rec[rec_id])}")
        
        for rec_id in pred_by_rec:
            if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                ders[rec_id] = calculate_der(
                    pred_by_rec[rec_id],
                    lab_by_rec[rec_id],
                    power_set_encoder,
                    speaker_id_list=test_speaker_id_list,
                    debug=True,
                    frame_shift=0.01,
                    uri=rec_id
                )
        
        # Calculate final metrics
        test_loss = test_loss / len(test_loader)
        der = np.mean(list(ders.values())) if ders else float('nan')
        
        print(f"Test recordings processed: {len(ders)}")
        print(f"DER per recording: {ders}")
        print(f"Average DER: {der}")
        print(f"\n==================== TESTING FINISHED ====================\n")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final Test DER: {der:.4f}")
        print(f"Final Validation Loss: {val_loss:.4f}")
        print(f"Final Validation DER: {val_der:.4f}")
        
        # Calculate and display total execution time
        total_time = time.time() - start_time
        total_minutes = total_time / 60
        total_hours = total_minutes / 60
        
        if total_hours >= 1:
            print(f"Total Execution Time: {total_time:.2f} seconds ({total_hours:.2f} hours)")
        elif total_minutes >= 1:
            print(f"Total Execution Time: {total_time:.2f} seconds ({total_minutes:.2f} minutes)")
        else:
            print(f"Total Execution Time: {total_time:.2f} seconds")

        # === EXPORT DIARIZATION RESULTS TO RTTM FORMAT ===
        print("\n===== EXPORTING DIARIZATION RESULTS =====")
        
        # Create experiment tag for export directory (already created above)
        # dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # dt_str_human = datetime.now().strftime("%Y-%m-%d-%H-%M")
        # exp_tag = f"exp_{test_size}size_{epochs}epochs_centralized_{dt_str_human}"
        
        try:
            from diarization_export import export_diarization_results
            
            # Create dictionary with speaker_id_lists for each recording
            speaker_id_lists = {}
            for rec_id in pred_by_rec:
                if rec_id in test_loader.dataset.speaker_id_list:
                    speaker_id_lists[rec_id] = test_loader.dataset.speaker_id_list[rec_id]
                else:
                    # Use general speaker list
                    speaker_id_lists[rec_id] = test_speaker_id_list
            
            # Export results to various formats
            # Create diarization export directory with experiment parameters
            diarization_export_dir = os.path.join("experiments", "centralized_out_artifacts", "diarization_export", exp_tag)
            export_results = export_diarization_results(
                predictions_by_recording=pred_by_rec,
                power_set_encoder=power_set_encoder,
                output_dir=diarization_export_dir,
                speaker_id_lists=speaker_id_lists,
                ground_truth_by_recording=lab_by_rec,
                formats=['rttm', 'ctm', 'summary', 'metrics']
            )
            
            print("Diarization results export completed:")
            for format_name, files in export_results.items():
                print(f"  {format_name.upper()}: {len(files)} files")
                for file_path in files:
                    print(f"    - {file_path}")
                    
        except ImportError as e:
            print(f"[WARNING] Could not import export module: {e}")
            print("Diarization results not exported to RTTM format")
        except Exception as e:
            print(f"[ERROR] Error exporting diarization results: {e}")
            print("Diarization results not exported to RTTM format")

        # === LOG FINAL RESULTS TO FILE ===
        # Use actual experiment parameters, not hardcoded values
        # Artifact directories already created above
        # artifact_logs_dir = os.path.join("experiments", "centralized_out_artifacts", "logs", exp_tag)
        # artifact_plots_dir = os.path.join("experiments", "centralized_out_artifacts", "plots", exp_tag)
        # os.makedirs(artifact_logs_dir, exist_ok=True)
        # os.makedirs(artifact_plots_dir, exist_ok=True)
        # File paths for logs and metrics (simple names)
        exp_filename = "experiment.txt"
        exp_filepath = os.path.join(artifact_logs_dir, exp_filename)
        # Prepare lines for logging
        result_lines = [
            f"Experiment: {exp_tag}",
            f"Dataset size: {'ALL' if use_all_data else f'{train_size} records'}",
            f"Train records: {train_size}",
            f"Validation records: {val_size}",
            f"Test records: {test_size}",
            f"Max epochs: {epochs}",
            f"Actual epochs trained: {len(epoch_metrics)}",
            f"Early stopping patience: {early_stopping_patience}",
            f"Early stopping min delta: {early_stopping_min_delta}",
            f"Early stopping triggered: {'Yes' if len(epoch_metrics) < epochs else 'No'}",
            f"Training type: Centralized",
            f"Datetime: {dt_str}",
            f"",
            f"=== TRAINING DATA STATISTICS ===",
            f"Total training samples: {total_training_samples}",
            f"Total validation samples: {total_validation_samples}",
            f"Total test samples: {total_test_samples}",
            f"Total training frames: {total_training_frames}",
            f"",
        ]
        
        
        result_lines.extend([
            f"=== FINAL RESULTS ===",
            f"Final Test Loss: {test_loss:.4f}",
            f"Final Test DER: {der:.4f}",
            f"Final Validation Loss: {val_loss:.4f}",
            f"Final Validation DER: {val_der:.4f}",
            f"Best Validation Loss: {min([m.get('val_loss') for m in epoch_metrics if m.get('val_loss') is not None], default='N/A') if epoch_metrics else 'N/A'}",
            f"Model Status: {'TRAINED with centralized learning' if epoch_metrics else 'NOT TRAINED'}",
            f"Training Epochs: {len(epoch_metrics) if epoch_metrics else 0}",
            f"Total Execution Time: {time.time() - start_time:.2f} seconds ({((time.time() - start_time)/60):.2f} minutes)",
        ])
        
        # Print final results using statistics module
        print_final_results(der, der, time.time() - start_time)
        # Save to file and print to console (only artifact directory)
        print("\n===== SAVING FINAL RESULTS TO FILE =====")
        print(f"Results will be saved to: {exp_filepath}")
        for line in result_lines:
            print(line)
        try:
            with open(exp_filepath, "w") as f:
                for line in result_lines:
                    f.write(line + "\n")
            print(f"Results saved to {exp_filepath}")
        except Exception as e:
            print(f"[ERROR] Could not save results to file: {e}")

        # After training and final evaluation, plot and save metrics
        def plot_training_metrics(epoch_metrics):
            """Plot training metrics for centralized learning."""
            if not epoch_metrics:
                print("No training metrics to plot")
                return
                
            epochs = list(range(1, len(epoch_metrics) + 1))
            train_losses = [m.get('train_loss') for m in epoch_metrics]
            train_ders = [m.get('der') for m in epoch_metrics]
            train_accs = [m.get('acc') for m in epoch_metrics]
            
            # Plot training loss per epoch
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_losses, marker='o', label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss per Epoch')
            plt.tight_layout()
            plot_path = os.path.join(artifact_plots_dir, "training_loss_per_epoch.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Plot training DER per epoch
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_ders, marker='o', label='Training DER')
            plt.xlabel('Epoch')
            plt.ylabel('DER')
            plt.title('Training DER per Epoch')
            plt.tight_layout()
            plot_path = os.path.join(artifact_plots_dir, "training_der_per_epoch.png")
            plt.savefig(plot_path)
            plt.close()
            
            # Plot training accuracy per epoch
            plt.figure(figsize=(8, 5))
            plt.plot(epochs, train_accs, marker='o', label='Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy per Epoch')
            plt.tight_layout()
            plot_path = os.path.join(artifact_plots_dir, "training_accuracy_per_epoch.png")
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Training metrics plots saved to {artifact_plots_dir}")

        # Call plotting after experiment
        plot_training_metrics(epoch_metrics)

        # === SAVE METRICS TO CSV FILES ===
        def save_metrics_to_csv(epoch_metrics, exp_tag, artifact_logs_dir):
            """Save all metrics to CSV files for detailed analysis."""
            print("\n===== SAVING METRICS TO CSV FILES =====")
            
            # 1. Detailed metrics per epoch
            detailed_metrics = []
            for epoch_idx, epoch_metrics_dict in enumerate(epoch_metrics):
                        detailed_metrics.append({
                            'epoch': epoch_idx + 1,
                    'train_loss': epoch_metrics_dict.get('train_loss'),
                    'der': epoch_metrics_dict.get('der'),
                    'acc': epoch_metrics_dict.get('acc')
                        })
            
            if detailed_metrics:
                detailed_df = pd.DataFrame(detailed_metrics)
                detailed_csv_path = os.path.join(artifact_logs_dir, "detailed_metrics.csv")
                detailed_df.to_csv(detailed_csv_path, index=False)
                print(f"Detailed metrics saved to: {detailed_csv_path}")
                print(f"Shape: {detailed_df.shape}")
                print(f"Columns: {list(detailed_df.columns)}")
            
            # 2. Final experiment results
            final_results = [{
                'experiment_tag': exp_tag,
            'dataset_size': 'ALL' if use_all_data else f'{train_size}',
            'train_records': train_size,
            'val_records': val_size,
            'test_records': test_size,
                'num_epochs': epochs,
            'actual_epochs_trained': len(epoch_metrics),
            'early_stopping_triggered': len(epoch_metrics) < epochs,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'final_test_loss': test_loss,
                'final_der': der,
            'final_val_loss': val_loss,
            'final_val_der': val_der,
            'total_execution_time_seconds': total_time,
            'total_execution_time_minutes': total_minutes,
                'device_used': str(device),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }]
            
            final_results_df = pd.DataFrame(final_results)
            final_results_csv_path = os.path.join(artifact_logs_dir, "experiment_results.csv")
            final_results_df.to_csv(final_results_csv_path, index=False)
            print(f"Final results saved to: {final_results_csv_path}")
            
            print("All CSV files saved successfully!")
            return detailed_df if detailed_metrics else None

        # Save metrics to CSV
        detailed_df = save_metrics_to_csv(epoch_metrics, exp_tag, artifact_logs_dir)

        # === ADDITIONAL ANALYSIS CSV FILES ===
        def create_epoch_progress_csv(epoch_metrics, exp_tag, artifact_logs_dir):
            """Create CSV file showing training progress across epochs."""
            print("\n===== CREATING EPOCH PROGRESS ANALYSIS =====")
            
            epoch_progress = []
            for epoch_idx, epoch_metrics_dict in enumerate(epoch_metrics):
                        epoch_progress.append({
                            'epoch': epoch_idx + 1,
                    'train_loss': epoch_metrics_dict.get('train_loss'),
                    'der': epoch_metrics_dict.get('der'),
                    'acc': epoch_metrics_dict.get('acc'),
                    'epoch_label': f"E{epoch_idx + 1}"
                        })
            
            if epoch_progress:
                epoch_progress_df = pd.DataFrame(epoch_progress)
                
                # Sort by epoch for better readability
                epoch_progress_df = epoch_progress_df.sort_values(['epoch'])
                
                epoch_progress_csv_path = os.path.join(artifact_logs_dir, "epoch_progress.csv")
                epoch_progress_df.to_csv(epoch_progress_csv_path, index=False)
                print(f"Epoch progress analysis saved to: {epoch_progress_csv_path}")
                print(f"Shape: {epoch_progress_df.shape}")
                
                return epoch_progress_df
            return None

        # Create epoch progress analysis
        epoch_progress_df = create_epoch_progress_csv(epoch_metrics, exp_tag, artifact_logs_dir)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        # Cleanup will happen in finally block
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # CRITICAL: Close DataLoader workers to prevent semaphore leaks
        # This is especially important when persistent_workers=True
        # PyTorch DataLoader with persistent_workers=True requires explicit shutdown
        print("Cleaning up DataLoader workers...")
        try:
            # Method 1: Explicitly delete iterators to trigger worker shutdown
            if 'train_loader' in locals() and train_loader is not None:
                try:
                    # Force iterator cleanup
                    if hasattr(train_loader, '_iterator'):
                        del train_loader._iterator
                    # Shutdown workers directly
                    if hasattr(train_loader, '_workers') and train_loader._workers:
                        for worker in train_loader._workers:
                            if hasattr(worker, 'terminate'):
                                worker.terminate()
                        for worker in train_loader._workers:
                            if hasattr(worker, 'join'):
                                worker.join(timeout=2.0)
                except Exception as e:
                    logger.debug(f"Could not close train_loader workers: {e}")
            
            if 'val_loader' in locals() and val_loader is not None:
                try:
                    if hasattr(val_loader, '_iterator'):
                        del val_loader._iterator
                    if hasattr(val_loader, '_workers') and val_loader._workers:
                        for worker in val_loader._workers:
                            if hasattr(worker, 'terminate'):
                                worker.terminate()
                        for worker in val_loader._workers:
                            if hasattr(worker, 'join'):
                                worker.join(timeout=2.0)
                except Exception as e:
                    logger.debug(f"Could not close val_loader workers: {e}")
            
            if 'test_loader' in locals() and test_loader is not None:
                try:
                    if hasattr(test_loader, '_iterator'):
                        del test_loader._iterator
                    if hasattr(test_loader, '_workers') and test_loader._workers:
                        for worker in test_loader._workers:
                            if hasattr(worker, 'terminate'):
                                worker.terminate()
                        for worker in test_loader._workers:
                            if hasattr(worker, 'join'):
                                worker.join(timeout=2.0)
                except Exception as e:
                    logger.debug(f"Could not close test_loader workers: {e}")
            
            # Force garbage collection to ensure cleanup
            import gc
            gc.collect()
            
            print("DataLoader workers cleanup completed.")
        except Exception as cleanup_error:
            logger.warning(f"Error during DataLoader cleanup: {cleanup_error}")
        print("Process completed.")

if __name__ == "__main__":
    main() 