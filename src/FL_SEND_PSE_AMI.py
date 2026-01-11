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
import gc
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
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context, Metrics
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
    extract_features, 
    simulate_overlapping_speech, 
    group_by_meeting,
    prepare_data_loaders,
    power_set_encoding,
    calculate_der,
    compute_speaker_embeddings,
    OverlappingSpeechDataset,
    build_train_dataset_for_grouped_meetings,
    build_eval_dataset
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
import copy


# ============================================================================
# FL INSTRUMENTATION FUNCTIONS
# ============================================================================

def model_checksum(model: nn.Module) -> float:
    """Compute checksum of model parameters (sum of absolute values).
    
    Returns:
        float: Sum of absolute values of all parameters.
    """
    return sum(p.detach().float().abs().sum().item() for p in model.parameters())


def state_dict_checksum(state_dict: dict) -> float:
    """Compute checksum of state dict (sum of absolute values).
    
    Args:
        state_dict: Model state dictionary.
    
    Returns:
        float: Sum of absolute values of all tensor values.
    """
    return sum(v.detach().float().abs().sum().item() 
               for v in state_dict.values() 
               if torch.is_tensor(v))


def state_dict_delta(a: dict, b: dict) -> float:
    """Compute mean absolute difference between two state dicts.
    
    Args:
        a: First state dictionary.
        b: Second state dictionary.
    
    Returns:
        float: Mean absolute difference across common keys.
    """
    common_keys = set(a.keys()) & set(b.keys())
    if not common_keys:
        return float('inf')
    
    deltas = []
    for key in common_keys:
        if torch.is_tensor(a[key]) and torch.is_tensor(b[key]):
            delta = (a[key] - b[key]).abs().mean().item()
            deltas.append(delta)
    
    return sum(deltas) / len(deltas) if deltas else 0.0


def compute_l2_norm_diff(state_dict_a: dict, state_dict_b: dict) -> float:
    """Compute L2 norm difference between two state dicts.
    
    Args:
        state_dict_a: First state dictionary.
        state_dict_b: Second state dictionary.
    
    Returns:
        float: L2 norm of the difference (sqrt of sum of squared differences).
    """
    common_keys = set(state_dict_a.keys()) & set(state_dict_b.keys())
    if not common_keys:
        return float('inf')
    
    total_squared_diff = 0.0
    for key in common_keys:
        if torch.is_tensor(state_dict_a[key]) and torch.is_tensor(state_dict_b[key]):
            diff = (state_dict_a[key] - state_dict_b[key]).flatten()
            total_squared_diff += (diff ** 2).sum().item()
    
    return np.sqrt(total_squared_diff)


def compute_grad_norm(model: nn.Module) -> float:
    """Compute L2 norm of all gradients in the model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        float: L2 norm of all gradients (0.0 if no gradients).
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
    return np.sqrt(total_norm) if total_norm > 0 else 0.0


def verify_state_dict_deep_copy(state_dict: dict) -> bool:
    """Verify that state_dict values are deep copies (not shared references).
    
    Args:
        state_dict: State dictionary to check.
    
    Returns:
        bool: True if values appear to be deep copies (different memory locations).
    """
    # Check if values are tensors and have unique memory addresses
    seen_addresses = set()
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            # Get memory address (id of underlying storage)
            addr = id(value.storage()) if hasattr(value, 'storage') else id(value)
            if addr in seen_addresses:
                return False  # Shared reference found
            seen_addresses.add(addr)
    return True


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


class SENDClient(NumPyClient):
    """Federated Learning client for SEND model.
    
    Memory-optimized: Does NOT store DataLoaders or large datasets in actor state.
    Creates DataLoaders lazily inside fit()/evaluate() and cleans up after.
    """
    def __init__(
        self,
        model: SENDModel,
        train_dataset,  # Dataset object, not DataLoader
        val_dataset,    # Dataset object, not DataLoader
        device: torch.device,
        power_set_encoder: PowerSetEncoder,
        speaker_encoder: EncoderClassifier,
        batch_size: int,
        client_id: int = 0
    ):
        print(f"SENDClient: Initializing client {client_id} (memory-optimized)")
        self.model = model
        # C1: Store only datasets (not DataLoaders) - DataLoaders created lazily
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.power_set_encoder = power_set_encoder
        self.speaker_encoder = speaker_encoder
        self.batch_size = batch_size
        self.client_id = client_id
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding labels (-100) - CRITICAL FIX
        print(f"SENDClient: Initialization complete for client {client_id}")
        print(f"[DEBUG] SENDClient: train_dataset size: {len(self.train_dataset) if self.train_dataset else 0}")
        print(f"[DEBUG] SENDClient: val_dataset size: {len(self.val_dataset) if self.val_dataset else 0}")
    
    def get_parameters(self, config):
        # ROBUSTNESS FIX: Ensure deep copy of parameters (no shared references)
        # Clone and detach to ensure no shared tensor storage
        state_dict = self.model.state_dict()
        params = []
        for key, val in state_dict.items():
            # Create a deep copy: clone() creates new tensor, detach() removes from computation graph
            param_copy = val.clone().detach().cpu().numpy()
            params.append(param_copy)
        return params
    
    def set_parameters(self, parameters):
        """Set model parameters from Flower parameters.
        
        B2 Fix: Ensure device placement happens after load_state_dict.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        # Ensure model is on correct device
        self.model.to(self.device)
    
    def _create_train_loader(self):
        """Create train DataLoader with memory-optimized settings for Ray."""
        from data_processing import collate_fn_overlapping_speech
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_overlapping_speech,
            num_workers=0,  # D: Disable multiprocessing in Ray
            pin_memory=False,  # D: Disable pin_memory for Ray workers
            persistent_workers=False,  # D: Disable persistent workers
        )
    
    def _create_val_loader(self):
        """Create validation DataLoader with memory-optimized settings for Ray."""
        from data_processing import collate_fn_overlapping_speech
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn_overlapping_speech,
            num_workers=0,  # D: Disable multiprocessing in Ray
            pin_memory=False,  # D: Disable pin_memory for Ray workers
            persistent_workers=False,  # D: Disable persistent workers
        )
    
    def fit(self, parameters, config):
        """Fit model on client data.
        
        C2: Creates DataLoader lazily inside fit() and cleans up after.
        C3: Returns only small metrics, no large objects.
        E: DER computation is optional (only when enabled or every N rounds).
        F: Wrapped in try/except for error handling.
        """
        # F: Error handling - return previous params on failure
        try:
            print("=== CLIENT LOG: fit started ===")
            print(f"SENDClient: Starting fit for client {self.client_id}")
            
            # A2: Log checksum BEFORE loading parameters
            checksum_before = model_checksum(self.model)
            print(f"CLIENT {self.client_id} start checksum (before set_parameters): {checksum_before:.6f}")
            
            self.set_parameters(parameters)
            
            # A2: Log checksum AFTER loading parameters
            checksum_after_load = model_checksum(self.model)
            print(f"CLIENT {self.client_id} start checksum (after set_parameters): {checksum_after_load:.6f}")
            
            # DEBUG: Store initial state dict for L2 norm comparison
            initial_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
            initial_l2_norm = sum(p.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
            print(f"🔍 CLIENT {self.client_id}: Initial L2 norm of parameters: {initial_l2_norm:.6f}")
            
            # DEBUG: Verify model is in training mode
            self.model.train()
            if not self.model.training:
                logger.error(f"❌ CLIENT {self.client_id}: CRITICAL - model.train() did not set training mode!")
            print(f"✅ CLIENT {self.client_id}: Model training mode: {self.model.training}")
            
            # DEBUG: Update optimizer AFTER loading global weights (critical for FL)
            # Update learning rate without recreating optimizer to preserve state (momentum, Adam running averages)
            # Note: load_state_dict() updates parameter values but keeps the same parameter objects,
            # so the optimizer (which holds references to these objects) still works correctly.
            new_lr = config.get("lr", 1e-4)
            if len(self.optimizer.param_groups) > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"✅ CLIENT {self.client_id}: Optimizer learning rate updated to {new_lr} after loading global weights (state preserved)")
            else:
                # Fallback: recreate optimizer if param_groups is empty (shouldn't happen, but safety check)
                logger.warning(f"⚠️ CLIENT {self.client_id}: Optimizer param_groups is empty, recreating optimizer")
                self.optimizer = optim.Adam(self.model.parameters(), lr=new_lr)
                print(f"✅ CLIENT {self.client_id}: Optimizer recreated with LR={new_lr}")
            
            epochs = config.get("epochs", 1)
            debug_mode = config.get("debug_mode", False)
            debug_max_batches = config.get("debug_max_batches", 200)
            server_round = config.get("server_round", 0)
            compute_der = config.get("compute_der", False)  # E: DER computation optional
            der_round_interval = config.get("der_round_interval", 5)  # E: Compute DER every N rounds
            
            # E: Only compute DER if enabled or every N rounds
            should_compute_der = compute_der or (server_round % der_round_interval == 0)
            
            # C2: Create DataLoader lazily inside fit()
            train_loader = self._create_train_loader()
            num_examples = len(train_loader.dataset)
            num_batches = len(train_loader)
            
            # ROBUSTNESS FIX: Check for empty dataset/loader before training
            print(f"[DEBUG] fit: train_dataset size: {len(self.train_dataset) if self.train_dataset else 0}")
            print(f"[DEBUG] fit: train_loader size: {num_batches} batches")
            print(f"[DEBUG] fit: num_examples: {num_examples}")
            
            if num_examples == 0 or num_batches == 0:
                logger.warning(f"⚠️ CLIENT {self.client_id}: Empty training dataset! num_examples={num_examples}, num_batches={num_batches}")
                logger.warning(f"   Returning safe defaults: loss=nan, num_examples=0")
                return self.get_parameters({}), 0, {"train_loss": float('nan'), "epoch_metrics": json.dumps([]), "empty_dataset": True}
            
            # Enable Mixed Precision Training for faster GPU computation (2-3x speedup)
            # MEMORY FIX: Only use AMP if device is CUDA (not CPU)
            use_amp = self.device.type == 'cuda' and torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
            if use_amp:
                print(f"SENDClient: Mixed Precision Training (AMP) ENABLED for client {self.client_id} (CUDA mode)")
            else:
                print(f"SENDClient: Mixed Precision Training (AMP) DISABLED - Using {self.device.type.upper()} mode for client {self.client_id}")
            
            start_time = time.time()
            epoch_metrics = []  # Collect metrics for each epoch - MUST be initialized before any return
            
            for epoch in range(epochs):
                train_loss = 0.0
                batch_losses = []
                # E: Only accumulate predictions if DER is needed (memory optimization)
                pred_by_rec = defaultdict(list) if should_compute_der else None
                lab_by_rec = defaultdict(list) if should_compute_der else None
                
                # Debug mode: limit batches
                max_batches = debug_max_batches if debug_mode else len(train_loader)
                if debug_mode:
                    print(f"[DEBUG MODE] Limiting training to {max_batches} batches per epoch")
                
                for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(train_loader):
                    if debug_mode and batch_idx >= max_batches:
                        print(f"[DEBUG MODE] Reached max_batches={max_batches}, stopping epoch early")
                        break
                        
                    if batch_idx == 0:
                        print(f"SENDClient: First batch in fit for client {self.client_id} (epoch {epoch+1}/{epochs})")
                    features, speaker_embeddings, labels = features.to(self.device, non_blocking=True), speaker_embeddings.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                    speaker_embeddings = speaker_embeddings.float()
                    
                    # Check input tensors for NaN/Inf before forward pass
                    if torch.isnan(features).any() or torch.isinf(features).any():
                        logger.warning(f"Batch {batch_idx}: NaN/Inf detected in features. Skipping batch.")
                        logger.warning(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}, has_nan={torch.isnan(features).any().item()}, has_inf={torch.isinf(features).any().item()}")
                        continue
                    if torch.isnan(speaker_embeddings).any() or torch.isinf(speaker_embeddings).any():
                        logger.warning(f"Batch {batch_idx}: NaN/Inf detected in speaker_embeddings. Skipping batch.")
                        logger.warning(f"  Speaker embeddings stats: min={speaker_embeddings.min().item():.4f}, max={speaker_embeddings.max().item():.4f}, mean={speaker_embeddings.mean().item():.4f}")
                        continue
                    
                    self.optimizer.zero_grad()
                    
                    # DEBUG: Verify model is still in training mode (no no_grad() context)
                    if not self.model.training:
                        logger.error(f"❌ CLIENT {self.client_id} Batch {batch_idx}: CRITICAL - Model not in training mode!")
                    
                    # Forward pass with Mixed Precision if available
                    # CRITICAL: No torch.no_grad() here - we need gradients!
                    if use_amp:
                        with torch.cuda.amp.autocast():  # autocast does NOT disable gradients
                            outputs = self.model(features, speaker_embeddings)
                            batch_size, seq_len, num_classes = outputs.shape
                            outputs = outputs.reshape(-1, num_classes)
                            labels_flat = labels.reshape(-1)
                            
                            # DEBUG: Log class distribution for first few batches
                            if batch_idx < 3 or (batch_idx % 50 == 0):
                                unique_labels, counts = torch.unique(labels_flat, return_counts=True)
                                label_dist = {int(l): int(c) for l, c in zip(unique_labels.cpu().numpy(), counts.cpu().numpy())}
                                # Check for class imbalance (empty set prediction)
                                empty_set_class = 0  # Assuming class 0 is empty set
                                empty_set_count = label_dist.get(empty_set_class, 0)
                                total_count = sum(label_dist.values())
                                empty_set_ratio = empty_set_count / total_count if total_count > 0 else 0.0
                                print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Label distribution: {label_dist}")
                                print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Empty set (class 0) ratio: {empty_set_ratio:.2%}")
                                if empty_set_ratio > 0.9:
                                    logger.warning(f"⚠️ CLIENT {self.client_id} Batch {batch_idx}: WARNING - >90% empty set labels! Model may predict only empty set.")
                            
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
                            
                            loss = self.criterion(outputs, labels_flat)
                            
                            # DEBUG: Log per-batch loss
                            if batch_idx < 5 or (batch_idx % 10 == 0):
                                print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Loss = {loss.item():.6f}")
                            
                            # Check for NaN/Inf in loss or outputs (early detection)
                            if not torch.isfinite(loss):
                                logger.warning(f"Batch {batch_idx}: NaN/Inf loss detected! Loss: {loss.item()}")
                                logger.warning(f"  Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                                logger.warning(f"  Logits stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                                logger.warning(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
                                logger.warning(f"  Labels stats: min={labels_flat.min().item()}, max={labels_flat.max().item()}, unique={torch.unique(labels_flat).cpu().numpy()[:10]}")
                                # Get label distribution for debugging
                                unique_labels_tensor, counts = torch.unique(labels_flat, return_counts=True)
                                label_dist = {int(l): int(c) for l, c in zip(unique_labels_tensor.cpu().numpy(), counts.cpu().numpy())}
                                logger.warning(f"  Label distribution: {label_dist}")
                            # B4: Skip batch completely - do NOT call scaler.update() without scaler.step()
                            # scaler.update() requires scaler.step() to be called first (records inf checks)
                            self.optimizer.zero_grad()
                            continue
                        
                        # Backward pass
                        # DEBUG: Verify backward() is called (no no_grad() context)
                        scaler.scale(loss).backward()
                        
                        # DEBUG: Compute and log gradient norm
                        scaler.unscale_(self.optimizer)
                        grad_norm_before_clip = compute_grad_norm(self.model)
                        
                        # DEBUG: Log gradient norm for first few batches
                        if batch_idx < 5 or (batch_idx % 10 == 0):
                            print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Gradient norm (before clip) = {grad_norm_before_clip:.6f}")
                            if grad_norm_before_clip < 1e-8:
                                logger.warning(f"⚠️ CLIENT {self.client_id} Batch {batch_idx}: WARNING - Gradient norm is near zero! Gradients may not be flowing.")
                        
                        # Gradient clipping to prevent gradient explosion (fixes NaN loss issue)
                        # B4: Check gradients for inf/NaN before stepping (AMP safety)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
                        if torch.isfinite(grad_norm):
                            # Clip gradients to max_norm
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            grad_norm_after_clip = compute_grad_norm(self.model)
                            
                            # DEBUG: Log gradient norm after clipping
                            if batch_idx < 5 or (batch_idx % 10 == 0):
                                print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Gradient norm (after clip) = {grad_norm_after_clip:.6f}")
                            
                            # B4: AMP safety - scaler.step() internally handles inf/NaN
                            # If scaler.step() skips due to inf/NaN, scaler.update() will handle it correctly
                            # But we already checked grad_norm, so step should succeed
                            scaler.step(self.optimizer)
                            scaler.update()
                            
                            # DEBUG: Verify optimizer step was called
                            if batch_idx < 5:
                                print(f"✅ CLIENT {self.client_id} Batch {batch_idx}: optimizer.step() called successfully")
                        else:
                            # Gradients contain inf/NaN, skip step and update
                            logger.warning(f"Batch {batch_idx}: Gradients contain inf/NaN (norm={grad_norm}), skipping optimizer step and scaler.update()")
                            self.optimizer.zero_grad()
                else:
                    outputs = self.model(features, speaker_embeddings)
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
                    
                    loss = self.criterion(outputs, labels_flat)
                    
                    # Check for NaN/Inf in loss or outputs (early detection)
                    if not torch.isfinite(loss):
                        logger.warning(f"Batch {batch_idx}: NaN/Inf loss detected! Loss: {loss.item()}")
                        logger.warning(f"  Outputs stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}, has_nan={torch.isnan(outputs).any().item()}, has_inf={torch.isinf(outputs).any().item()}")
                        logger.warning(f"  Logits stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                        logger.warning(f"  Features stats: min={features.min().item():.4f}, max={features.max().item():.4f}, mean={features.mean().item():.4f}")
                        logger.warning(f"  Labels stats: min={labels_flat.min().item()}, max={labels_flat.max().item()}, unique={torch.unique(labels_flat).cpu().numpy()[:10]}")
                        # Get label distribution for debugging
                        unique_labels_tensor, counts = torch.unique(labels_flat, return_counts=True)
                        label_dist = {int(l): int(c) for l, c in zip(unique_labels_tensor.cpu().numpy(), counts.cpu().numpy())}
                        logger.warning(f"  Label distribution: {label_dist}")
                        # Skip batch: zero grad and continue
                        self.optimizer.zero_grad()
                        continue
                    
                    # Backward pass
                    # DEBUG: Verify backward() is called (no no_grad() context)
                    loss.backward()
                    
                    # DEBUG: Compute and log gradient norm
                    grad_norm_before_clip = compute_grad_norm(self.model)
                    
                    # DEBUG: Log gradient norm for first few batches
                    if batch_idx < 5 or (batch_idx % 10 == 0):
                        print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Gradient norm (before clip) = {grad_norm_before_clip:.6f}")
                        if grad_norm_before_clip < 1e-8:
                            logger.warning(f"⚠️ CLIENT {self.client_id} Batch {batch_idx}: WARNING - Gradient norm is near zero! Gradients may not be flowing.")
                    
                    # Gradient clipping to prevent gradient explosion (fixes NaN loss issue)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    grad_norm_after_clip = compute_grad_norm(self.model)
                    
                    # DEBUG: Log gradient norm after clipping
                    if batch_idx < 5 or (batch_idx % 10 == 0):
                        print(f"🔍 CLIENT {self.client_id} Batch {batch_idx}: Gradient norm (after clip) = {grad_norm_after_clip:.6f}")
                    
                    self.optimizer.step()
                    
                    # DEBUG: Verify optimizer step was called
                    if batch_idx < 5:
                        print(f"✅ CLIENT {self.client_id} Batch {batch_idx}: optimizer.step() called successfully")
                
                    # Check loss value before adding (additional safety check)
                    loss_value = loss.item()
                    if np.isnan(loss_value) or np.isinf(loss_value):
                        logger.warning(f"Batch {batch_idx}: NaN/Inf loss value detected after backward pass! Loss: {loss_value}. Skipping batch.")
                        continue
                    
                    # Log batch statistics (for first few batches and periodically)
                    if batch_idx < 3 or batch_idx % 50 == 0:
                        logger.info(f"Batch {batch_idx}: loss={loss_value:.4f}, features_shape={features.shape}, labels_unique={torch.unique(labels_flat).cpu().numpy()[:10]}")
                    
                    # Only accumulate loss if batch was successful
                    train_loss += loss_value
                    batch_losses.append(loss_value)
                    
                    # E: Only accumulate predictions if DER is needed (memory optimization)
                    if should_compute_der:
                        predictions = torch.argmax(outputs, dim=-1)
                        predictions_np = predictions.cpu().numpy()
                        labels_np = labels_flat.cpu().numpy()
                        # FIX: Handle meeting_ids properly - it's a 1D array (one per sample), not per-frame
                        # Create per-frame meeting_id array by repeating for each frame in the sequence
                        # batch_size and seq_len are already extracted from outputs.shape at line 833
                        meeting_ids_flat = []
                        for i in range(batch_size):
                            # Handle zero-dimensional arrays and scalars
                            if isinstance(meeting_ids, np.ndarray):
                                # Handle both regular arrays and object arrays
                                if meeting_ids.dtype == object:
                                    meeting_id = meeting_ids[i]
                                else:
                                    meeting_id = meeting_ids[i].item() if meeting_ids[i].ndim == 0 else meeting_ids[i]
                            elif isinstance(meeting_ids, list):
                                meeting_id = meeting_ids[i]
                            else:
                                meeting_id = meeting_ids
                            # Repeat meeting_id for each frame in this sequence
                            meeting_ids_flat.extend([meeting_id] * seq_len)
                        meeting_ids_flat = np.array(meeting_ids_flat)
                        
                        for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                            if meeting_id is not None and label != -100:  # Skip padded frames
                                pred_by_rec[meeting_id].append(pred)
                                lab_by_rec[meeting_id].append(label)
                    
                    if batch_idx % 10 == 0:
                        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                    if batch_idx == 0 or (debug_mode and batch_idx < 5):
                        print(f"Batch {batch_idx}, labels shape: {labels_flat.shape}, unique labels: {torch.unique(labels_flat).cpu().numpy()}")
                        if should_compute_der:
                            print(f"Batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {torch.unique(predictions).cpu().numpy()}")
                    
                    # Debug mode: print label distribution and loss stats for first batches
                    if debug_mode and batch_idx < 10:
                        unique_labels, counts = torch.unique(labels_flat, return_counts=True)
                        label_dist = {int(l): int(c) for l, c in zip(unique_labels.cpu().numpy(), counts.cpu().numpy())}
                        print(f"[DEBUG] Batch {batch_idx} label distribution: {label_dist}")
                        print(f"[DEBUG] Batch {batch_idx} loss: {loss_value:.4f}, outputs_range=[{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
                
                # E: Calculate DER only if needed (memory optimization)
                der = None
                acc = None
                if should_compute_der and pred_by_rec and lab_by_rec:
                    ders = {}
                    for rec_id in pred_by_rec:
                        if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                            speaker_id_list = train_loader.dataset.get_speaker_id_list() if hasattr(train_loader.dataset, 'get_speaker_id_list') else None
                            ders[rec_id] = self.calculate_der(
                                pred_by_rec[rec_id],
                                lab_by_rec[rec_id],
                                speaker_id_list=speaker_id_list,
                                debug=False,
                                frame_shift=0.01,
                                uri=rec_id
                            )
                    der = np.mean(list(ders.values())) if ders else float('nan')
                    # Calculate accuracy across all frames
                    all_predictions = []
                    all_labels = []
                    for rec_id in pred_by_rec:
                        all_predictions.extend(pred_by_rec[rec_id])
                        all_labels.extend(lab_by_rec[rec_id])
                    acc = (np.array(all_predictions) == np.array(all_labels)).mean() if all_labels else float('nan')
                    # Clean up large objects immediately after DER calculation
                    del pred_by_rec, lab_by_rec, ders, all_predictions, all_labels
                else:
                    # C3: Don't accumulate predictions if DER not needed - save memory
                    pass
                
                # Metrics per epoch
                mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
                print(f"SENDClient: Epoch {epoch+1}/{epochs} summary for client {self.client_id}: min_loss={min(batch_losses) if batch_losses else 'nan'}, max_loss={max(batch_losses) if batch_losses else 'nan'}, mean_loss={mean_loss}, acc={acc if acc is not None else 'N/A'}, DER={der if der is not None else 'N/A'}")
                # Collect metrics for this epoch (C3: only small values)
                # ROBUSTNESS FIX: Use float('nan') instead of None for JSON serialization compatibility
                epoch_metrics.append({
                    "train_loss": float(mean_loss),
                    "acc": float(acc) if acc is not None and not np.isnan(acc) else float('nan'),
                    "der": float(der) if der is not None and not np.isnan(der) else float('nan'),
                })
            
            elapsed = time.time() - start_time
            print(f"SENDClient: Finished fit for client {self.client_id}, total time: {elapsed:.2f} sec")
            
            # DEBUG: Compute L2 norm difference to verify weights changed
            final_state_dict = {k: v.clone().detach() for k, v in self.model.state_dict().items()}
            final_l2_norm = sum(p.norm(2).item() ** 2 for p in self.model.parameters()) ** 0.5
            l2_norm_diff = compute_l2_norm_diff(final_state_dict, initial_state_dict)
            
            print(f"🔍 CLIENT {self.client_id}: Final L2 norm of parameters: {final_l2_norm:.6f}")
            print(f"🔍 CLIENT {self.client_id}: L2 norm difference (final - initial): {l2_norm_diff:.6f}")
            
            if l2_norm_diff < 1e-6:
                logger.error(f"❌ CLIENT {self.client_id}: CRITICAL - L2 norm difference is near zero ({l2_norm_diff:.2e})!")
                logger.error(f"   This means local training did NOT update model weights!")
                logger.error(f"   Possible causes:")
                logger.error(f"   1. Gradients are zero (check gradient norms above)")
                logger.error(f"   2. optimizer.step() not being called")
                logger.error(f"   3. Learning rate is zero")
                logger.error(f"   4. Model parameters have requires_grad=False")
            else:
                print(f"✅ CLIENT {self.client_id}: L2 norm difference confirms weights changed during training")
            
            # A3: Log checksum AFTER training and compute delta
            checksum_end = model_checksum(self.model)
            delta_vs_start = state_dict_delta(
                {k: v.cpu() for k, v in self.model.state_dict().items()},
                {k: torch.tensor(p).cpu() for k, p in zip(self.model.state_dict().keys(), parameters)}
            ) if parameters else 0.0
            print(f"CLIENT {self.client_id} end checksum: {checksum_end:.6f}; delta vs start: {delta_vs_start:.6f}")
            
            # B3: Verify training actually updated weights
            if delta_vs_start < 1e-8:
                logger.warning(f"CLIENT {self.client_id}: WARNING - delta_vs_start is ~0 ({delta_vs_start:.2e}), training may not have updated weights!")
            
            # DEBUG: Verify state_dict is a deep copy (not shared reference)
            returned_state_dict = self.model.state_dict()
            is_deep_copy = verify_state_dict_deep_copy(returned_state_dict)
            if not is_deep_copy:
                logger.warning(f"⚠️ CLIENT {self.client_id}: WARNING - state_dict may contain shared references!")
            else:
                print(f"✅ CLIENT {self.client_id}: state_dict verified as deep copy")
            
            print("=== CLIENT LOG: fit finished ===")
            
            # ROBUSTNESS FIX: Compute return values BEFORE deleting variables
            # C3: Return only small metrics (no large objects)
            final_mean_loss = epoch_metrics[-1]["train_loss"] if epoch_metrics else float('nan')
            epoch_metrics_json = json.dumps(epoch_metrics)  # Serialize before deletion
            updated_params = self.get_parameters({})
            
            # Log return values for debugging
            print(f"[DEBUG] CLIENT {self.client_id} fit() returning: num_examples={num_examples}, final_mean_loss={final_mean_loss:.6f}, epochs_trained={len(epoch_metrics)}")
            
            # MEMORY FIX: Explicit cleanup after training
            # Delete large tensors and force garbage collection
            del train_loader
            if 'pred_by_rec' in locals() and pred_by_rec is not None:
                del pred_by_rec, lab_by_rec
            gc.collect()
            
            # Only clear CUDA cache if using CUDA
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            return updated_params, num_examples, {"train_loss": final_mean_loss, "epoch_metrics": epoch_metrics_json}
        
        except Exception as e:
            # F: Error handling - return previous parameters unchanged
            logger.error(f"Client {self.client_id} fit() failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return previous parameters, 0 examples, and failure flag
            return self.get_parameters({}), 0, {"failed": 1, "error": str(e)}
    
    def evaluate(self, parameters, config):
        """Evaluate model on client validation data.
        
        C2: Creates DataLoader lazily inside evaluate() and cleans up after.
        C3: Returns only small metrics, no large objects.
        E: DER computation is optional (only when enabled or every N rounds).
        F: Wrapped in try/except for error handling.
        """
        # F: Error handling - return previous params on failure
        try:
            print("=== CLIENT LOG: evaluate started ===")
            print(f"SENDClient: Starting evaluate for client {self.client_id}")
            self.set_parameters(parameters)
            self.model.eval()
            
            server_round = config.get("server_round", 0)
            compute_der = config.get("compute_der", False)  # E: DER computation optional
            der_round_interval = config.get("der_round_interval", 5)  # E: Compute DER every N rounds
            
            # E: Only compute DER if enabled or every N rounds
            should_compute_der = compute_der or (server_round % der_round_interval == 0)
            
            # C2: Create DataLoader lazily inside evaluate()
            val_loader = self._create_val_loader()
            num_examples = len(val_loader.dataset)
            num_batches = len(val_loader)
            
            # ROBUSTNESS FIX: Check for empty dataset/loader before evaluation
            print(f"[DEBUG] evaluate: val_dataset size: {len(self.val_dataset) if self.val_dataset else 0}")
            print(f"[DEBUG] evaluate: val_loader size: {num_batches} batches")
            print(f"[DEBUG] evaluate: num_examples: {num_examples}")
            
            if num_examples == 0 or num_batches == 0:
                logger.warning(f"⚠️ CLIENT {self.client_id}: Empty validation dataset! num_examples={num_examples}, num_batches={num_batches}")
                logger.warning(f"   Returning safe defaults: loss=nan, num_examples=0")
                # Return safe tuple: (loss, num_examples, metrics_dict)
                # ROBUSTNESS FIX: Flower doesn't accept None in metrics - use float('nan') instead
                return float('nan'), 0, {"val_loss": float('nan'), "der": float('nan'), "epoch_metrics": json.dumps([]), "empty_dataset": True}
            
            val_loss = 0.0
            batch_losses = []  # MUST be initialized before any return
            # E: Only accumulate predictions if DER is needed (memory optimization)
            pred_by_rec = defaultdict(list) if should_compute_der else None
            lab_by_rec = defaultdict(list) if should_compute_der else None
            start_time = time.time()
            epoch_metrics = []  # Collect metrics for each epoch (for compatibility) - MUST be initialized
            
            with torch.no_grad():
                for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(val_loader):
                    if batch_idx == 0:
                        print(f"SENDClient: First batch in evaluate for client {self.client_id}")
                    features, speaker_embeddings, labels = features.to(self.device), speaker_embeddings.to(self.device), labels.to(self.device)
                    speaker_embeddings = speaker_embeddings.float()
                    
                    # Save original shape before processing (labels can be 1D or 2D)
                    labels_original_shape = labels.shape
                    labels_original_ndim = labels.ndim
                    
                    outputs = self.model(features, speaker_embeddings)
                    
                    # Handle both 1D and 2D labels (aligned with centralized version)
                    if outputs.ndim == 3:
                        # Standard case: outputs are (batch_size, seq_len, num_classes)
                        batch_size, seq_len, num_classes = outputs.shape
                        outputs = outputs.reshape(-1, num_classes)
                    elif outputs.ndim == 2:
                        # Outputs already flattened or (batch_size*seq_len, num_classes)
                        num_classes = outputs.shape[-1]
                        # Try to infer batch_size from features
                        batch_size = features.shape[0] if features.ndim >= 2 else 1
                        seq_len = outputs.shape[0] // batch_size if batch_size > 0 else outputs.shape[0]
                    else:
                        raise ValueError(f"Unexpected outputs ndim={outputs.ndim}, shape={outputs.shape}")
                    
                    # Handle labels: can be 1D or 2D
                    if labels.ndim == 2:
                        labels_flat = labels.reshape(-1)
                    elif labels.ndim == 1:
                        labels_flat = labels
                    else:
                        raise ValueError(f"Unexpected labels ndim={labels.ndim}, shape={labels.shape}")
                    
                    # Check for valid frames: skip batch if all labels are padding (-100)
                    valid = (labels_flat != -100)
                    if valid.sum() == 0:
                        logger.warning(f"Eval batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                        continue
                    
                    loss = self.criterion(outputs, labels_flat)
                    val_loss += loss.item()
                    batch_losses.append(loss.item())
                    
                    # Compute predictions for sanity-check and DER
                    predictions_flat = torch.argmax(outputs, dim=-1)
                    
                    # E: Only accumulate predictions if DER is needed (memory optimization)
                    if should_compute_der:
                        predictions_np = predictions_flat.cpu().numpy()
                        labels_np = labels_flat.cpu().numpy()
                        # MEMORY FIX: meeting_ids is now a 1D array (one per sample), not per-frame
                        # Create per-frame meeting_id array by repeating for each frame in the sequence
                        meeting_ids_flat = []
                        for i in range(batch_size):
                            meeting_id = meeting_ids[i] if isinstance(meeting_ids, (list, np.ndarray)) else meeting_ids
                            # Repeat meeting_id for each frame in this sequence
                            meeting_ids_flat.extend([meeting_id] * seq_len)
                        meeting_ids_flat = np.array(meeting_ids_flat)
                        
                        # Ensure meeting_ids_flat matches the length of predictions_np and labels_np
                        if len(meeting_ids_flat) != len(predictions_np):
                            min_len = min(len(meeting_ids_flat), len(predictions_np))
                            meeting_ids_flat = meeting_ids_flat[:min_len]
                            predictions_np = predictions_np[:min_len]
                            labels_np = labels_np[:min_len]
                        
                        for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                            if meeting_id is not None and label != -100:  # Skip padded frames
                                pred_by_rec[meeting_id].append(pred)
                                lab_by_rec[meeting_id].append(label)
                    
                    # Sanity-check (only for first 2 batches to avoid spam)
                    if batch_idx < 2:
                        valid_ratio = valid.float().mean().item() if valid.numel() > 0 else 0.0
                        uniq_labels = torch.unique(labels_flat[valid]).detach().cpu().tolist() if valid.any() else []
                        uniq_preds = torch.unique(predictions_flat[valid]).detach().cpu().tolist() if valid.any() else []
                        logger.info(
                            f"[EVAL DEBUG] Client {self.client_id}, Batch {batch_idx}: labels.ndim={labels_original_ndim}, "
                            f"labels.shape={tuple(labels_original_shape)}, "
                            f"outputs.shape={tuple(outputs.shape)}, "
                            f"valid_ratio={valid_ratio:.3f}, "
                            f"unique_labels={uniq_labels}, unique_preds={uniq_preds}"
                        )
                    
                    if batch_idx == 0:
                        print(f"Eval batch {batch_idx}, labels shape: {labels_original_shape}, unique labels: {np.unique(labels_flat[valid].cpu().numpy()) if valid.any() else []}")
                        print(f"Eval batch {batch_idx}, outputs shape: {outputs.shape}")
            
            # E: Calculate DER only if needed (memory optimization)
            der = None
            if should_compute_der and pred_by_rec and lab_by_rec:
                ders = {}
                for rec_id in pred_by_rec:
                    if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                        speaker_id_list = val_loader.dataset.get_speaker_id_list() if hasattr(val_loader.dataset, 'get_speaker_id_list') else None
                        ders[rec_id] = self.calculate_der(
                            pred_by_rec[rec_id],
                            lab_by_rec[rec_id],
                            speaker_id_list=speaker_id_list,
                            debug=False,
                            frame_shift=0.01,
                            uri=rec_id
                        )
                der = np.mean(list(ders.values())) if ders else float('nan')
                # Clean up large objects immediately after DER calculation
                del pred_by_rec, lab_by_rec, ders
            else:
                # C3: Don't accumulate predictions if DER not needed - save memory
                pass
            
            # ROBUSTNESS FIX: Compute return values BEFORE deleting variables
            # C3: Return only small metrics (no large objects)
            mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
            # ROBUSTNESS FIX: Use float('nan') instead of None for Flower compatibility
            epoch_metrics.append({
                "val_loss": float(mean_loss),
                "der": float(der) if der is not None and not np.isnan(der) else float('nan'),
            })
            epoch_metrics_json = json.dumps(epoch_metrics)  # Serialize before deletion
            
            # Log return values for debugging
            print(f"[DEBUG] CLIENT {self.client_id} evaluate() returning: num_examples={num_examples}, mean_loss={mean_loss:.6f}, batches_processed={len(batch_losses)}")
            
            # Safe print (batch_losses might be empty)
            if batch_losses:
                print(f"SENDClient: Eval summary for client {self.client_id}: min_loss={min(batch_losses):.4f}, max_loss={max(batch_losses):.4f}, mean_loss={mean_loss:.4f}, DER={der if der is not None else 'N/A'}")
            else:
                print(f"SENDClient: Eval summary for client {self.client_id}: No batches processed, mean_loss={mean_loss:.4f}, DER={der if der is not None else 'N/A'}")
            
            elapsed = time.time() - start_time
            print(f"SENDClient: Finished evaluate for client {self.client_id}, total time: {elapsed:.2f} sec")
            print("=== CLIENT LOG: evaluate finished ===")
            
            # MEMORY FIX: Explicit cleanup after evaluation
            # Delete large tensors and force garbage collection
            del val_loader
            if 'pred_by_rec' in locals() and pred_by_rec is not None:
                del pred_by_rec, lab_by_rec
            gc.collect()
            
            # Only clear CUDA cache if using CUDA
            if self.device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            
            # ROBUSTNESS FIX: Flower doesn't accept None in metrics - use float('nan') as sentinel
            metrics_dict = {
                "val_loss": mean_loss,
                "epoch_metrics": epoch_metrics_json
            }
            # Only add der if it's not None, otherwise use nan (Flower accepts float, not None)
            if der is not None and not np.isnan(der):
                metrics_dict["der"] = float(der)
            else:
                metrics_dict["der"] = float('nan')  # Use nan instead of None for Flower compatibility
            
            return (
                float(mean_loss),
                num_examples,
                metrics_dict
            )
        
        except Exception as e:
            # F: Error handling - return previous parameters unchanged
            logger.error(f"Client {self.client_id} evaluate() failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Return 0 loss, 0 examples, and failure flag
            return 0.0, 0, {"failed": 1, "error": str(e)}
    
    def calculate_der(self, predictions: List[int], labels: List[int], speaker_id_list: list = None, debug: bool = True, frame_shift: float = 0.01, uri: str = None) -> float:
        """Calculate Diarization Error Rate using the common function from data_processing."""
        from data_processing import calculate_der as common_calculate_der
        # FIX: Pass parameters by name to avoid parameter order confusion
        # der_frame_shift is not passed (will use default None -> 0.05)
        return common_calculate_der(
            predictions, 
            labels, 
            self.power_set_encoder, 
            speaker_id_list=speaker_id_list, 
            debug=debug, 
            frame_shift=frame_shift, 
            der_frame_shift=None,  # Use default
            uri=uri
        )


def get_domain(meeting_id: str) -> str:
    """
    Extract domain from meeting_id.
    
    AMI style: EN2001a, ES2011a, IB4001, IS1008a ...
    Domain is the first 2 characters.
    
    Args:
        meeting_id: Meeting identifier string
        
    Returns:
        str: Domain code (e.g., "EN", "ES", "IB", "IS")
    """
    return meeting_id[:2]


def subsample_grouped_by_meetings(grouped: dict, max_meetings, seed: int) -> dict:
    """
    Subsample grouped data by selecting a subset of meetings.
    
    Args:
        grouped: Dictionary of meeting_id to samples {meeting_id: [samples...]}
        max_meetings: Maximum number of meetings to keep (None = keep all)
        seed: Random seed for reproducibility
        
    Returns:
        dict: Subsampled grouped dictionary with selected meetings
    """
    if max_meetings is None or max_meetings >= len(grouped):
        return grouped
    
    rng = random.Random(seed)
    mids = list(grouped.keys())
    rng.shuffle(mids)
    selected_mids = mids[:max_meetings]
    
    result = {mid: grouped[mid] for mid in selected_mids}
    
    logger.info(f"Subsampled {len(grouped)} meetings to {len(result)} meetings (seed={seed})")
    logger.info(f"  Selected meeting IDs (first 5): {selected_mids[:5]}")
    
    return result


def stratified_partition_meetings(grouped_train: dict, num_clients: int, seed: int) -> list[dict]:
    """
    Partition meetings among FL clients using stratified (near-IID) distribution by domain.
    
    This ensures each client gets a similar mix of domains (EN/ES/IB/IS),
    which is important for stable FL training and better DER.
    
    Args:
        grouped_train: Dictionary of meeting_id to samples {meeting_id: [samples...]}
        num_clients: Number of clients to partition meetings among
        seed: Random seed for reproducibility
        
    Returns:
        list[dict]: List of grouped_train subsets, one per client
                   Each element is a dict {meeting_id: [samples...]}
    """
    logger.info(f"Stratified partitioning {len(grouped_train)} meetings among {num_clients} clients (near-IID by domain)...")
    
    # Validate input
    if not grouped_train:
        raise ValueError(f"grouped_train is empty! Cannot partition meetings among {num_clients} clients.")
    
    # Filter out empty meetings and log warnings
    non_empty_meetings = {}
    empty_meetings = []
    for meeting_id, samples in grouped_train.items():
        if samples and len(samples) > 0:
            non_empty_meetings[meeting_id] = samples
        else:
            empty_meetings.append(meeting_id)
    
    if empty_meetings:
        logger.warning(f"Found {len(empty_meetings)} empty meetings that will be excluded: {empty_meetings[:10]}{'...' if len(empty_meetings) > 10 else ''}")
    
    if not non_empty_meetings:
        raise ValueError(
            f"All meetings in grouped_train are empty! Cannot partition.\n"
            f"  - Total meetings: {len(grouped_train)}\n"
            f"  - Empty meetings: {len(empty_meetings)}\n"
            f"Please check your data and ensure meetings contain valid samples."
        )
    
    if len(non_empty_meetings) < num_clients:
        raise ValueError(
            f"Not enough non-empty meetings ({len(non_empty_meetings)}) for {num_clients} clients!\n"
            f"  - Total meetings: {len(grouped_train)}\n"
            f"  - Non-empty meetings: {len(non_empty_meetings)}\n"
            f"  - Required clients: {num_clients}\n"
            f"Please reduce num_clients or ensure you have more meetings with valid samples."
        )
    
    # Set random seed for reproducibility
    rng = random.Random(seed)
    
    # Bucket meetings by domain
    domain_to_mids = {}
    for mid in non_empty_meetings.keys():
        dom = get_domain(mid)
        domain_to_mids.setdefault(dom, []).append(mid)
    
    logger.info(f"Found {len(domain_to_mids)} domains: {list(domain_to_mids.keys())}")
    for dom, mids in domain_to_mids.items():
        logger.info(f"  Domain {dom}: {len(mids)} meetings")
    
    # Shuffle within each domain
    for dom in domain_to_mids:
        rng.shuffle(domain_to_mids[dom])
    
    # Round-robin assign within each domain (stratified distribution)
    client_mids = [[] for _ in range(num_clients)]
    for dom, mids in domain_to_mids.items():
        for i, mid in enumerate(mids):
            client_mids[i % num_clients].append(mid)
    
    # Build grouped dict for each client
    client_grouped = []
    for c in range(num_clients):
        client_grouped.append({mid: non_empty_meetings[mid] for mid in client_mids[c]})
    
    # Log detailed distribution for each client
    logger.info(f"✓ Stratified partition complete. Distribution by client:")
    for client_idx in range(num_clients):
        client_meeting_ids = client_mids[client_idx]
        num_samples = sum(len(non_empty_meetings[mid]) for mid in client_meeting_ids)
        
        # Count meetings by domain for this client
        domain_counts = {}
        for mid in client_meeting_ids:
            dom = get_domain(mid)
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
        
        logger.info(f"  Client {client_idx}: meetings={len(client_meeting_ids)}, segments={num_samples}, domains={domain_counts}")
        logger.info(f"    Meeting IDs (first 5): {client_meeting_ids[:5]}{'...' if len(client_meeting_ids) > 5 else ''}")
    
    return client_grouped


def partition_meetings_among_clients(grouped_train: dict, num_clients: int, seed: int = 42):
    """Partition meetings among FL clients by meeting_id.
    
    Each meeting belongs to exactly one client. Meetings are randomly shuffled
    and distributed approximately evenly among clients.
    
    Args:
        grouped_train: Dictionary of meeting_id to samples {meeting_id: [samples...]}
        num_clients: Number of clients to partition meetings among
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        list[dict]: List of grouped_train subsets, one per client
                   Each element is a dict {meeting_id: [samples...]}
    """
    logger.info(f"Partitioning {len(grouped_train)} meetings among {num_clients} clients...")
    
    # Validate input
    if not grouped_train:
        raise ValueError(f"grouped_train is empty! Cannot partition meetings among {num_clients} clients.")
    
    # Filter out empty meetings and log warnings
    non_empty_meetings = {}
    empty_meetings = []
    for meeting_id, samples in grouped_train.items():
        if samples and len(samples) > 0:
            non_empty_meetings[meeting_id] = samples
        else:
            empty_meetings.append(meeting_id)
    
    if empty_meetings:
        logger.warning(f"Found {len(empty_meetings)} empty meetings that will be excluded: {empty_meetings[:10]}{'...' if len(empty_meetings) > 10 else ''}")
    
    if not non_empty_meetings:
        raise ValueError(
            f"All meetings in grouped_train are empty! Cannot partition.\n"
            f"  - Total meetings: {len(grouped_train)}\n"
            f"  - Empty meetings: {len(empty_meetings)}\n"
            f"Please check your data and ensure meetings contain valid samples."
        )
    
    if len(non_empty_meetings) < num_clients:
        raise ValueError(
            f"Not enough non-empty meetings ({len(non_empty_meetings)}) for {num_clients} clients!\n"
            f"  - Total meetings: {len(grouped_train)}\n"
            f"  - Non-empty meetings: {len(non_empty_meetings)}\n"
            f"  - Required clients: {num_clients}\n"
            f"Please reduce num_clients or ensure you have more meetings with valid samples."
        )
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Get list of meeting IDs (only non-empty) and shuffle
    meeting_ids = list(non_empty_meetings.keys())
    np.random.shuffle(meeting_ids)
    
    logger.info(f"Shuffled {len(meeting_ids)} non-empty meeting IDs with seed={seed}")
    
    # Distribute meetings among clients (approximately evenly)
    meetings_per_client = len(meeting_ids) // num_clients
    remainder = len(meeting_ids) % num_clients
    
    client_grouped_trains = []
    start_idx = 0
    
    for client_idx in range(num_clients):
        # First 'remainder' clients get one extra meeting
        num_meetings = meetings_per_client + (1 if client_idx < remainder else 0)
        end_idx = start_idx + num_meetings
        
        client_meeting_ids = meeting_ids[start_idx:end_idx]
        
        # Create subset of grouped_train for this client (only non-empty meetings)
        client_grouped = {mid: non_empty_meetings[mid] for mid in client_meeting_ids}
        client_grouped_trains.append(client_grouped)
        
        # Count samples for this client
        num_samples = sum(len(samples) for samples in client_grouped.values())
        meeting_sample_counts = {mid: len(samples) for mid, samples in client_grouped.items()}
        logger.info(f"  Client {client_idx}: {len(client_meeting_ids)} meetings, {num_samples} samples")
        logger.info(f"    Meeting IDs: {client_meeting_ids[:5]}{'...' if len(client_meeting_ids) > 5 else ''}")
        logger.info(f"    Sample counts: {dict(list(meeting_sample_counts.items())[:3])}{'...' if len(meeting_sample_counts) > 3 else ''}")
        
        # Validate that client has samples
        if num_samples == 0:
            logger.error(f"  ⚠️  WARNING: Client {client_idx} has 0 samples! This will cause dataset creation to fail.")
        
        start_idx = end_idx
    
    logger.info(f"✓ Partitioned {len(meeting_ids)} non-empty meetings among {num_clients} clients")
    
    return client_grouped_trains


def main():
    # Start timing
    start_time = time.time()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Federated Learning for Overlapping Speech Diarization")
    parser.add_argument('--test_size', type=int, default=None, help='[DEPRECATED] Number of dataset records to use for training. Use --subset_*_meetings instead to avoid domain skew. If subset_*_meetings are set, this parameter is ignored.')
    parser.add_argument('--subset_train_meetings', type=int, default=None, help='Number of meetings to use for training (default: None = use all meetings). Recommended: 40 for quick testing.')
    parser.add_argument('--subset_val_meetings', type=int, default=None, help='Number of meetings to use for validation (default: None = use all meetings). Recommended: 10 for quick testing.')
    parser.add_argument('--subset_test_meetings', type=int, default=None, help='Number of meetings to use for testing (default: None = use all meetings). Recommended: 10 for quick testing.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for local training')
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of federated learning rounds')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of federated clients')
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
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode: train on 200-500 batches and print detailed label distribution and loss stats')
    parser.add_argument('--debug_max_batches', type=int, default=200, help='Maximum number of batches to process in debug mode (default: 200)')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizer (default: 3e-4, matches centralized training)')
    # CPU-only parallel FL configuration
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use: auto (detect), cpu (force CPU), cuda (force GPU). For parallel FL on CPU, use --device cpu')
    parser.add_argument('--clients_per_round', type=int, default=None, help='Number of clients to train per round (default: None = sequential/1 per round). For parallel FL, set to 2-4')
    parser.add_argument('--num_cpus_per_client', type=int, default=1, help='Number of CPUs per client (default: 1). Increase for parallel CPU clients')
    # Early stopping parameters
    parser.add_argument('--early_stopping_patience', type=int, default=None, help='Early stopping patience in rounds (default: None = disabled). If set, training stops if validation metric does not improve for this many rounds.')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement for early stopping (default: 0.001)')
    args = parser.parse_args()

    # Assign arguments to variables
    test_size = args.test_size
    subset_train_meetings = args.subset_train_meetings
    subset_val_meetings = args.subset_val_meetings
    subset_test_meetings = args.subset_test_meetings
    seed = args.seed
    epochs = args.epochs
    num_rounds = args.num_rounds
    num_clients = args.num_clients
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
    debug_mode = args.debug_mode
    debug_max_batches = args.debug_max_batches
    learning_rate = args.lr
    device_arg = args.device
    clients_per_round = args.clients_per_round
    num_cpus_per_client = args.num_cpus_per_client
    early_stopping_patience = args.early_stopping_patience
    early_stopping_min_delta = args.early_stopping_min_delta
    
    # Determine if we're using all data or a subset
    # Determine if using all data (no subsampling)
    use_all_data = (test_size is None and 
                    subset_train_meetings is None and 
                    subset_val_meetings is None and 
                    subset_test_meetings is None)

    print("MAIN STARTED")
    print(f"MAIN: Starting main()")
    try:
        # Device selection: force CPU if requested, otherwise auto-detect
        if device_arg == 'cpu':
            device = torch.device("cpu")
            # Disable CUDA initialization to prevent GPU memory allocation
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print(f"MAIN: FORCING CPU mode (CUDA disabled)")
        elif device_arg == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            device = torch.device("cuda")
            print(f"MAIN: FORCING CUDA mode")
        else:  # auto
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"MAIN: Auto-detected device: {device}")
        
        print(f"MAIN: Using device: {device}")
        
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
        
        # Group data by meeting ID for all splits (before subsampling)
        print(f"MAIN: Grouping data by meeting ID...")
        grouped_train_raw = group_by_meeting(dataset["train"])
        grouped_validation_raw = group_by_meeting(dataset["validation"])
        grouped_test_raw = group_by_meeting(dataset["test"])
        
        print(f"MAIN: Raw grouped data - Train: {len(grouped_train_raw)} meetings, Val: {len(grouped_validation_raw)} meetings, Test: {len(grouped_test_raw)} meetings")
        
        # Apply subsampling by meetings if specified (NEW: better than test_size which cuts segments)
        if subset_train_meetings is not None or subset_val_meetings is not None or subset_test_meetings is not None:
            print(f"MAIN: Applying meeting-based subsampling (recommended for near-IID distribution)...")
            if test_size is not None:
                logger.warning("⚠️  WARNING: Both --test_size and --subset_*_meetings are set. Ignoring --test_size (deprecated).")
            
            grouped_train = subsample_grouped_by_meetings(grouped_train_raw, subset_train_meetings, seed)
            grouped_validation = subsample_grouped_by_meetings(grouped_validation_raw, subset_val_meetings, seed + 1)
            grouped_test = subsample_grouped_by_meetings(grouped_test_raw, subset_test_meetings, seed + 2)
            
            print(f"MAIN: After subsampling - Train: {len(grouped_train)} meetings, Val: {len(grouped_validation)} meetings, Test: {len(grouped_test)} meetings")
        elif test_size is not None:
            # Fallback to old behavior (test_size cuts segments - can cause domain skew)
            logger.warning("⚠️  WARNING: Using deprecated --test_size parameter. This cuts segments and can cause extreme non-IID distribution.")
            logger.warning("   Consider using --subset_train_meetings, --subset_val_meetings, --subset_test_meetings instead.")
            
            train_size = test_size
            val_size = round(test_size/0.7*0.3)
            test_size_val = round(test_size/0.7*0.3)
            print(f"MAIN: Using SUBSET (deprecated) - Train: {train_size} segments, Val: {val_size} segments, Test: {test_size_val} segments")
            
            grouped_train = group_by_meeting(dataset["train"].select(range(train_size)))
            grouped_validation = group_by_meeting(dataset["validation"].select(range(val_size)))
            grouped_test = group_by_meeting(dataset["test"].select(range(test_size_val)))
        else:
            # Use all data
            grouped_train = grouped_train_raw
            grouped_validation = grouped_validation_raw
            grouped_test = grouped_test_raw
            print(f"MAIN: Using ALL data - Train: {len(grouped_train)} meetings, Val: {len(grouped_validation)} meetings, Test: {len(grouped_test)} meetings")
        
        print_dataset_overview("AMI", len(dataset["train"]), len(grouped_train))
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
        
        # DIAGNOSTIC: Check label distribution before training (run once)
        def analyze_label_distribution(loader, loader_name, power_set_encoder, num_classes):
            """Analyze label distribution in a data loader."""
            print(f"\n{'='*80}")
            print(f"LABEL DISTRIBUTION ANALYSIS: {loader_name}")
            print(f"{'='*80}")
            
            all_labels = []
            total_frames = 0
            ignore_index_count = 0
            
            print(f"Scanning {len(loader)} batches...")
            for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(loader):
                labels_flat = labels.reshape(-1).cpu().numpy()
                all_labels.extend(labels_flat.tolist())
                total_frames += len(labels_flat)
                ignore_index_count += (labels_flat == -100).sum()
                
                # Show progress for large datasets
                if (batch_idx + 1) % max(1, len(loader) // 10) == 0:
                    print(f"  Processed {batch_idx + 1}/{len(loader)} batches...")
            
            all_labels = np.array(all_labels)
            valid_labels = all_labels[all_labels != -100]
            
            print(f"\nTotal frames: {total_frames:,}")
            print(f"Ignore index (-100) frames: {ignore_index_count:,} ({100*ignore_index_count/total_frames:.2f}%)")
            print(f"Valid frames: {len(valid_labels):,} ({100*len(valid_labels)/total_frames:.2f}%)")
            
            if len(valid_labels) > 0:
                unique_labels, counts = np.unique(valid_labels, return_counts=True)
                print(f"\nUnique classes (excluding -100): {len(unique_labels)}")
                print(f"Expected num_classes: {num_classes}")
                
                if len(unique_labels) == 1:
                    print(f"⚠️  WARNING: Only ONE class found in {loader_name}!")
                    print(f"   This indicates a problem with data split or label encoding.")
                    print(f"   Unique class: {unique_labels[0]}")
                else:
                    print(f"✓ Multiple classes found: {len(unique_labels)} classes")
                
                # Top-N most frequent classes
                top_n = min(10, len(unique_labels))
                sorted_indices = np.argsort(counts)[::-1][:top_n]
                print(f"\nTop-{top_n} most frequent classes:")
                for idx in sorted_indices:
                    label = unique_labels[idx]
                    count = counts[idx]
                    percentage = 100 * count / len(valid_labels)
                    # Decode label to show speaker combination
                    try:
                        speakers = power_set_encoder.decode(int(label))
                        speaker_str = f"speakers={speakers}"
                    except:
                        speaker_str = "decode_error"
                    print(f"  Class {label:3d} ({speaker_str:20s}): {count:8,} frames ({percentage:5.2f}%)")
                
                # Check for out-of-range labels
                out_of_range = (valid_labels < 0) | (valid_labels >= num_classes)
                if out_of_range.any():
                    invalid_labels = valid_labels[out_of_range]
                    unique_invalid = np.unique(invalid_labels)
                    print(f"\n⚠️  ERROR: Found {out_of_range.sum()} out-of-range labels!")
                    print(f"   Invalid label values: {unique_invalid}")
                    print(f"   Valid range: [0, {num_classes-1}]")
                else:
                    print(f"\n✓ All labels are in valid range [0, {num_classes-1}]")
            else:
                print(f"\n⚠️  WARNING: No valid labels found in {loader_name}!")
            
            print(f"{'='*80}\n")
            return {
                'total_frames': total_frames,
                'ignore_index_count': ignore_index_count,
                'valid_frames': len(valid_labels),
                'unique_classes': len(unique_labels) if len(valid_labels) > 0 else 0,
                'unique_labels': unique_labels.tolist() if len(valid_labels) > 0 else []
            }
        
        # Run diagnostic analysis
        print("\n" + "="*80)
        print("RUNNING PRE-TRAINING LABEL DISTRIBUTION DIAGNOSTICS")
        print("="*80)
        train_label_stats = analyze_label_distribution(train_loader, "TRAIN", power_set_encoder, num_classes)
        val_label_stats = analyze_label_distribution(val_loader, "VALIDATION", power_set_encoder, num_classes)
        
        # Check if validation has only one class
        if val_label_stats['unique_classes'] == 1:
            print("⚠️  CRITICAL: Validation set contains only ONE class!")
            print("   This will cause training issues. Please check:")
            print("   1. How train/val split is formed (should not be speaker/scene-based)")
            print("   2. How power_set_encoder is applied (all combinations should not map to one id)")
            print("   3. Label overwriting during padding/truncation")
        
        # Print experiment configuration
        # For backward compatibility, pass test_size if using old method, otherwise pass number of test meetings
        config_test_size = test_size if test_size is not None and subset_test_meetings is None else (len(grouped_test) if subset_test_meetings is None else subset_test_meetings)
        print_experiment_config(num_clients, num_rounds, epochs, config_test_size)
        
        # Log meeting-based subsampling info if used
        if subset_train_meetings is not None or subset_val_meetings is not None or subset_test_meetings is not None:
            print(f"MAIN: Meeting-based subsampling: Train={len(grouped_train)} meetings, Val={len(grouped_validation)} meetings, Test={len(grouped_test)} meetings")
            print(f"MAIN: Seed used: {seed} (train), {seed+1} (val), {seed+2} (test)")

        
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
        
        # NEW FL PIPELINE: Partition meetings among clients using stratified (near-IID) distribution
        print(f"MAIN: Partitioning meetings among {num_clients} clients using stratified distribution (near-IID by domain)...")
        client_grouped_trains = stratified_partition_meetings(
            grouped_train, num_clients, seed=seed
        )
        
        # Validate partition
        if not client_grouped_trains or len(client_grouped_trains) < num_clients:
            raise ValueError(f"Not enough meetings for {num_clients} clients. Only {len(client_grouped_trains) if client_grouped_trains else 0} clients can be created.")
        
        print(f"MAIN: Partitioned meetings among {len(client_grouped_trains)} clients")
        
        # OPTIMIZATION: Build val and test datasets once globally (shared across all clients)
        print(f"MAIN: Building shared val and test datasets (once for all clients)...")
        val_dataset = build_eval_dataset(
            grouped_eval=grouped_validation,
            speaker_encoder=speaker_encoder,
            power_set_encoder=power_set_encoder,
            max_speakers=N,
            chunk_size=chunk_size,
            max_sequence_length=max_sequence_length,
            dataset_name="val"
        )
        test_dataset = build_eval_dataset(
            grouped_eval=grouped_test,
            speaker_encoder=speaker_encoder,
            power_set_encoder=power_set_encoder,
            max_speakers=N,
            chunk_size=chunk_size,
            max_sequence_length=max_sequence_length,
            dataset_name="test"
        )
        print(f"MAIN: Built shared datasets: val={len(val_dataset)} samples, test={len(test_dataset)} samples")
        print(f"MAIN: These datasets will be safely shared across all {num_clients} clients (OverlappingSpeechDataset is read-only)")
        
        # Build train datasets for each client (only train, val/test are shared)
        print(f"MAIN: Building train datasets for each client...")
        client_datasets = []
        
        for client_idx, client_grouped_train in enumerate(client_grouped_trains):
            print(f"MAIN: Building train dataset for client {client_idx}...")
            train_dataset = build_train_dataset_for_grouped_meetings(
                grouped_train_subset=client_grouped_train,
                speaker_encoder=speaker_encoder,
                power_set_encoder=power_set_encoder,
                max_speakers=N,
                chunk_size=chunk_size,
                max_sequence_length=max_sequence_length
            )
            # Each client gets its own train_dataset but shares val_dataset
            client_datasets.append((train_dataset, val_dataset))
            
            print(f"MAIN: Client {client_idx}: train={len(train_dataset)} samples, val={len(val_dataset)} samples (shared)")
        
        print(f"MAIN: Built train datasets for {len(client_datasets)} clients")
        
        # Calculate training samples statistics for logging
        total_training_samples = sum(len(train_dataset) for train_dataset, _ in client_datasets)
        total_val_samples = len(val_dataset) if val_dataset else 0
        total_test_samples = len(test_dataset) if test_dataset else 0
        
        # Estimate total training frames (approximate, using average sequence length)
        total_training_frames = 0
        client_samples_info = []
        for client_idx, (train_dataset, _) in enumerate(client_datasets):
            client_train_samples = len(train_dataset)
            client_val_samples = total_val_samples  # Shared across all clients
            client_total_samples = client_train_samples + client_val_samples
            
            # Estimate frames (approximate - actual frames vary per sample)
            # Use a rough estimate: average sequence length * number of samples
            # This is just for logging, not critical
            avg_seq_len_estimate = max_sequence_length if max_sequence_length else 1000
            client_frames_estimate = client_total_samples * avg_seq_len_estimate
            
            total_training_frames += client_frames_estimate
            
            client_samples_info.append({
                'client_id': client_idx,
                'train_samples': client_train_samples,
                'val_samples': client_val_samples,
                'total_samples': client_total_samples,
                'actual_frames': client_frames_estimate
            })
            
            print(f"MAIN: Client {client_idx}: {client_train_samples} train samples, {client_val_samples} val samples (shared), {client_total_samples} total samples")
        
        print(f"MAIN: Total training samples across all clients: {total_training_samples}")
        print(f"MAIN: Total training frames (estimated): {total_training_frames}")
        print(f"MAIN: Shared validation samples: {total_val_samples}")
        print(f"MAIN: Shared test samples: {total_test_samples}")
        
        # SANITY CHECKS: Verify dataset structure before starting FL
        print(f"\n{'='*80}")
        print(f"MAIN: Running sanity checks before FL training...")
        print(f"{'='*80}")
        
        sanity_check_passed = True
        
        # Check 2-3 random samples from different clients
        for check_idx in range(min(3, len(client_datasets))):
            train_dataset, val_dataset = client_datasets[check_idx]
            
            if len(train_dataset) == 0:
                print(f"❌ SANITY CHECK FAILED: Client {check_idx} has empty train dataset!")
                sanity_check_passed = False
                continue
            
            # Get a random sample
            sample_idx = np.random.randint(0, len(train_dataset))
            features, speaker_embeddings, labels, meeting_id = train_dataset[sample_idx]
            
            # Check shapes
            if speaker_embeddings.shape != (N, speaker_embeddings.shape[1]):
                print(f"❌ SANITY CHECK FAILED: Client {check_idx}, sample {sample_idx}")
                print(f"   Expected speaker_embeddings.shape == ({N}, emb_dim), got {speaker_embeddings.shape}")
                sanity_check_passed = False
            
            if labels.shape[0] != features.shape[0]:
                print(f"❌ SANITY CHECK FAILED: Client {check_idx}, sample {sample_idx}")
                print(f"   Expected labels.shape[0] == features.shape[0], got labels.shape={labels.shape}, features.shape={features.shape}")
                sanity_check_passed = False
            
            # Check that embeddings are not all zeros (should have actual values)
            if torch.allclose(speaker_embeddings, torch.zeros_like(speaker_embeddings)):
                print(f"⚠️  SANITY CHECK WARNING: Client {check_idx}, sample {sample_idx}")
                print(f"   All speaker embeddings are zeros - this may indicate missing embeddings")
            
            print(f"✓ Client {check_idx}, sample {sample_idx}: features.shape={features.shape}, "
                  f"speaker_embeddings.shape={speaker_embeddings.shape}, labels.shape={labels.shape}, meeting_id={meeting_id}")
        
        # Check that different meetings have different embeddings
        if len(client_datasets) >= 2:
            train_dataset_0, _ = client_datasets[0]
            train_dataset_1, _ = client_datasets[1]
            
            if len(train_dataset_0) > 0 and len(train_dataset_1) > 0:
                sample_0 = train_dataset_0[0]
                sample_1 = train_dataset_1[0]
                
                _, emb_0, _, meeting_id_0 = sample_0
                _, emb_1, _, meeting_id_1 = sample_1
                
                if meeting_id_0 != meeting_id_1:
                    emb_diff_norm = torch.norm(emb_0[0] - emb_1[0]).item()
                    if emb_diff_norm == 0:
                        print(f"❌ SANITY CHECK FAILED: Different meetings have identical embeddings!")
                        print(f"   meeting_id_0={meeting_id_0}, meeting_id_1={meeting_id_1}, norm(emb_0[0] - emb_1[0])={emb_diff_norm}")
                        sanity_check_passed = False
                    else:
                        print(f"✓ Different meetings have different embeddings: norm(emb_0[0] - emb_1[0])={emb_diff_norm:.6f}")
        
        if not sanity_check_passed:
            print(f"\n❌ SANITY CHECKS FAILED! Cannot start FL training.")
            print(f"   Please fix the issues above before proceeding.")
            raise ValueError("Sanity checks failed - see logs above")
        
        print(f"{'='*80}")
        print(f"✓ All sanity checks passed! Ready to start FL training.")
        print(f"{'='*80}\n")
        
        # Define client function for simulation
        def client_fn(context: Context):
            cid = context.node_config['partition-id']
            print(f"[client_fn] Got cid from context.node_config['partition-id']: {cid}")
            print(f"MAIN: Creating client {cid}")
            try:
                client_idx = int(cid)
                if client_idx >= len(client_datasets):
                    raise ValueError(f"Client ID {client_idx} is out of range. Only {len(client_datasets)} clients available.")
                train_dataset, val_dataset = client_datasets[client_idx]
                
                # ROBUSTNESS FIX: Log dataset sizes at client initialization
                train_size = len(train_dataset) if train_dataset else 0
                val_size = len(val_dataset) if val_dataset else 0
                
                print(f"[client_fn] Client {cid} dataset sizes:")
                print(f"  - Train dataset: {train_size} samples")
                print(f"  - Val dataset: {val_size} samples")
                
                if train_size == 0:
                    logger.warning(f"⚠️ CLIENT {cid}: Empty training dataset! This will cause fit() to return 0 examples.")
                if val_size == 0:
                    logger.warning(f"⚠️ CLIENT {cid}: Empty validation dataset! This will cause evaluate() to return 0 examples.")
                
                # Create new model instance for each client with configurable architecture
                client_model = SENDModel(
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    num_speech_encoder_layers=num_speech_encoder_layers,
                    num_post_net_layers=num_post_net_layers,
                    num_transformer_layers=num_transformer_layers
                ).to(device)
                print(f"MAIN: Client {cid} created and ready")
                return SENDClient(
                    model=client_model,
                    train_dataset=train_dataset,  # Pass dataset directly (not DataLoader)
                    val_dataset=val_dataset,    # Pass dataset directly (not DataLoader)
                    device=device,
                    power_set_encoder=power_set_encoder,
                    speaker_encoder=speaker_encoder,
                    batch_size=batch_size,
                    client_id=client_idx
                ).to_client()
            except Exception as e:
                logger.error(f"Error creating client {cid}: {str(e)}")
                raise
        
    
        # Start federated learning with simulation
        print("Starting federated learning simulation...")
        # Prepare to collect metrics per client and per round
        client_epoch_metrics = defaultdict(lambda: defaultdict(list))  # client_epoch_metrics[client_id][round] = list of epoch dicts
        round_metrics = []  # list of dicts: {'round': r, 'mean_loss': ..., 'mean_der': ..., 'client_metrics': {cid: {...}}}

        # Custom aggregation function for fit metrics
        def fit_metrics_aggregation_fn(metrics):
            # metrics: List[Tuple[int, dict]]
            round_info = {'client_metrics': {}}
            for cid, (num_examples, m) in enumerate(metrics):
                if 'epoch_metrics' in m:
                    epoch_metrics = json.loads(m['epoch_metrics'])
                    client_epoch_metrics[cid][len(round_metrics)].extend(epoch_metrics)
                    last = epoch_metrics[-1] if epoch_metrics else {}
                    round_info['client_metrics'][cid] = last
            # Compute mean loss/der for this round
            losses = [v.get('train_loss') for v in round_info['client_metrics'].values() if 'train_loss' in v and v.get('train_loss') is not None]
            ders = [v.get('der') for v in round_info['client_metrics'].values() if 'der' in v and v.get('der') is not None]
            round_info['mean_loss'] = sum(losses)/len(losses) if losses else None
            round_info['mean_der'] = sum(ders)/len(ders) if ders else None
            round_info['round'] = len(round_metrics)
            round_metrics.append(round_info)
            return {}

        # Create a custom strategy that captures final parameters and ensures fit is executed
        class SaveFinalParams(fl.server.strategy.FedAvg):
            def __init__(self, global_model=None, val_loader=None, early_stopping_patience=None, early_stopping_min_delta=0.001, **kwargs):
                super().__init__(**kwargs)
                self.final_parameters = None  # aggregated weights
                self.global_model = global_model  # C: Store reference for prediction sanity check
                self.val_loader = val_loader  # C: Store reference for prediction sanity check
                self.round_predictions = {}  # C: Store predictions per round for sanity check
                # Early stopping state
                self.early_stopping_patience = early_stopping_patience
                self.early_stopping_min_delta = early_stopping_min_delta
                self.best_val_der = float('inf')  # DER: lower is better
                self.best_val_loss = float('inf')  # Loss: lower is better
                self.patience_counter = 0
                self.best_round = 0
                self.best_parameters = None  # Store best model parameters (will be initialized in first aggregate_fit)
                self.should_stop = False  # Flag to signal early stopping
                if early_stopping_patience is not None:
                    print(f"✓ SaveFinalParams strategy initialized with early stopping (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})")
                else:
                    print(f"✓ SaveFinalParams strategy initialized (early stopping disabled)")
            
            def aggregate_fit(self, server_round, results, failures):
                print(f"✓ Round {server_round}: aggregate_fit called with {len(results)} results, {len(failures)} failures")
                
                # Check if early stopping was triggered in previous round
                if self.should_stop:
                    logger.warning(f"Round {server_round}: Early stopping was triggered in previous round. Skipping fit aggregation.")
                    return None, {"skipped": 1, "reason": "early_stopping"}
                
                # Protection against division by zero: check if we have valid results
                if not results:
                    logger.error(f"Round {server_round}: no fit results (all clients failed). Skipping aggregation.")
                    return None, {"skipped": 1, "reason": "no_results"}
                
                # A4: Log each client's num_examples and checksum of returned params BEFORE aggregation
                print(f"SERVER round {server_round}: Receiving results from {len(results)} clients")
                client_params_list = []
                client_num_examples_list = []
                for cid, fit_res in results:
                    num_examples = fit_res.num_examples
                    # Convert Flower parameters to state dict for checksum
                    params_ndarrays = fl.common.parameters_to_ndarrays(fit_res.parameters)
                    client_params_list.append(params_ndarrays)
                    client_num_examples_list.append(num_examples)
                    # Create a temporary state dict for checksum calculation
                    temp_state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(params_ndarrays)}
                    checksum = state_dict_checksum(temp_state_dict)
                    print(f"SERVER round {server_round}: CLIENT {cid} num_examples={num_examples}, checksum={checksum:.6f}")
                    logger.info(f"Round {server_round}: Client {cid} returned {num_examples} examples in fit")
                    
                    # DEBUG: Verify state_dict is a deep copy
                    is_deep_copy = verify_state_dict_deep_copy(temp_state_dict)
                    if not is_deep_copy:
                        logger.warning(f"⚠️ SERVER round {server_round}: CLIENT {cid} state_dict may contain shared references!")
                    else:
                        print(f"✅ SERVER round {server_round}: CLIENT {cid} state_dict verified as deep copy")
                
                # Check if total num_examples is zero (would cause ZeroDivisionError in FedAvg)
                num_examples_total = sum(fit_res.num_examples for _, fit_res in results)
                if num_examples_total == 0:
                    logger.error(f"Round {server_round}: num_examples_total=0. Skipping aggregation to avoid division by zero.")
                    return None, {"skipped": 1, "reason": "zero_examples", "num_results": len(results)}
                
                # A1: Log checksum BEFORE aggregation (from previous round's parameters)
                if hasattr(self, 'final_parameters') and self.final_parameters is not None:
                    prev_params_ndarrays = fl.common.parameters_to_ndarrays(self.final_parameters)
                    prev_state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(prev_params_ndarrays)}
                    prev_checksum = state_dict_checksum(prev_state_dict)
                    print(f"SERVER round {server_round} global checksum (before aggregation): {prev_checksum:.6f}")
                else:
                    # First round - use initial parameters
                    if hasattr(self, 'initial_parameters') and self.initial_parameters is not None:
                        init_params_ndarrays = fl.common.parameters_to_ndarrays(self.initial_parameters)
                        init_state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(init_params_ndarrays)}
                        init_checksum = state_dict_checksum(init_state_dict)
                        print(f"SERVER round {server_round} global checksum (initial, before aggregation): {init_checksum:.6f}")
                
                # DEBUG: Verify FedAvg computes true weighted average
                # Manually compute weighted average for first parameter to verify
                if len(client_params_list) > 0 and len(client_params_list[0]) > 0:
                    first_param_idx = 0
                    manual_weighted_avg = np.zeros_like(client_params_list[0][first_param_idx])
                    total_weight = sum(client_num_examples_list)
                    for client_params, num_examples in zip(client_params_list, client_num_examples_list):
                        weight = num_examples / total_weight if total_weight > 0 else 1.0 / len(client_params_list)
                        manual_weighted_avg += client_params[first_param_idx] * weight
                    print(f"🔍 SERVER round {server_round}: Manual FedAvg check - first param weighted avg (sample): {manual_weighted_avg.flatten()[:5]}")
                
                # Proceed with normal aggregation
                aggregated, metrics = super().aggregate_fit(server_round, results, failures)
                if aggregated is not None:
                    # A5: Log checksum AFTER aggregation and compute delta
                    aggregated_params_ndarrays = fl.common.parameters_to_ndarrays(aggregated)
                    aggregated_state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(aggregated_params_ndarrays)}
                    aggregated_checksum = state_dict_checksum(aggregated_state_dict)
                    
                    # DEBUG: Verify FedAvg result matches manual computation
                    if len(client_params_list) > 0 and len(client_params_list[0]) > 0:
                        fedavg_first_param = aggregated_params_ndarrays[0]
                        diff = np.abs(manual_weighted_avg - fedavg_first_param).max()
                        print(f"🔍 SERVER round {server_round}: FedAvg verification - max diff vs manual: {diff:.2e}")
                        if diff > 1e-5:
                            logger.warning(f"⚠️ SERVER round {server_round}: WARNING - FedAvg result differs from manual computation by {diff:.2e}!")
                        else:
                            print(f"✅ SERVER round {server_round}: FedAvg verified - matches manual weighted average")
                    
                    # Compute delta vs previous
                    delta_vs_prev = 0.0
                    l2_norm_diff_prev = 0.0
                    if hasattr(self, 'final_parameters') and self.final_parameters is not None:
                        prev_params_ndarrays = fl.common.parameters_to_ndarrays(self.final_parameters)
                        prev_state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(prev_params_ndarrays)}
                        delta_vs_prev = state_dict_delta(aggregated_state_dict, prev_state_dict)
                        l2_norm_diff_prev = compute_l2_norm_diff(aggregated_state_dict, prev_state_dict)
                    elif hasattr(self, 'initial_parameters') and self.initial_parameters is not None:
                        init_params_ndarrays = fl.common.parameters_to_ndarrays(self.initial_parameters)
                        init_state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(init_params_ndarrays)}
                        delta_vs_prev = state_dict_delta(aggregated_state_dict, init_state_dict)
                        l2_norm_diff_prev = compute_l2_norm_diff(aggregated_state_dict, init_state_dict)
                    
                    print(f"SERVER round {server_round} aggregated checksum: {aggregated_checksum:.6f}; delta vs previous: {delta_vs_prev:.6f}")
                    print(f"📊 SERVER round {server_round}: L2 norm of parameter delta (aggregated vs previous): {l2_norm_diff_prev:.6f}")
                    logger.info(f"Round {server_round}: L2 norm of parameter delta = {l2_norm_diff_prev:.6f}")
                    
                    if l2_norm_diff_prev < 1e-6:
                        logger.error(f"❌ SERVER round {server_round}: CRITICAL - L2 norm difference is near zero ({l2_norm_diff_prev:.2e})!")
                        logger.error(f"   This means FedAvg did NOT update global weights!")
                    else:
                        print(f"✅ SERVER round {server_round}: L2 norm difference confirms FedAvg updated global weights")
                    
                    # B5: Ensure aggregated parameters are actually applied
                    # Flower's FedAvg automatically updates self.parameters after aggregate_fit
                    # We also store in final_parameters for final model update
                    self.final_parameters = aggregated  # save on server
                    # For early stopping: initialize best_parameters on first round if not set
                    if self.early_stopping_patience is not None and self.best_parameters is None:
                        self.best_parameters = aggregated
                        print(f"✓ Round {server_round}: Initialized best_parameters with first aggregated parameters")
                    # B5: Explicitly update strategy's parameters (FedAvg should do this, but ensure it)
                    if hasattr(self, 'parameters'):
                        self.parameters = aggregated
                    
                    # DEBUG: Load aggregated parameters into global model to verify
                    if self.global_model is not None:
                        try:
                            model_keys = list(self.global_model.state_dict().keys())
                            model_device = next(self.global_model.parameters()).device
                            global_state_dict_before = {k: v.clone() for k, v in self.global_model.state_dict().items()}
                            
                            # Load aggregated parameters
                            state_dict = {}
                            for i, key in enumerate(model_keys):
                                if i < len(aggregated_params_ndarrays):
                                    state_dict[key] = torch.tensor(aggregated_params_ndarrays[i], 
                                                                   dtype=self.global_model.state_dict()[key].dtype).to(model_device)
                            self.global_model.load_state_dict(state_dict, strict=True)
                            
                            # Verify global model was updated
                            global_state_dict_after = {k: v.clone() for k, v in self.global_model.state_dict().items()}
                            global_l2_diff = compute_l2_norm_diff(global_state_dict_after, global_state_dict_before)
                            print(f"🔍 SERVER round {server_round}: Global model L2 norm difference after load_state_dict: {global_l2_diff:.6f}")
                            
                            if global_l2_diff < 1e-6:
                                logger.error(f"❌ SERVER round {server_round}: CRITICAL - Global model not updated by load_state_dict!")
                            else:
                                print(f"✅ SERVER round {server_round}: Global model successfully updated with aggregated parameters")
                        except Exception as e:
                            logger.warning(f"⚠️ SERVER round {server_round}: Failed to update global model: {e}")
                    
                    print(f"✓ Round {server_round}: Parameters aggregated and saved")
                    
                    # B5: Verify parameters changed
                    if delta_vs_prev < 1e-8:
                        logger.warning(f"SERVER round {server_round}: WARNING - delta_vs_prev is ~0 ({delta_vs_prev:.2e}), aggregation may not have updated weights!")
                    
                    # C: Prediction sanity check - compute predictions on fixed batch
                    if self.global_model is not None and self.val_loader is not None and delta_vs_prev > 1e-8:
                        try:
                            # Get device from model's first parameter
                            model_device = next(self.global_model.parameters()).device
                            
                            # Load aggregated parameters into global model
                            aggregated_params_ndarrays = fl.common.parameters_to_ndarrays(aggregated)
                            state_dict = {}
                            model_keys = list(self.global_model.state_dict().keys())
                            for i, key in enumerate(model_keys):
                                if i < len(aggregated_params_ndarrays):
                                    state_dict[key] = torch.tensor(aggregated_params_ndarrays[i], 
                                                                   dtype=self.global_model.state_dict()[key].dtype).to(model_device)
                            self.global_model.load_state_dict(state_dict, strict=True)
                            self.global_model.eval()
                            
                            # Get first batch from validation set
                            val_iter = iter(self.val_loader)
                            features, speaker_embeddings, labels, meeting_ids = next(val_iter)
                            features = features.to(model_device)
                            speaker_embeddings = speaker_embeddings.to(model_device).float()
                            
                            with torch.no_grad():
                                outputs = self.global_model(features, speaker_embeddings)
                                outputs = outputs.reshape(-1, outputs.shape[-1])
                                predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
                                logits_mean = outputs.mean().item()
                                logits_std = outputs.std().item()
                                unique_preds = np.unique(predictions)
                            
                            # Store predictions for this round
                            self.round_predictions[server_round] = {
                                'predictions': predictions,
                                'unique_preds': unique_preds,
                                'logits_mean': logits_mean,
                                'logits_std': logits_std
                            }
                            
                            # Compare with previous round
                            if server_round > 1 and (server_round - 1) in self.round_predictions:
                                prev_preds = self.round_predictions[server_round - 1]['predictions']
                                preds_changed = not np.array_equal(predictions, prev_preds)
                                unique_changed = not np.array_equal(unique_preds, self.round_predictions[server_round - 1]['unique_preds'])
                                
                                print(f"C: Round {server_round} prediction sanity check:")
                                print(f"  Unique predictions: {unique_preds}")
                                print(f"  Logits stats: mean={logits_mean:.4f}, std={logits_std:.4f}")
                                print(f"  Predictions changed vs round {server_round-1}: {preds_changed}")
                                print(f"  Unique predictions changed: {unique_changed}")
                                
                                if not preds_changed and delta_vs_prev > 1e-6:
                                    logger.warning(f"C: Round {server_round}: WARNING - weights changed (delta={delta_vs_prev:.6f}) but predictions identical! Model output head might be frozen or labels mismatch.")
                        except Exception as e:
                            logger.warning(f"C: Round {server_round}: Prediction sanity check failed: {e}")
                else:
                    print(f"⚠ Round {server_round}: No parameters aggregated")
                return aggregated, metrics
            
            def aggregate_evaluate(self, server_round, results, failures):
                """Override aggregate_evaluate to prevent ZeroDivisionError when all clients return 0 examples.
                
                ROBUSTNESS FIX: Handle case when num_total_evaluation_examples == 0.
                EARLY STOPPING: Check validation metrics and stop if no improvement.
                """
                print(f"✓ Round {server_round}: aggregate_evaluate called with {len(results)} results, {len(failures)} failures")
                
                # Protection against division by zero: check if we have valid results
                if not results:
                    logger.error(f"Round {server_round}: no evaluation results (all clients failed). Skipping aggregation.")
                    # FIX: Return (loss, metrics) tuple - Flower expects 2 values, not None
                    return float('nan'), {"skipped": 1, "reason": "no_results", "num_examples": 0}
                
                # Log each client's num_examples in evaluate
                for cid, eval_res in results:
                    num_examples = eval_res.num_examples
                    print(f"SERVER round {server_round}: CLIENT {cid} num_examples={num_examples} (evaluate)")
                    logger.info(f"Round {server_round}: Client {cid} returned {num_examples} examples in evaluate")
                
                # Check total evaluation examples
                total_examples = sum(eval_res.num_examples for _, eval_res in results)
                print(f"SERVER round {server_round}: Total evaluation examples: {total_examples}")
                
                if total_examples == 0:
                    logger.warning(f"⚠️ Round {server_round}: All clients returned 0 evaluation examples!")
                    logger.warning(f"   This may indicate empty validation datasets. Returning safe defaults.")
                    # ROBUSTNESS FIX: Return safe aggregated result - Flower expects (loss, metrics) tuple
                    # Return safe aggregated result: (loss, metrics) - only 2 values, not 3
                    return float('nan'), {"val_loss": float('nan'), "der": float('nan'), "all_clients_empty": True, "num_examples": 0}
                
                # Use parent's aggregate_evaluate (which uses weighted_loss_avg)
                # This is safe now because total_examples > 0
                loss, metrics = super().aggregate_evaluate(server_round, results, failures)
                
                # Early stopping logic (only if enabled)
                if self.early_stopping_patience is not None and not self.should_stop:
                    # Extract validation metrics
                    val_loss = loss if not (np.isnan(loss) or np.isinf(loss)) else float('nan')
                    val_der = metrics.get('der', float('nan')) if isinstance(metrics, dict) else float('nan')
                    
                    # Early stopping based on DER (lower is better) - primary metric
                    # Fallback to loss if DER is not available
                    if not (np.isnan(val_der) or np.isinf(val_der)):
                        # Primary: DER-based early stopping
                        improvement = self.best_val_der - val_der
                        if improvement > self.early_stopping_min_delta:
                            # Improvement detected
                            self.best_val_der = val_der
                            self.best_val_loss = val_loss if not (np.isnan(val_loss) or np.isinf(val_loss)) else self.best_val_loss
                            self.patience_counter = 0
                            self.best_round = server_round
                            # Save best parameters from current round's aggregation (final_parameters was updated in aggregate_fit)
                            # Note: aggregate_fit is called before aggregate_evaluate, so final_parameters contains current round's aggregated params
                            if hasattr(self, 'final_parameters') and self.final_parameters is not None:
                                self.best_parameters = self.final_parameters
                                print(f" ✅ Round {server_round}: Validation improved (DER)! New best val_der: {val_der:.4f}, val_loss: {val_loss:.4f}")
                                print(f" ✅ Round {server_round}: Saved best parameters")
                            else:
                                print(f" ✅ Round {server_round}: Validation improved (DER)! New best val_der: {val_der:.4f}, val_loss: {val_loss:.4f}")
                                print(f" ⚠️  Round {server_round}: Warning - final_parameters is None, cannot save best parameters")
                        else:
                            self.patience_counter += 1
                            print(f" ⚠️  Round {server_round}: No improvement (DER) for {self.patience_counter}/{self.early_stopping_patience} rounds (best DER: {self.best_val_der:.4f}, current: {val_der:.4f})")
                    elif not (np.isnan(val_loss) or np.isinf(val_loss)):
                        # Fallback: Loss-based early stopping if DER is invalid
                        improvement = self.best_val_loss - val_loss
                        if improvement > self.early_stopping_min_delta:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.best_round = server_round
                            if hasattr(self, 'final_parameters') and self.final_parameters is not None:
                                self.best_parameters = self.final_parameters
                                print(f" ✅ Round {server_round}: Validation improved (loss)! New best val_loss: {val_loss:.4f}")
                                print(f" ✅ Round {server_round}: Saved best parameters")
                            else:
                                print(f" ✅ Round {server_round}: Validation improved (loss)! New best val_loss: {val_loss:.4f}")
                                print(f" ⚠️  Round {server_round}: Warning - final_parameters is None, cannot save best parameters")
                        else:
                            self.patience_counter += 1
                            print(f" ⚠️  Round {server_round}: No improvement (loss) for {self.patience_counter}/{self.early_stopping_patience} rounds (best loss: {self.best_val_loss:.4f}, current: {val_loss:.4f})")
                    else:
                        # Both metrics are invalid - increment patience but don't stop
                        self.patience_counter += 1
                        logger.warning(f"Round {server_round}: Invalid validation metrics (NaN/Inf) - No improvement for {self.patience_counter}/{self.early_stopping_patience} rounds")
                    
                    # Check if early stopping should be triggered
                    if self.patience_counter >= self.early_stopping_patience:
                        self.should_stop = True
                        print(f"\n{'='*80}")
                        print(f" 🛑 Round {server_round}: Early stopping triggered! No improvement for {self.early_stopping_patience} rounds.")
                        print(f" Best validation DER: {self.best_val_der:.4f}, Best validation loss: {self.best_val_loss:.4f} (at round {self.best_round})")
                        print(f"{'='*80}\n")
                        # Restore best parameters
                        if self.best_parameters is not None:
                            self.final_parameters = self.best_parameters
                            print(f" ✅ Restored best model parameters from round {self.best_round}")
                        else:
                            logger.warning(f"⚠️  Warning: best_parameters is None, cannot restore best model")
                
                return loss, metrics
            
            def configure_fit(self, server_round, parameters, client_manager):
                print(f"✓ Round {server_round}: configure_fit called")
                
                # Check if early stopping was triggered
                if self.should_stop:
                    logger.warning(f"Round {server_round}: Early stopping was triggered. Skipping fit configuration.")
                    # Return empty configuration to prevent training
                    return {}
                
                # A1: Log checksum BEFORE distributing parameters each round
                if parameters is not None:
                    params_ndarrays = fl.common.parameters_to_ndarrays(parameters)
                    state_dict = {f"param_{i}": torch.tensor(p) for i, p in enumerate(params_ndarrays)}
                    checksum = state_dict_checksum(state_dict)
                    print(f"SERVER round {server_round} global checksum (before distributing): {checksum:.6f}")
                return super().configure_fit(server_round, parameters, client_manager)
            
            def configure_evaluate(self, server_round, parameters, client_manager):
                """Override configure_evaluate to skip evaluation if early stopping was triggered."""
                if self.should_stop:
                    logger.warning(f"Round {server_round}: Early stopping was triggered. Skipping evaluation configuration.")
                    return {}
                return super().configure_evaluate(server_round, parameters, client_manager)
        
        # FL Strategy Configuration: Support both sequential and parallel modes
        # Parallel mode: multiple clients per round (requires memory optimizations)
        # Sequential mode: 1 client per round (current default for memory safety)
        if clients_per_round is not None and clients_per_round > 1:
            # Parallel mode: train multiple clients concurrently
            if clients_per_round > num_clients:
                logger.warning(f"clients_per_round ({clients_per_round}) > num_clients ({num_clients}), using {num_clients}")
                clients_per_round = num_clients
            fraction_fit = clients_per_round / num_clients
            fraction_evaluate = clients_per_round / num_clients
            min_fit_clients = clients_per_round
            min_evaluate_clients = clients_per_round
            print(f"FL Strategy Configuration (PARALLEL MODE):")
            print(f"  - min_available_clients: {num_clients} (all clients must be available)")
            print(f"  - min_fit_clients: {min_fit_clients} (train {clients_per_round} clients per round in parallel)")
            print(f"  - min_evaluate_clients: {min_evaluate_clients} (evaluate {clients_per_round} clients per round)")
            print(f"  - fraction_fit: {fraction_fit:.2f} (train {clients_per_round}/{num_clients} clients per round)")
            print(f"  - fraction_evaluate: {fraction_evaluate:.2f} (evaluate {clients_per_round}/{num_clients} clients per round)")
        else:
            # Sequential mode: 1 client per round (memory-safe default)
            fraction_fit = 1.0 / num_clients  # For 2 clients = 0.5 (train 1 client per round)
            fraction_evaluate = 1.0 / num_clients
            min_fit_clients = 1  # Only need 1 client per round
            min_evaluate_clients = 1  # Only need 1 client per round
            print(f"FL Strategy Configuration (SEQUENTIAL MODE - memory-optimized):")
            print(f"  - min_available_clients: {num_clients} (all clients must be available)")
            print(f"  - min_fit_clients: {min_fit_clients} (train 1 client per round)")
            print(f"  - min_evaluate_clients: {min_evaluate_clients} (evaluate 1 client per round)")
            print(f"  - fraction_fit: {fraction_fit:.2f} (train 1/{num_clients} clients per round)")
            print(f"  - fraction_evaluate: {fraction_evaluate:.2f} (evaluate 1/{num_clients} clients per round)")
        
        strategy = SaveFinalParams(
            global_model=model,  # C: Pass global model for prediction sanity check
            val_loader=val_loader,  # C: Pass validation loader for prediction sanity check
            early_stopping_patience=early_stopping_patience,  # Early stopping patience
            early_stopping_min_delta=early_stopping_min_delta,  # Early stopping min delta
            min_available_clients=num_clients,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            # E: Pass server_round and compute_der=False by default (DER only every 5 rounds)
            # For early stopping, we need DER every round to check improvement
            on_fit_config_fn=lambda server_round: {
                "epochs": epochs, 
                "debug_mode": debug_mode, 
                "debug_max_batches": debug_max_batches,
                "server_round": server_round,
                "lr": learning_rate,  # Pass learning rate to clients
                "compute_der": False,  # E: DER disabled by default
                "der_round_interval": 5  # E: Compute DER every 5 rounds
            },
            on_evaluate_config_fn=lambda server_round: {
                "epochs": 1,
                "server_round": server_round,
                # Enable DER computation every round if early stopping is enabled
                "compute_der": early_stopping_patience is not None,  # Enable DER if early stopping is enabled
                "der_round_interval": 1 if early_stopping_patience is not None else 5  # Compute DER every round if early stopping enabled
            },
            initial_parameters=fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in model.state_dict().items()]
            ),
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
        
        # Calculate GPU/CPU resources based on device mode
        if device.type == 'cpu':
            # CPU-only mode: no GPUs, allocate CPUs per client
            num_gpus = 0
            gpus_per_client = 0
            # For parallel CPU clients, ensure we have enough CPUs
            total_cpus_needed = num_clients * num_cpus_per_client
            print(f"CPU-only mode: {num_cpus_per_client} CPUs per client, {total_cpus_needed} total CPUs needed")
        else:
            # GPU mode: allocate GPUs
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if clients_per_round is not None and clients_per_round > 1:
                # Parallel mode: share GPUs among clients
                gpus_per_client = max(1, num_gpus // clients_per_round) if num_gpus > 0 else 0
                print(f"Parallel GPU mode: {num_gpus} GPUs available, {gpus_per_client} GPUs per client")
            else:
                # Sequential mode: each client gets full GPU
                gpus_per_client = 1 if num_gpus > 0 else 0
                print(f"Sequential GPU mode: {num_gpus} GPUs available, {gpus_per_client} GPUs per client (full GPU per client)")
        
        print(f"Resource allocation: GPUs={num_gpus}, GPUs/client={gpus_per_client}, CPUs/client={num_cpus_per_client}")
        
        # Start simulation and get final parameters
        print("\n==================== STARTING FEDERATED LEARNING ====================\n")
        if early_stopping_patience is not None:
            print(f"Early stopping ENABLED:")
            print(f"  - Patience: {early_stopping_patience} rounds")
            print(f"  - Min delta: {early_stopping_min_delta}")
            print(f"  - Metric: Validation DER (fallback to validation loss if DER unavailable)")
            print(f"  - Training will stop if no improvement for {early_stopping_patience} consecutive rounds")
            print("")
        else:
            print(f"Early stopping DISABLED - training will run for all {num_rounds} rounds")
            print("")
        
        # ROBUSTNESS FIX: Mac has 2GB limit for object store, reduce for Mac compatibility
        import platform
        is_mac = platform.system() == "Darwin"
        if device.type == 'cpu':
            # Mac limit: 2GB, Linux: 5GB
            object_store_memory = 2_000_000_000 if is_mac else 5_000_000_000
        else:
            object_store_memory = None
        
        # Memory-optimized Ray configuration to prevent OOM
        print(f"Ray memory configuration:")
        print(f"  - Object store memory: {object_store_memory / 1e9:.1f} GB" if object_store_memory else "  - Object store memory: auto")
        print(f"  - Memory usage threshold: {os.environ.get('RAY_memory_usage_threshold', '0.90')}")
        print(f"  - Memory monitor refresh: {os.environ.get('RAY_memory_monitor_refresh_ms', '1000')} ms")
        
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            ray_init_args={
                "num_cpus": num_clients * num_cpus_per_client if device.type == 'cpu' else num_clients,
                "num_gpus": num_gpus,
                "include_dashboard": False,
                "ignore_reinit_error": True,
                # Memory optimizations to prevent OOM
                # Set object store memory limit (2GB for Mac, 5GB for Linux)
                "object_store_memory": object_store_memory,
            },
            client_resources={
                "num_cpus": num_cpus_per_client,
                "num_gpus": gpus_per_client
            }
        )
        
        # Get final parameters from the strategy after simulation
        # Try different ways to get the final parameters
        final_parameters = None
        
        # Debug: Print information about what we received
        print(f"\n=== DEBUG: Simulation Results ===")
        print(f"History type: {type(history)}")
        print(f"History attributes: {dir(history)}")
        if hasattr(history, 'losses_distributed'):
            print(f"History losses: {history.losses_distributed}")
        if hasattr(history, 'metrics_distributed'):
            print(f"History metrics: {history.metrics_distributed}")
        print(f"Strategy type: {type(strategy)}")
        print(f"Strategy attributes: {dir(strategy)}")
        if hasattr(strategy, 'parameters'):
            print(f"Strategy parameters: {strategy.parameters is not None}")
        if hasattr(strategy, 'initial_parameters'):
            print(f"Strategy initial_parameters: {strategy.initial_parameters is not None}")
        if hasattr(strategy, 'final_parameters'):
            print(f"Strategy final_parameters: {strategy.final_parameters is not None}")
        if hasattr(strategy, 'final_parameters') and strategy.final_parameters is not None:
            print(f"Strategy final_parameters type: {type(strategy.final_parameters)}")
        print(f"=== END DEBUG ===\n")
        
        # Method 1: Try to get from our custom strategy's final_parameters
        if hasattr(strategy, 'final_parameters') and strategy.final_parameters is not None:
            final_parameters = strategy.final_parameters
            print("✓ Got final parameters from custom strategy's final_parameters")
        
        # Method 2: Try to get from strategy.parameters (fallback)
        elif hasattr(strategy, 'parameters') and strategy.parameters is not None:
            final_parameters = strategy.parameters
            print("✓ Got final parameters from strategy.parameters")
        
        # Method 3: Check if we can extract from history
        elif hasattr(history, 'losses_distributed') and len(history.losses_distributed) > 0:
            print("⚠ History contains training data but no parameters accessible")
            print("   This may indicate the simulation completed but parameters are not accessible")
        
        # Check if simulation crashed before first aggregation (OOM protection)
        if final_parameters is None:
            print("\n" + "="*80)
            print("⚠️  SIMULATION CRASHED BEFORE FIRST AGGREGATION")
            print("="*80)
            print("Most likely cause: Ray OOM (Out of Memory)")
            print("The simulation did not reach aggregate_fit, which means:")
            print("  - Ray worker was killed due to memory exhaustion")
            print("  - Check node memory usage (current: likely >119GB/125GB)")
            print("  - Consider:")
            print("    * Reducing batch_size")
            print("    * Reducing max_sequence_length")
            print("    * Reducing num_clients or using sequential training (already enabled)")
            print("    * Reducing chunk_size in data processing")
            print("="*80)
            print("\nSkipping model update - using original model for final evaluation.")
            print("="*80 + "\n")
        # Update the model with final parameters from federated learning
        elif final_parameters is not None:
            print("\n==================== UPDATING MODEL WITH FINAL PARAMETERS ====================\n")
            # Store original parameters for comparison - use deep copy with detach().cpu().clone()
            original_params = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
            
            # Convert final parameters back to model state dict
            final_state_dict = {}
            
            try:
                # Convert Flower parameters back to numpy arrays
                final_params_numpy = fl.common.parameters_to_ndarrays(final_parameters)
                
                # Map parameter names to their indices
                model_state_dict = model.state_dict()
                param_keys = list(model_state_dict.keys())
                
                for i, key in enumerate(param_keys):
                    if i < len(final_params_numpy):
                        param_tensor = torch.tensor(final_params_numpy[i], dtype=model_state_dict[key].dtype)
                        original_shape = model_state_dict[key].shape
                        
                        # FIX: Skip combine_adapter if sizes don't match (it's created dynamically)
                        # combine_adapter dimensions depend on num_speakers + hidden_dim, which can vary
                        if key == 'combine_adapter.weight' or key == 'combine_adapter.bias':
                            if param_tensor.shape != original_shape:
                                print(f"Warning: Skipping {key} due to size mismatch (checkpoint: {param_tensor.shape}, model: {original_shape}). "
                                      f"This is normal - combine_adapter is created dynamically based on number of speakers.")
                                # Don't add it to final_state_dict - let the model keep its current dynamically created value
                                continue
                            # If shapes match, we can still load it
                            final_state_dict[key] = param_tensor
                        else:
                            final_state_dict[key] = param_tensor
                    else:
                        print(f"Warning: Parameter {key} not found in final parameters")
                        # Only add to final_state_dict if it's not combine_adapter (which is dynamic)
                        if not (key == 'combine_adapter.weight' or key == 'combine_adapter.bias'):
                            final_state_dict[key] = model_state_dict[key].clone()
                
                # Filter out combine_adapter from final_state_dict if it has size mismatches
                # This ensures we don't try to load incompatible combine_adapter parameters
                filtered_state_dict = {}
                for key, value in final_state_dict.items():
                    if key.startswith('combine_adapter'):
                        # Only include if it exists in model and shapes match
                        if key in model_state_dict and value.shape == model_state_dict[key].shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"Info: Excluding {key} from state_dict (dynamic parameter, will be recreated during forward pass)")
                    else:
                        filtered_state_dict[key] = value
                
                # Load final parameters into model with strict=False to allow skipping combine_adapter
                # Ensure parameters are on the correct device
                device = next(model.parameters()).device
                filtered_state_dict_device = {k: v.to(device) if isinstance(v, torch.Tensor) else torch.tensor(v).to(device) 
                                             for k, v in filtered_state_dict.items()}
                model.load_state_dict(filtered_state_dict_device, strict=False)
                # Ensure model is on correct device (should already be, but double-check)
                model.to(device)
                print("Model updated with final federated learning parameters (explicitly applied via load_state_dict)")
                
                # Verify that parameters actually changed - compare with deep copied original
                param_changed = False
                changed_count = 0
                total_params = len(model.state_dict())
                
                for key in model.state_dict():
                    # Move current param to CPU for comparison with original (which is on CPU)
                    current_param = model.state_dict()[key].detach().cpu()
                    if not torch.equal(original_params[key], current_param):
                        param_changed = True
                        changed_count += 1
                
                if param_changed:
                    change_percentage = (changed_count / total_params) * 100
                    print(f"✓ Model parameters successfully updated - {changed_count}/{total_params} parameters changed ({change_percentage:.1f}%)")
                else:
                    print("⚠ Warning: Model parameters appear unchanged - this may indicate an issue")
                    
            except Exception as e:
                print(f"Error updating model parameters: {e}")
                raise RuntimeError(f"Failed to update model with final parameters: {e}")
            # Note: If final_parameters is None, we already logged the error above and skip model update
        
        # Final evaluation on test set with UPDATED model (server-side evaluation)
        print("\n==================== SERVER-SIDE TEST EVALUATION STARTED ====================\n")
        
        if len(test_dataset) == 0:
            print("⚠️  WARNING: Test dataset is empty. Skipping test evaluation.")
            test_loss = float('nan')
            der = float('nan')
        else:
            # Create DataLoader for test dataset
            from data_processing import collate_fn_overlapping_speech
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn_overlapping_speech,
                num_workers=0,  # Disable multiprocessing for server-side evaluation
                pin_memory=False
            )
            
        model.eval()
        test_loss = 0.0
        # Group predictions by meeting_id for proper DER calculation
        pred_by_rec = defaultdict(list)
        lab_by_rec = defaultdict(list)
        
        # Debug: Check model parameters
        print(f"[FINAL TEST DEBUG] Model parameters changed: {strategy.final_parameters is not None}")
        if strategy.final_parameters:
            print(f"[FINAL TEST DEBUG] Final parameters shape: {len(strategy.final_parameters.tensors)}")
        
        # Debug: Check test dataset
            print(f"[FINAL TEST DEBUG] Test dataset size: {len(test_dataset)}")
            print(f"[FINAL TEST DEBUG] Test dataset speaker_id_list: {test_dataset.get_speaker_id_list() if hasattr(test_dataset, 'get_speaker_id_list') else 'N/A'}")
        
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
                
                # Save original shape before reshape (labels can be 1D or 2D)
                labels_original_shape = labels.shape
                labels_original_ndim = labels.ndim
                
                # Get outputs from model
                outputs = model(features, speaker_embeddings)
                
                # Handle both 1D and 2D labels (aligned with centralized version)
                if labels.ndim == 2:
                    # Labels are (batch_size, seq_len) - standard case
                    batch_size, seq_len = labels.shape
                    outputs = outputs.reshape(-1, outputs.shape[-1])
                    labels_flat = labels.reshape(-1)
                elif labels.ndim == 1:
                    # Labels are already flattened (T,) - from collate/processing
                    # Infer batch_size from features, seq_len from labels length
                    batch_size = features.shape[0] if features.ndim >= 2 else 1
                    seq_len = labels.shape[0] // batch_size if batch_size > 0 else labels.shape[0]
                    # Ensure outputs are flattened to match labels
                    if outputs.ndim == 3:
                        outputs = outputs.reshape(-1, outputs.shape[-1])
                    elif outputs.ndim == 2:
                        # Already (T, num_classes) or (batch_size*seq_len, num_classes)
                        pass
                    labels_flat = labels
                else:
                    raise ValueError(f"Unexpected labels ndim={labels.ndim}, shape={labels.shape}")
                
                # Check for valid frames: skip batch if all labels are padding (-100)
                valid = (labels_flat != -100)
                if valid.sum() == 0:
                    logger.warning(f"Test batch {batch_idx}: All labels are -100 (padding). Skipping batch safely.")
                    continue
                
                # Use ignore_index=-100 to skip padded frames (same as in training)
                loss = nn.CrossEntropyLoss(ignore_index=-100)(outputs, labels_flat)
                test_loss += loss.item()
                predictions_flat = torch.argmax(outputs, dim=-1)
                
                # Sanity-check (only for first 2 batches to avoid spam)
                if batch_idx < 2:
                    valid_ratio = valid.float().mean().item() if valid.numel() > 0 else 0.0
                    uniq_labels = torch.unique(labels_flat[valid]).detach().cpu().tolist() if valid.any() else []
                    uniq_preds = torch.unique(predictions_flat[valid]).detach().cpu().tolist() if valid.any() else []
                    logger.info(
                        f"[FINAL TEST DEBUG] Batch {batch_idx}: labels.ndim={labels_original_ndim}, "
                        f"labels.shape={tuple(labels_original_shape)}, "
                        f"outputs.shape={tuple(outputs.shape)}, predictions_flat.shape={tuple(predictions_flat.shape)}, "
                        f"valid_ratio={valid_ratio:.3f}, "
                        f"unique_labels={uniq_labels}, unique_preds={uniq_preds}"
                    )
                
                # Debug: Check predictions (only for first batch)
                if batch_idx == 0:
                    unique_preds = torch.unique(predictions_flat).cpu().numpy()
                    unique_labels = torch.unique(labels_flat).cpu().numpy()
                    print(f"[FINAL TEST DEBUG] Batch {batch_idx}: unique predictions: {unique_preds}")
                    print(f"[FINAL TEST DEBUG] Batch {batch_idx}: unique labels: {unique_labels}")
                    print(f"[FINAL TEST DEBUG] Batch {batch_idx}: loss: {loss.item():.4f}")
                    first_pred = predictions_flat[valid][0].item() if valid.any() else predictions_flat[0].item()
                    print(f"[FINAL TEST DEBUG] First prediction: {first_pred}")
                
                # Group predictions by meeting_id
                predictions_np = predictions_flat.cpu().numpy()
                labels_np = labels_flat.cpu().numpy()
                
                # Create per-frame meeting_id array by repeating for each frame in the sequence
                # meeting_ids is a 1D array (one per sample), not per-frame
                meeting_ids_flat = []
                for i in range(batch_size):
                    meeting_id = meeting_ids[i] if isinstance(meeting_ids, (list, np.ndarray)) else meeting_ids
                    # Repeat meeting_id for each frame in this sequence
                    meeting_ids_flat.extend([meeting_id] * seq_len)
                meeting_ids_flat = np.array(meeting_ids_flat)
                
                # Ensure meeting_ids_flat matches the length of predictions_np and labels_np
                if len(meeting_ids_flat) != len(predictions_np):
                    # Adjust if lengths don't match (can happen with variable-length sequences)
                    min_len = min(len(meeting_ids_flat), len(predictions_np))
                    meeting_ids_flat = meeting_ids_flat[:min_len]
                    predictions_np = predictions_np[:min_len]
                    labels_np = labels_np[:min_len]
                
                for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                    if meeting_id is not None and label != -100:  # Skip padded frames
                        pred_by_rec[meeting_id].append(pred)
                        lab_by_rec[meeting_id].append(label)
        
            # Calculate DER per recording and aggregate
            ders = {}
            # Get speaker_id_list from test dataset for consistency
            test_speaker_id_list = test_dataset.get_speaker_id_list() if hasattr(test_dataset, 'get_speaker_id_list') else None
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
            test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else float('nan')
            der = np.mean(list(ders.values())) if ders else float('nan')
            
            print(f"Test recordings processed: {len(ders)}")
            print(f"DER per recording: {ders}")
            print(f"Average DER: {der}")
            print(f"\n==================== SERVER-SIDE TEST EVALUATION FINISHED ====================\n")
            print(f"Final Test Loss: {test_loss:.4f}")
            print(f"Final DER: {der:.4f}")

        # === EXPORT DIARIZATION RESULTS TO RTTM FORMAT ===
        print("\n===== EXPORTING DIARIZATION RESULTS =====")
        
        # Create experiment tag for export directory
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dt_str_human = datetime.now().strftime("%Y-%m-%d-%H-%M")
        if use_all_data:
            exp_tag = f"exp_all_{epochs}epochs_{num_rounds}rounds_{num_clients}clients_{dt_str_human}"
        else:
            exp_tag = f"exp_{train_size}size_{epochs}epochs_{num_rounds}rounds_{num_clients}clients_{dt_str_human}"
        
        try:
            from diarization_export import export_diarization_results
            
            # Create dictionary with speaker_id_lists for each recording
            speaker_id_lists = {}
            for rec_id in pred_by_rec:
                # Use meeting-specific slot speakers from test_dataset if available
                if hasattr(test_dataset, 'meeting_to_slot_speakers') and rec_id in test_dataset.meeting_to_slot_speakers:
                    speaker_id_lists[rec_id] = test_dataset.meeting_to_slot_speakers[rec_id]
                else:
                    # Use general speaker list
                    speaker_id_lists[rec_id] = test_speaker_id_list
            
            # Export results to various formats
            # Create diarization export directory with experiment parameters
            diarization_export_dir = os.path.join("experiments", "out_artifacts", "diarization_export", exp_tag)
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
        # Artifact directories for logs and plots
        artifact_logs_dir = os.path.join("experiments", "out_artifacts", "logs", exp_tag)
        artifact_plots_dir = os.path.join("experiments", "out_artifacts", "plots", exp_tag)
        os.makedirs(artifact_logs_dir, exist_ok=True)
        os.makedirs(artifact_plots_dir, exist_ok=True)
        # File paths for logs and metrics (simple names)
        exp_filename = "experiment.txt"
        exp_filepath = os.path.join(artifact_logs_dir, exp_filename)
        # Prepare lines for logging
        result_lines = [
            f"Experiment: {exp_tag}",
            f"Dataset size: {'ALL' if use_all_data else f'{train_size} records'}",
            f"Train records: {train_size}",
            f"Validation records: {val_size}",
            f"Test records: {test_size_val}",
            f"Num clients: {num_clients}",
            f"Num epochs: {epochs}",
            f"Num rounds: {num_rounds}",
            f"Datetime: {dt_str}",
            f"",
            f"=== TRAINING DATA STATISTICS ===",
            f"Total training samples across all clients: {total_training_samples}",
            f"Estimated total training frames across all clients: {total_training_frames}",
            f"",
        ]
        
        # Add per-client sample information
        for client_info in client_samples_info:
            result_lines.extend([
                f"Client {client_info['client_id']}:",
                f"  - Train samples: {client_info['train_samples']}",
                f"  - Validation samples: {client_info['val_samples']}",
                f"  - Total samples: {client_info['total_samples']}",
                f"  - Actual frames: {client_info['actual_frames']}",
                f""
            ])
        
        result_lines.extend([
            f"=== FINAL RESULTS ===",
            f"Final Test Loss: {test_loss:.4f}",
            f"Final DER: {der:.4f}",
            f"Model Status: {'UPDATED with FL parameters' if final_parameters is not None else 'ORIGINAL (no FL parameters)'}",
            f"Parameters Changed: {changed_count if 'changed_count' in locals() else 'N/A'}/{total_params if 'total_params' in locals() else 'N/A'} ({(changed_count/total_params*100) if 'changed_count' in locals() and 'total_params' in locals() else 'N/A':.1f}%)",
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

        # After simulation and final evaluation, plot and save metrics
        def plot_client_epoch_metrics(client_epoch_metrics):
            for cid, rounds in client_epoch_metrics.items():
                # For each client, aggregate metrics per round (use last epoch of each round)
                round_train_loss = []
                round_der = []
                round_numbers = []
                for rnd, epoch_list in rounds.items():
                    if epoch_list:
                        last_epoch = epoch_list[-1]
                        round_train_loss.append(last_epoch.get('train_loss'))
                        round_der.append(last_epoch.get('der'))
                        round_numbers.append(rnd + 1)  # Use integer round number for x-axis
                # Plot loss per round
                plt.figure(figsize=(8, 5))
                plt.plot(round_numbers, round_train_loss, marker='o', label='Train Loss')
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.title(f'Client {cid} Loss per Round')
                plt.tight_layout()
                plot_path = os.path.join(artifact_plots_dir, f"client{cid}_loss_per_round.png")
                plt.savefig(plot_path)
                plt.close()
                # Plot DER per round
                plt.figure(figsize=(8, 5))
                plt.plot(round_numbers, round_der, marker='o', label='DER')
                plt.xlabel('Round')
                plt.ylabel('DER')
                plt.title(f'Client {cid} DER per Round')
                plt.tight_layout()
                plot_path = os.path.join(artifact_plots_dir, f"client{cid}_der_per_round.png")
                plt.savefig(plot_path)
                plt.close()
            # Now, per-epoch plots are removed in favor of per-round plots only

        def plot_round_metrics(round_metrics):
            rounds = [rm['round']+1 for rm in round_metrics]
            mean_loss = [rm['mean_loss'] for rm in round_metrics]
            mean_der = [rm['mean_der'] for rm in round_metrics]
            # Plot mean loss per round
            plt.figure(figsize=(8, 5))
            plt.plot(rounds, mean_loss, marker='o', label='Mean Train Loss')
            plt.xlabel('Round')
            plt.ylabel('Loss')
            plt.title('Mean Train Loss per Round (all clients)')
            plt.tight_layout()
            plot_path = os.path.join(artifact_plots_dir, "mean_loss_per_round.png")
            plt.savefig(plot_path)
            plt.close()
            # Plot mean DER per round
            plt.figure(figsize=(8, 5))
            plt.plot(rounds, mean_der, marker='o', label='Mean DER')
            plt.xlabel('Round')
            plt.ylabel('DER')
            plt.title('Mean DER per Round (all clients)')
            plt.tight_layout()
            plot_path = os.path.join(artifact_plots_dir, "mean_der_per_round.png")
            plt.savefig(plot_path)
            plt.close()
            # Optionally: plot per-client loss/der per round
            for metric_name in ['train_loss', 'der']:
                plt.figure(figsize=(10, 6))
                for cid in range(num_clients):
                    vals = []
                    for rm in round_metrics:
                        cm = rm['client_metrics'].get(cid, {})
                        vals.append(cm.get(metric_name))
                    plt.plot(rounds, vals, marker='o', label=f'Client {cid}')
                plt.xlabel('Round')
                plt.ylabel(metric_name.replace('_', ' ').title())
                plt.title(f'{metric_name.replace("_", " ").title()} per Round (per client)')
                plt.legend()
                plt.tight_layout()
                plot_path = os.path.join(artifact_plots_dir, f"{metric_name}_per_round_per_client.png")
                plt.savefig(plot_path)
                plt.close()

        # Call plotting after experiment
        plot_client_epoch_metrics(client_epoch_metrics)
        plot_round_metrics(round_metrics)

        # === SAVE METRICS TO CSV FILES ===
        def save_metrics_to_csv(client_epoch_metrics, round_metrics, exp_tag, artifact_logs_dir):
            """Save all metrics to CSV files for detailed analysis."""
            print("\n===== SAVING METRICS TO CSV FILES =====")
            
            # 1. Detailed metrics per client per round per epoch
            detailed_metrics = []
            for cid, rounds in client_epoch_metrics.items():
                for round_num, epoch_list in rounds.items():
                    for epoch_idx, epoch_metrics in enumerate(epoch_list):
                        detailed_metrics.append({
                            'round': round_num + 1,  # Convert to 1-based indexing
                            'client_id': cid,
                            'epoch': epoch_idx + 1,
                            'train_loss': epoch_metrics.get('train_loss'),
                            'val_loss': epoch_metrics.get('val_loss'),
                            'der': epoch_metrics.get('der'),
                            'acc': epoch_metrics.get('acc')
                        })
            
            if detailed_metrics:
                detailed_df = pd.DataFrame(detailed_metrics)
                detailed_csv_path = os.path.join(artifact_logs_dir, "detailed_metrics.csv")
                detailed_df.to_csv(detailed_csv_path, index=False)
                print(f"Detailed metrics saved to: {detailed_csv_path}")
                print(f"Shape: {detailed_df.shape}")
                print(f"Columns: {list(detailed_df.columns)}")
            
            # 2. Aggregated metrics per round (mean/std across clients)
            if round_metrics:
                round_summary = []
                for rm in round_metrics:
                    round_num = rm['round'] + 1
                    client_metrics = rm['client_metrics']
                    
                    # Extract metrics for this round
                    train_losses = [cm.get('train_loss') for cm in client_metrics.values() if 'train_loss' in cm]
                    val_losses = [cm.get('val_loss') for cm in client_metrics.values() if 'val_loss' in cm]
                    ders = [cm.get('der') for cm in client_metrics.values() if 'der' in cm]
                    
                    # Calculate statistics
                    round_summary.append({
                        'round': round_num,
                        'mean_train_loss': np.mean(train_losses) if train_losses else None,
                        'std_train_loss': np.std(train_losses) if len(train_losses) > 1 else None,
                        'mean_val_loss': np.mean(val_losses) if val_losses else None,
                        'std_val_loss': np.std(val_losses) if len(val_losses) > 1 else None,
                        'mean_der': np.mean(ders) if ders else None,
                        'std_der': np.std(ders) if len(ders) > 1 else None,
                        'num_clients': len(client_metrics)
                    })
                
                round_summary_df = pd.DataFrame(round_summary)
                round_summary_csv_path = os.path.join(artifact_logs_dir, "round_summary.csv")
                round_summary_df.to_csv(round_summary_csv_path, index=False)
                print(f"Round summary saved to: {round_summary_csv_path}")
                print(f"Shape: {round_summary_df.shape}")
            
            # 3. Final experiment results
            final_results = [{
                'experiment_tag': exp_tag,
                'dataset_size': 'ALL' if use_all_data else f'{train_size}',
                'train_records': train_size,
                'val_records': val_size,
                'test_records': test_size_val,
                'num_clients': num_clients,
                'num_epochs': epochs,
                'num_rounds': num_rounds,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'final_test_loss': test_loss,
                'final_der': der,
                'device_used': str(device),
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }]
            
            final_results_df = pd.DataFrame(final_results)
            final_results_csv_path = os.path.join(artifact_logs_dir, "experiment_results.csv")
            final_results_df.to_csv(final_results_csv_path, index=False)
            print(f"Final results saved to: {final_results_csv_path}")
            
            # 4. Client performance comparison (last round metrics)
            if round_metrics:
                last_round = round_metrics[-1]
                client_comparison = []
                for cid, metrics in last_round['client_metrics'].items():
                    client_comparison.append({
                        'client_id': cid,
                        'final_round': len(round_metrics),
                        'final_train_loss': metrics.get('train_loss'),
                        'final_val_loss': metrics.get('val_loss'),
                        'final_der': metrics.get('der'),
                        'final_acc': metrics.get('acc')
                    })
                
                client_comparison_df = pd.DataFrame(client_comparison)
                client_comparison_csv_path = os.path.join(artifact_logs_dir, "client_comparison.csv")
                client_comparison_df.to_csv(client_comparison_csv_path, index=False)
                print(f"Client comparison saved to: {client_comparison_csv_path}")
            
            print("All CSV files saved successfully!")
            return detailed_df if detailed_metrics else None

        # Save metrics to CSV
        detailed_df = save_metrics_to_csv(client_epoch_metrics, round_metrics, exp_tag, artifact_logs_dir)

        # === ADDITIONAL ANALYSIS CSV FILES ===
        def create_epoch_progress_csv(client_epoch_metrics, exp_tag, artifact_logs_dir):
            """Create CSV file showing training progress across epochs for each client."""
            print("\n===== CREATING EPOCH PROGRESS ANALYSIS =====")
            
            epoch_progress = []
            for cid, rounds in client_epoch_metrics.items():
                for round_num, epoch_list in rounds.items():
                    for epoch_idx, epoch_metrics in enumerate(epoch_list):
                        epoch_progress.append({
                            'client_id': cid,
                            'round': round_num + 1,
                            'epoch': epoch_idx + 1,
                            'train_loss': epoch_metrics.get('train_loss'),
                            'val_loss': epoch_metrics.get('val_loss'),
                            'der': epoch_metrics.get('der'),
                            'acc': epoch_metrics.get('acc'),
                            'round_epoch': f"R{round_num + 1}E{epoch_idx + 1}"
                        })
            
            if epoch_progress:
                epoch_progress_df = pd.DataFrame(epoch_progress)
                
                # Sort by client, round, epoch for better readability
                epoch_progress_df = epoch_progress_df.sort_values(['client_id', 'round', 'epoch'])
                
                epoch_progress_csv_path = os.path.join(artifact_logs_dir, "epoch_progress.csv")
                epoch_progress_df.to_csv(epoch_progress_csv_path, index=False)
                print(f"Epoch progress analysis saved to: {epoch_progress_csv_path}")
                print(f"Shape: {epoch_progress_df.shape}")
                
                # Create pivot table for easier analysis
                try:
                    # Pivot for train loss
                    train_loss_pivot = epoch_progress_df.pivot_table(
                        values='train_loss', 
                        index='round', 
                        columns='client_id', 
                        aggfunc='mean'
                    )
                    train_loss_pivot.to_csv(os.path.join(artifact_logs_dir, "train_loss_pivot.csv"))
                    
                    # Pivot for DER
                    der_pivot = epoch_progress_df.pivot_table(
                        values='der', 
                        index='round', 
                        columns='client_id', 
                        aggfunc='mean'
                    )
                    der_pivot.to_csv(os.path.join(artifact_logs_dir, "der_pivot.csv"))
                    
                    print("Pivot tables created for train_loss and DER")
                except Exception as e:
                    print(f"Could not create pivot tables: {e}")
                
                return epoch_progress_df
            return None

        # Create epoch progress analysis
        epoch_progress_df = create_epoch_progress_csv(client_epoch_metrics, exp_tag, artifact_logs_dir)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
        # Add any necessary cleanup code here
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        print("Process completed.")

if __name__ == "__main__":
    main() 