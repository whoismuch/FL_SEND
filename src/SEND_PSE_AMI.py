import os
import logging
import re
import sys
import builtins

# Disable numba debug output (IR - Intermediate Representation)
# This prevents verbose compilation details from appearing in logs
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Keep JIT enabled
os.environ['NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING'] = '1'  # Disable highlighting

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
    """Feedforward Sequential Memory Network layer."""
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
        memory = torch.zeros_like(h)
        for i in range(x.shape[1]):
            start_idx = max(0, i - self.stride)
            memory[:, i] = self.memory(x[:, start_idx:i+1].mean(dim=1))
        return h + memory

class SENDModel(nn.Module):
    """Speaker Embedding-aware Neural Diarization model with Power-Set Encoding."""
    def __init__(self, input_dim: int = 80, hidden_dim: int = 512, num_classes: int = 16, dropout_p: float = 0.1):
        super().__init__()
        # Speech Encoder (FSMN)
        self.speech_encoder = nn.ModuleList([
            nn.Sequential(
                FSMNLayer(input_dim if i == 0 else hidden_dim, hidden_dim, stride=2**i),
                nn.Dropout(dropout_p)
            ) for i in range(8)
        ])
        # Speaker Encoder (MLP) with Dropout after each activation
        self.speaker_encoder = nn.Sequential(
            nn.Linear(192, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_p)
        )
        # CI Scorer (Context-Independent)
        self.ci_scorer = nn.Linear(hidden_dim, 1)
        # CD Scorer (Context-Dependent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_p,
            batch_first=True
        )
        self.cd_scorer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # Post-Net (FSMN) with Dropout after each layer
        self.post_net = nn.ModuleList([
            nn.Sequential(
                FSMNLayer(hidden_dim, hidden_dim, stride=2**i),
                nn.Dropout(dropout_p)
            ) for i in range(6)
        ])
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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
        speaker_features = self.speaker_encoder(speaker_embeddings)
        # CI Scoring
        ci_scores = []
        for i in range(num_speakers):
            # Dot product between audio features and speaker embeddings
            score = torch.matmul(x, speaker_features[:, i].unsqueeze(-1)).squeeze(-1)
            ci_scores.append(score)
        ci_scores = torch.stack(ci_scores, dim=1)
        ci_scores = ci_scores.transpose(1, 2)
        # CD Scoring
        cd_scores = self.cd_scorer(x)
        # Combine CI and CD scores
        combined = torch.cat([
            ci_scores,
            cd_scores
        ], dim=2)
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
def train_model(model, train_loader, val_loader, device, power_set_encoder, epochs=50, compute_der_during_training=False, progress_log_file=None, early_stopping_patience=5, early_stopping_min_delta=0.001):
    """Train the SEND model in centralized manner with early stopping."""
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    epoch_metrics = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Initialize progress logging
    if progress_log_file:
        with open(progress_log_file, 'w') as f:
            f.write(f"{'EPOCH':<6} {'LOSS':<12} {'DER':<10} {'ACCURACY':<10} {'VAL_LOSS':<12} {'VAL_DER':<10} {'TIMESTAMP':<10}\n")
            f.write("="*80 + "\n")
    
    for epoch in range(epochs):
        print(f"[{datetime.now()}] Starting epoch {epoch+1}/{epochs}")
        train_loss = 0.0
        batch_losses = []
        # Group predictions by meeting_id for proper DER calculation
        pred_by_rec = defaultdict(list)
        lab_by_rec = defaultdict(list)
            
        total_batches = len(train_loader)
        print(f"[{datetime.now()}] Epoch {epoch+1}/{epochs}: Processing {total_batches} batches...")
        
        for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"[{datetime.now()}] First batch in epoch {epoch+1}")
            features, speaker_embeddings, labels = features.to(device), speaker_embeddings.to(device), labels.to(device)
            speaker_embeddings = speaker_embeddings.float()
            optimizer.zero_grad()
            outputs = model(features, speaker_embeddings)
            batch_size, seq_len, num_classes = outputs.shape
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.reshape(-1)
            if (labels >= num_classes).any() or ((labels < 0) & (labels != -100)).any():
                logger.error(f"Found label out of range! min={labels.min()}, max={labels.max()}, num_classes={num_classes}")
                raise ValueError("Label out of range for CrossEntropyLoss")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=-1)
            
            # Group predictions by meeting_id
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            # meeting_ids: List[np.ndarray] (each with length = max_len of batch)
            meeting_ids_flat = np.concatenate(meeting_ids, axis=0)  # => shape: [batch_size*seq_len]
            
            for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                if meeting_id is not None:  # Skip padded frames
                    pred_by_rec[meeting_id].append(pred)
                    lab_by_rec[meeting_id].append(label)
            
            # Progress indicator for batches
            if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                progress = (batch_idx + 1) / total_batches * 100
                print(f"[{datetime.now()}] Epoch {epoch+1} Progress: {progress:.1f}% ({batch_idx+1}/{total_batches}) - Loss: {loss.item():.4f}")
            
            if batch_idx == 0:
                print(f"Batch {batch_idx}, labels shape: {labels.shape}, unique labels: {torch.unique(labels)}")
                print(f"Batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {torch.unique(predictions)}")
            
        # Calculate DER per recording and aggregate (only if requested)
        ders = {}
        if compute_der_during_training:
            print(f"[{datetime.now()}] Computing DER for {len(pred_by_rec)} recordings...")
            for i, rec_id in enumerate(pred_by_rec):
                if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
                    print(f"[{datetime.now()}] Processing recording {i+1}/{len(pred_by_rec)}: {rec_id}")
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
                    print(f"[{datetime.now()}] Recording {rec_id} DER: {ders[rec_id]:.4f}")
        else:
            print(f"[{datetime.now()}] Skipping DER computation during training for speed (set compute_der_during_training=True to enable)")
        
        # Metrics per epoch
        mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
        # Calculate accuracy across all frames
        all_predictions = []
        all_labels = []
        for rec_id in pred_by_rec:
            all_predictions.extend(pred_by_rec[rec_id])
            all_labels.extend(lab_by_rec[rec_id])
        acc = (np.array(all_predictions) == np.array(all_labels)).mean() if all_labels else float('nan')
        # Average DER across recordings (only if computed)
        der = np.mean(list(ders.values())) if ders else float('nan')
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
        
        print(f"[{datetime.now()}] Epoch {epoch+1}/{epochs} summary: min_loss={min(batch_losses) if batch_losses else 'nan'}, max_loss={max(batch_losses) if batch_losses else 'nan'}, mean_loss={mean_loss}, acc={acc}, DER={der if compute_der_during_training else 'skipped'}")
        
        # Validation after each epoch
        print(f"[{datetime.now()}] Running validation for epoch {epoch+1}...")
        val_loss, val_der, _, _ = evaluate_model(model, val_loader, device, power_set_encoder)
        
        # Early stopping logic
        improvement = best_val_loss - val_loss
        if improvement > early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"[{datetime.now()}] ✅ Validation improved! New best val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"[{datetime.now()}] ⚠️  No improvement for {patience_counter}/{early_stopping_patience} epochs")
        
        # CAPS progress output with validation metrics
        val_der_display = f"{val_der:.4f}" if not np.isnan(val_der) else "N/A"
        val_loss_display = f"{val_loss:.4f}" if not np.isnan(val_loss) else "N/A"
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{epochs} COMPLETED")
        print(f"TRAIN LOSS: {loss_display}")
        print(f"TRAIN DER:  {der_display}")
        print(f"TRAIN ACC:  {acc_display}")
        print(f"VAL LOSS:   {val_loss_display}")
        print(f"VAL DER:    {val_der_display}")
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
            print(f"[{datetime.now()}] 🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"[{datetime.now()}] Best validation loss: {best_val_loss:.4f}")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[{datetime.now()}] ✅ Restored best model (val_loss: {best_val_loss:.4f})")
    
    return epoch_metrics

def evaluate_model(model, val_loader, device, power_set_encoder):
    """Evaluate the SEND model."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    batch_losses = []
    # Group predictions by meeting_id for proper DER calculation
    pred_by_rec = defaultdict(list)
    lab_by_rec = defaultdict(list)

    with torch.no_grad():
        for batch_idx, (features, speaker_embeddings, labels, meeting_ids) in enumerate(val_loader):
            if batch_idx == 0:
                print(f"[{datetime.now()}] First batch in evaluation")
            features, speaker_embeddings, labels = features.to(device), speaker_embeddings.to(device), labels.to(device)
            speaker_embeddings = speaker_embeddings.float()
            outputs = model(features, speaker_embeddings)
            batch_size, seq_len, num_classes = outputs.shape
            outputs = outputs.reshape(-1, num_classes)
            labels = labels.reshape(-1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            batch_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=-1)
            
            # Group predictions by meeting_id
            predictions_np = predictions.cpu().numpy()
            labels_np = labels.cpu().numpy()
            # meeting_ids: List[np.ndarray] (each with length = max_len of batch)
            meeting_ids_flat = np.concatenate(meeting_ids, axis=0)  # => shape: [batch_size*seq_len]
            
            for pred, label, meeting_id in zip(predictions_np, labels_np, meeting_ids_flat):
                if meeting_id is not None:  # Skip padded frames
                    pred_by_rec[meeting_id].append(pred)
                    lab_by_rec[meeting_id].append(label)
            
            if batch_idx == 0:
                print(f"Eval batch {batch_idx}, labels shape: {labels.shape}, unique labels: {np.unique(labels.cpu().numpy())}")
                print(f"Eval batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {np.unique(predictions.cpu().numpy())}")
    
    # Calculate DER per recording and aggregate
    ders = {}
    print(f"[{datetime.now()}] Computing DER for {len(pred_by_rec)} recordings in evaluation...")
    for i, rec_id in enumerate(pred_by_rec):
        if pred_by_rec[rec_id] and lab_by_rec[rec_id]:
            print(f"[{datetime.now()}] Processing recording {i+1}/{len(pred_by_rec)}: {rec_id}")
            # Get speaker_id_list from the dataset
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
            print(f"[{datetime.now()}] Recording {rec_id} DER: {ders[rec_id]:.4f}")
    
    print(f"[{datetime.now()}] Eval summary: min_loss={min(batch_losses):.4f}, max_loss={max(batch_losses):.4f}, mean_loss={np.mean(batch_losses):.4f}")
    # Average DER across recordings
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
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Number of epochs to wait before early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001, help='Minimum improvement required to reset patience counter')
    parser.add_argument('--chunk_size', type=int, default=500, help='Number of samples to process at once during dataset creation (smaller = less memory, default: 500)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (smaller = less memory, default: 4)')
    parser.add_argument('--max_sequence_length', type=int, default=None, help='Maximum sequence length to truncate longer sequences (default: None = no limit). Use to limit memory usage.')
    parser.add_argument('--max_memory_gb', type=float, default=64.0, help='Maximum memory to use in GB (default: 64.0). Will auto-calculate max_sequence_length if needed.')
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
    
    # Determine if we're using all data or a subset
    use_all_data = test_size is None

    logger.info("="*80)
    logger.info("MAIN STARTED")
    logger.info(f"[{datetime.now()}] MAIN: Starting main()")
    sys.stdout.flush()  # Ensure output is written immediately
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{datetime.now()}] MAIN: Using device: {device}")
        
        # Debug GPU information
        if torch.cuda.is_available():
            print(f"[{datetime.now()}] MAIN: CUDA available: True")
            print(f"[{datetime.now()}] MAIN: CUDA device count: {torch.cuda.device_count()}")
            print(f"[{datetime.now()}] MAIN: Current CUDA device: {torch.cuda.current_device()}")
            print(f"[{datetime.now()}] MAIN: CUDA device name: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[{datetime.now()}] MAIN: CUDA available: False - Using CPU")
        
        # Initialize speaker encoder
        print(f"[{datetime.now()}] MAIN: Initializing speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": device}
        ).to(device)
        print(f"[{datetime.now()}] MAIN: Speaker encoder initialized successfully")
        
        # Load and preprocess data
        print_data_loading_info("AMI")
        dataset = load_dataset("edinburghcstr/ami", "ihm")
        print(f"[{datetime.now()}] MAIN: Dataset loaded successfully")
        
        # Determine dataset sizes
        if use_all_data:
            train_size = len(dataset["train"])
            val_size = len(dataset["validation"])
            test_size = len(dataset["test"])
            print(f"[{datetime.now()}] MAIN: Using ALL data - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        else:
            train_size = test_size
            val_size = round(test_size/0.7*0.3)
            test_size = round(test_size/0.7*0.3)
            print(f"[{datetime.now()}] MAIN: Using SUBSET - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        print_dataset_overview("AMI", len(dataset["train"]), train_size)
        
        # Group data by meeting ID for all splits
        print(f"[{datetime.now()}] MAIN: Grouping data by meeting ID...")
        grouped_train = group_by_meeting(dataset["train"].select(range(train_size)))
        grouped_validation = group_by_meeting(dataset["validation"].select(range(val_size)))
        grouped_test = group_by_meeting(dataset["test"].select(range(test_size)))
        
        print_grouping_results(grouped_train, grouped_validation, grouped_test)
        
        # Print statistics for each meeting
        print_meeting_statistics(grouped_train, grouped_validation, grouped_test)
        
        # PSE/SEND Configuration: Fixed N and K (as per original paper)
        N = 5  # Maximum number of target speakers per recording
        K = 3  # Maximum simultaneous overlap (2-4 as per paper)
        
        # Initialize Power Set Encoder with fixed N and K
        print(f"[{datetime.now()}] MAIN: Initializing Power Set Encoder with max_speakers={N}, max_overlap={K}")
        power_set_encoder = PowerSetEncoder(max_speakers=N, max_overlap=K)

        # Calculate number of classes using C(K,N) formula
        num_classes = power_set_encoder.num_classes
        print(f"[{datetime.now()}] MAIN: PSE Configuration: N={N} (max speakers per recording), K={K} (max overlap)")
        print(f"[{datetime.now()}] MAIN: Number of classes using C(K,N) = Σ(k=0 to {K}) C({N},k) = {num_classes}")
        
        # Print PowerSetEncoder examples and statistics
        print_power_set_encoder_examples(power_set_encoder)
        
        # Auto-calculate max_sequence_length based on available memory if not specified
        if max_sequence_length is None:
            # Estimate number of samples in training set (rough estimate)
            estimated_train_samples = sum(len(samples) for samples in grouped_train.values())
            feature_dim = 80  # mel-bands
            
            # Calculate memory per sample: features (float32) + labels (int64) + meeting_ids (object ~8 bytes)
            # features: max_len × 80 × 4 bytes
            # labels: max_len × 8 bytes
            # meeting_ids: max_len × 8 bytes (approx)
            bytes_per_frame = (feature_dim * 4) + 8 + 8  # 336 bytes per frame per sample
            
            # Calculate max_sequence_length that fits in max_memory_gb
            # Leave 20% buffer for other operations
            usable_memory_bytes = (max_memory_gb * 0.8) * (1024**3)
            max_sequence_length = int(usable_memory_bytes / (estimated_train_samples * bytes_per_frame))
            
            print(f"[{datetime.now()}] MAIN: Auto-calculated max_sequence_length={max_sequence_length} based on:")
            print(f"[{datetime.now()}] MAIN:   - Estimated train samples: {estimated_train_samples}")
            print(f"[{datetime.now()}] MAIN:   - Max memory: {max_memory_gb} GB")
            print(f"[{datetime.now()}] MAIN:   - Usable memory (80%): {max_memory_gb * 0.8:.1f} GB")
            print(f"[{datetime.now()}] MAIN:   - Estimated memory usage: {(estimated_train_samples * max_sequence_length * bytes_per_frame) / (1024**3):.2f} GB")
        else:
            print(f"[{datetime.now()}] MAIN: Using user-specified max_sequence_length={max_sequence_length}")
        
        # Prepare data loaders for training and evaluation
        train_loader, val_loader, test_loader = prepare_data_loaders(
            grouped_train, grouped_validation, grouped_test, speaker_encoder, power_set_encoder, 
            batch_size=batch_size, N=N, chunk_size=chunk_size, max_sequence_length=max_sequence_length
        ) 
        
        # Print experiment configuration
        print_experiment_config(1, 1, epochs, train_size)  # Single centralized training
        print(f"[{datetime.now()}] MAIN: Early stopping patience: {early_stopping_patience} epochs")
        print(f"[{datetime.now()}] MAIN: Early stopping min delta: {early_stopping_min_delta}")

        
        # Get all unique speakers for speaker embedding computation
        # speaker_ids = set()
        # for grouped in [grouped_train, grouped_validation, grouped_test]:
        #     for samples in grouped.values():
        #         for sample in samples:
        #             speaker_ids.add(sample["speaker_id"])
        # all_speaker_ids = sorted(list(speaker_ids))
        # speaker_id_list = all_speaker_ids[:N]  # Limit to N slots for PSE consistency
        # print(f"[{datetime.now()}] MAIN: Detected {len(all_speaker_ids)} unique speakers in dataset: {all_speaker_ids}")
        # print(f"[{datetime.now()}] MAIN: Using first {len(speaker_id_list)} speakers for PSE slots: {speaker_id_list}")
        # print(f"[{datetime.now()}] MAIN: Note: PSE uses fixed N={N} slots per recording, not all {len(all_speaker_ids)} speakers")
        
        # Analyze speaker distribution
        analyze_speaker_distribution(grouped_train)
        
        # Create and train model
        print(f"[{datetime.now()}] MAIN: Creating SEND model...")
        model = SENDModel(num_classes=num_classes).to(device)
        
        # Print SENDModel statistics
        print_send_model_statistics(model)
        
        # Use centralized training data loaders directly
        print(f"[{datetime.now()}] MAIN: Using centralized training data loaders...")
        
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
        
        print(f"[{datetime.now()}] MAIN: Training samples: {total_training_samples}")
        print(f"[{datetime.now()}] MAIN: Validation samples: {total_validation_samples}")
        print(f"[{datetime.now()}] MAIN: Test samples: {total_test_samples}")
        print(f"[{datetime.now()}] MAIN: Total training frames: {total_training_frames}")
        
        # Additional information about data distribution
        if total_training_samples > 0:
            avg_frames_per_sample = total_training_frames / total_training_samples
            print(f"[{datetime.now()}] MAIN: Average frames per sample: {avg_frames_per_sample:.1f}")
            if use_all_data:
                print(f"[{datetime.now()}] MAIN: Using ALL available data from AMI dataset")
            else:
                print(f"[{datetime.now()}] MAIN: Note: test_size={train_size} refers to number of dataset records selected for training")
            print(f"[{datetime.now()}] MAIN: Each meeting recording contains multiple audio segments, each segment becomes multiple training samples")
            print(f"[{datetime.now()}] MAIN: Each training sample contains multiple frames (time steps) for sequence learning")
        
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
        
        print(f"[{datetime.now()}] MAIN: Experiment tag: {exp_tag}")
        print(f"[{datetime.now()}] MAIN: Logs directory: {artifact_logs_dir}")
        print(f"[{datetime.now()}] MAIN: Plots directory: {artifact_plots_dir}")
        
        # Compute speaker embeddings for train set
        print(f"[{datetime.now()}] MAIN: Computing speaker embeddings for train set...")
        speaker_to_embedding = compute_speaker_embeddings(grouped_train, speaker_encoder)
        
        # === CENTRALIZED TRAINING ===
        print("\n==================== STARTING CENTRALIZED TRAINING ====================\n")
        
        # Train the model
        print(f"[{datetime.now()}] MAIN: DER computation during training: {'ENABLED' if compute_der_during_training else 'DISABLED (faster training)'}")
        
        # Create progress log file path
        progress_log_file = os.path.join(artifact_logs_dir, "training_progress.txt")
        print(f"[{datetime.now()}] MAIN: Progress will be logged to: {progress_log_file}")
        
        training_metrics = train_model(model, train_loader, val_loader, device, power_set_encoder, epochs, compute_der_during_training, progress_log_file, early_stopping_patience, early_stopping_min_delta)
        
        # Evaluate on validation set
        print(f"[{datetime.now()}] MAIN: Evaluating on validation set...")
        val_loss, val_der, val_pred_by_rec, val_lab_by_rec = evaluate_model(model, val_loader, device, power_set_encoder)
        
        print(f"[{datetime.now()}] MAIN: Training completed successfully")
        print(f"[{datetime.now()}] MAIN: Final validation loss: {val_loss:.4f}")
        print(f"[{datetime.now()}] MAIN: Final validation DER: {val_der:.4f}")
        
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
                loss = nn.CrossEntropyLoss()(outputs, labels)
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
            f"Best Validation Loss: {min([m.get('val_loss', float('inf')) for m in epoch_metrics]) if epoch_metrics else 'N/A'}",
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
        # Add any necessary cleanup code here
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        print("Process completed.")

if __name__ == "__main__":
    main() 