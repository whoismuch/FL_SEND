import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context, Metrics
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from speechbrain.pretrained import EncoderClassifier
from datasets import load_dataset
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from data_processing import (
    split_data_for_clients, 
    extract_features, 
    simulate_overlapping_speech, 
    group_by_meeting,
    prepare_data_loaders,
    power_set_encoding,
    calculate_der,
    compute_speaker_embeddings
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
    """Power Set Encoding for overlapping speech diarization."""
    def __init__(self, max_speakers: int = 4):
        self.max_speakers = max_speakers
        self.num_classes = 2 ** max_speakers
        
    def encode(self, speaker_labels: List[int]) -> int:
        """Encode speaker labels (as speaker IDs) into a single integer using power set encoding."""
        if any(label >= self.max_speakers or label < 0 for label in speaker_labels):
            raise ValueError(f"Speaker ID in labels exceeds max_speakers ({self.max_speakers}) or is negative.")
        # Create binary vector: 1 if speaker ID is active, else 0
        vector_bin = [1 if i in speaker_labels else 0 for i in range(self.max_speakers)]
        return sum(bit * (2 ** i) for i, bit in enumerate(vector_bin))
    
    def decode(self, encoded_value: int) -> List[int]:
        """Decode an encoded value back into a list of active speaker IDs (positions with 1)."""
        binary = format(encoded_value, f'0{self.max_speakers}b')
        return [i for i, bit in enumerate(binary[::-1]) if bit == '1']

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

class OverlappingSpeechDataset(Dataset):
    """Dataset for overlapping speech diarization."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, speaker_encoder: EncoderClassifier, speaker_to_embedding: Dict[int, np.ndarray]):
        self.features = features
        self.labels = labels
        self.speaker_encoder = speaker_encoder
        self.speaker_to_embedding = speaker_to_embedding
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Extract speaker embeddings
        with torch.no_grad():
            speaker_embedding = self.speaker_to_embedding[self.labels[idx]]
        
        return feature, speaker_embedding, label

class SENDClient(NumPyClient):
    """Federated Learning client for SEND model."""
    def __init__(
        self,
        model: SENDModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        power_set_encoder: PowerSetEncoder,
        speaker_encoder: EncoderClassifier,
        speaker_to_embedding: Dict[int, np.ndarray]
    ):
        print(f"[{datetime.now()}] SENDClient: Initializing client {id(self)}")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.power_set_encoder = power_set_encoder
        self.speaker_encoder = speaker_encoder
        self.speaker_to_embedding = speaker_to_embedding
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        print(f"[{datetime.now()}] SENDClient: Initialization complete for client {id(self)}")
        print(f"[DEBUG] SENDClient: train_loader size: {len(self.train_loader)}")
        print(f"[DEBUG] SENDClient: val_loader size: {len(self.val_loader)}")
        if len(self.train_loader) == 0:
            print(f"[WARNING] SENDClient: train_loader is EMPTY for client {id(self)}!")
        if len(self.val_loader) == 0:
            print(f"[WARNING] SENDClient: val_loader is EMPTY for client {id(self)}!")
    
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        print("=== CLIENT LOG: fit started ===")
        print(f"[DEBUG] fit: train_loader size: {len(self.train_loader)}")
        print(f"[DEBUG] fit: number of batches: {len(list(self.train_loader))}")
        print(f"=== aaa ===")
        print(f"=== bbb ===")
        print("=== CLIENT LOG: fit started ===")
        print(f"[{datetime.now()}] SENDClient: Starting fit for client {id(self)}")
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)
        start_time = time.time()
        epoch_metrics = []  # Collect metrics for each epoch
        for epoch in range(epochs):
            train_loss = 0.0
            batch_losses = []
            all_predictions = []
            all_labels = []
            for batch_idx, (features, speaker_embeddings, labels) in enumerate(self.train_loader):
                if batch_idx == 0:
                    print(f"[{datetime.now()}] SENDClient: First batch in fit for client {id(self)} (epoch {epoch+1}/{epochs})")
                features, speaker_embeddings, labels = features.to(self.device), speaker_embeddings.to(self.device), labels.to(self.device)
                speaker_embeddings = speaker_embeddings.float()
                self.optimizer.zero_grad()
                outputs = self.model(features, speaker_embeddings)
                batch_size, seq_len, num_classes = outputs.shape
                outputs = outputs.reshape(-1, num_classes)
                labels = labels.reshape(-1)
                if (labels >= num_classes).any() or ((labels < 0) & (labels != -100)).any():
                    logger.error(f"Found label out of range! min={labels.min()}, max={labels.max()}, num_classes={num_classes}")
                    raise ValueError("Label out of range for CrossEntropyLoss")
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                batch_losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                if batch_idx == 0:
                    print(f"Batch {batch_idx}, labels shape: {labels.shape}, unique labels: {torch.unique(labels)}")
                    print(f"Batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {torch.unique(predictions)}")
            # Metrics per epoch
            mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
            acc = (np.array(all_predictions) == np.array(all_labels)).mean() if all_labels else float('nan')
            der = self.calculate_der(all_predictions, all_labels) if all_labels else float('nan')
            print(f"[DEBUG] Epoch {epoch+1}/{epochs} unique labels: {np.unique(all_labels) if all_labels else 'EMPTY'}")
            print(f"[DEBUG] Epoch {epoch+1}/{epochs} unique predictions: {np.unique(all_predictions) if all_predictions else 'EMPTY'}")
            print(f"[{datetime.now()}] SENDClient: Epoch {epoch+1}/{epochs} summary for client {id(self)}: min_loss={min(batch_losses) if batch_losses else 'nan'}, max_loss={max(batch_losses) if batch_losses else 'nan'}, mean_loss={mean_loss}, acc={acc}, DER={der}")
            # Collect metrics for this epoch
            epoch_metrics.append({
                "train_loss": float(mean_loss),
                "acc": float(acc) if not np.isnan(acc) else None,
                "der": float(der) if not np.isnan(der) else None,
            })
        elapsed = time.time() - start_time
        print(f"[{datetime.now()}] SENDClient: Finished fit for client {id(self)}, total time: {elapsed:.2f} sec")
        print("=== CLIENT LOG: fit finished ===")
        print(f"=== CLIENT LOG: train_loader length: {len(self.train_loader)} ===")
        # Return epoch_metrics for aggregation and plotting (as JSON string)
        return self.get_parameters({}), len(self.train_loader), {"train_loss": mean_loss, "epoch_metrics": json.dumps(epoch_metrics)}
    
    def evaluate(self, parameters, config):
        print("=== CLIENT LOG: evaluate started ===")
        print("=== CLIENT LOG: evaluate started ===")
        print(f"[{datetime.now()}] SENDClient: Starting evaluate for client {id(self)}")
        self.set_parameters(parameters)
        self.model.eval()
        val_loss = 0.0
        batch_losses = []
        all_predictions = []
        all_labels = []
        start_time = time.time()
        epoch_metrics = []  # Collect metrics for each epoch (for compatibility)
        with torch.no_grad():
            for batch_idx, (features, speaker_embeddings, labels) in enumerate(self.val_loader):
                if batch_idx == 0:
                    print(f"[{datetime.now()}] SENDClient: First batch in evaluate for client {id(self)}")
                features, speaker_embeddings, labels = features.to(self.device), speaker_embeddings.to(self.device), labels.to(self.device)
                speaker_embeddings = speaker_embeddings.float()
                outputs = self.model(features, speaker_embeddings)
                batch_size, seq_len, num_classes = outputs.shape
                outputs = outputs.reshape(-1, num_classes)
                labels = labels.reshape(-1)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                batch_losses.append(loss.item())
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                if batch_idx == 0:
                    print(f"Eval batch {batch_idx}, labels shape: {labels.shape}, unique labels: {np.unique(labels.cpu().numpy())}")
                    print(f"Eval batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {np.unique(predictions.cpu().numpy())}")
        print(f"[{datetime.now()}] SENDClient: Eval summary for client {id(self)}: min_loss={min(batch_losses):.4f}, max_loss={max(batch_losses):.4f}, mean_loss={np.mean(batch_losses):.4f}")
        elapsed = time.time() - start_time
        print(f"[{datetime.now()}] SENDClient: Finished evaluate for client {id(self)}, total time: {elapsed:.2f} sec")
        der = self.calculate_der(all_predictions, all_labels)
        print("=== CLIENT LOG: evaluate finished ===")
        print("=== CLIENT LOG: evaluate finished ===")
        # For compatibility, return epoch_metrics (single epoch for val) as JSON string
        mean_loss = np.mean(batch_losses) if batch_losses else float('nan')
        epoch_metrics.append({
            "val_loss": float(mean_loss),
            "der": float(der) if not np.isnan(der) else None,
        })
        return (
            float(val_loss / len(self.val_loader)),
            len(self.val_loader),
            {"val_loss": val_loss / len(self.val_loader), "der": der, "epoch_metrics": json.dumps(epoch_metrics)}
        )
    
    def calculate_der(self, predictions: List[int], labels: List[int], speaker_id_list: list = None, debug: bool = True) -> float:
        """Calculate Diarization Error Rate with detailed logging and correct multi-speaker segments."""
        from pyannote.core import Segment, Annotation
        from pyannote.metrics.diarization import DiarizationErrorRate
        reference = Annotation()
        hypothesis = Annotation()
        mismatches = 0
        unique_label_values = set()
        unique_pred_values = set()
        active_speakers_labels = []
        active_speakers_preds = []
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if label == -100:
                continue
            true_bits = self.power_set_encoder.decode(label)
            pred_bits = self.power_set_encoder.decode(pred)
            if speaker_id_list is None:
                speaker_id_list = list(range(len(true_bits)))
            n_speakers = len(speaker_id_list)
            # Truncate bit vectors to match speaker_id_list length
            true_bits = true_bits[:n_speakers]
            pred_bits = pred_bits[:n_speakers]
            # Build reference and hypothesis speaker sets
            ref_speakers = frozenset(f"speaker_{speaker_id_list[idx]}" for idx, bit in enumerate(true_bits) if bit == 1)
            hyp_speakers = frozenset(f"speaker_{speaker_id_list[idx]}" for idx, bit in enumerate(pred_bits) if bit == 1)
            reference[Segment(i, i+1)] = ref_speakers
            hypothesis[Segment(i, i+1)] = hyp_speakers
            unique_label_values.add(label)
            unique_pred_values.add(pred)
            active_speakers_labels.append(sum(true_bits))
            active_speakers_preds.append(sum(pred_bits))
            if debug and mismatches < 10 and true_bits != pred_bits:
                print(f"[DER DEBUG] Frame {i}: label={label}, pred={pred}, true_bits={true_bits}, pred_bits={pred_bits}")
                mismatches += 1
        if debug:
            print(f"[DER DEBUG] speaker_id_list (bit mapping): {speaker_id_list}")
            print(f"[DER DEBUG] Unique label values: {unique_label_values}")
            print(f"[DER DEBUG] Unique pred values: {unique_pred_values}")
            print(f"[DER DEBUG] Active speakers per frame (labels): min={min(active_speakers_labels)}, max={max(active_speakers_labels)}, mean={np.mean(active_speakers_labels):.2f}")
            print(f"[DER DEBUG] Active speakers per frame (preds): min={min(active_speakers_preds)}, max={max(active_speakers_preds)}, mean={np.mean(active_speakers_preds):.2f}")
            print(f"[DER DEBUG] Reference segments (first 10): {list(reference.itertracks(yield_label=True))[:10]}")
            print(f"[DER DEBUG] Hypothesis segments (first 10): {list(hypothesis.itertracks(yield_label=True))[:10]}")
        metric = DiarizationErrorRate()
        der = metric(reference, hypothesis)
        print(f"[DER DEBUG] DER calculation: valid frames used = {len(predictions)}, DER = {der}")
        return der

def find_available_port(start_port: int = 8080, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port.
    
    Args:
        start_port: The port to start checking from
        max_attempts: Maximum number of ports to check
        
    Returns:
        An available port number
    """
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def is_running_in_colab():
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def start_client(client: SENDClient, server_address: str):
    """Start a Flower client.
    
    Args:
        client: The client to start
        server_address: The address of the server
    """
    try:
        print(f"Client attempting to connect to {server_address}")
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        print("Client finished successfully")
    except Exception as e:
        logger.error(f"Client error: {str(e)}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Federated Learning for Overlapping Speech Diarization")
    parser.add_argument('--test_size', type=int, default=6, help='Number of samples to use for testing')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for local training')
    parser.add_argument('--num_rounds', type=int, default=3, help='Number of federated learning rounds')
    parser.add_argument('--num_clients', type=int, default=2, help='Number of federated clients')
    args = parser.parse_args()

    # Assign arguments to variables
    test_size = args.test_size
    epochs = args.epochs
    num_rounds = args.num_rounds
    num_clients = args.num_clients

    print("MAIN STARTED")
    print(f"[{datetime.now()}] MAIN: Starting main()")
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[{datetime.now()}] MAIN: Using device: {device}")
        
        # Initialize speaker encoder
        print(f"[{datetime.now()}] MAIN: Initializing speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": device}
        ).to(device)
        print(f"[{datetime.now()}] MAIN: Speaker encoder initialized successfully")
        
        # Load and preprocess data
        print(f"[{datetime.now()}] MAIN: Loading AMI dataset...")
        dataset = load_dataset("edinburghcstr/ami", "ihm")
        print(f"[{datetime.now()}] MAIN: Dataset loaded successfully")
        
        # Take a small subset for testing
        print(f"[{datetime.now()}] MAIN: Using subset of {test_size} samples for testing")
        
        # Group data by meeting ID for all splits
        print(f"[{datetime.now()}] MAIN: Grouping data by meeting ID...")
        grouped_train = group_by_meeting(dataset["train"].select(range(test_size)))
        grouped_validation = group_by_meeting(dataset["validation"].select(range(test_size)))
        grouped_test = group_by_meeting(dataset["test"].select(range(test_size)))
        
        print(f"[{datetime.now()}] MAIN: Grouped {len(grouped_train)} meetings from training set")
        print(f"[{datetime.now()}] MAIN: Grouped {len(grouped_validation)} meetings from validation set")
        print(f"[{datetime.now()}] MAIN: Grouped {len(grouped_test)} meetings from test set")
        
        # Prepare test_loader for final evaluation
        _, _, test_loader = prepare_data_loaders(
            grouped_train, grouped_validation, grouped_test, speaker_encoder
        )
        
        # Determine number of unique speakers across all splits
        speaker_ids = set()
        for grouped in [grouped_train, grouped_validation, grouped_test]:
            for samples in grouped.values():
                for sample in samples:
                    speaker_ids.add(sample["speaker_id"])
        speaker_id_list = sorted(list(speaker_ids))
        num_speakers = len(speaker_id_list)
        num_classes = 2 ** num_speakers
        print(f"[{datetime.now()}] MAIN: Detected {num_speakers} unique speakers, num_classes={num_classes}")
        print(f"[{datetime.now()}] MAIN: speaker_ids: {speaker_id_list}")
        
        # Initialize Power Set Encoder
        print(f"[{datetime.now()}] MAIN: Initializing Power Set Encoder with max_speakers={num_speakers}")
        power_set_encoder = PowerSetEncoder(max_speakers=num_speakers)
        
        # Create and train model
        print(f"[{datetime.now()}] MAIN: Creating SEND model...")
        model = SENDModel(num_classes=num_classes).to(device)
        
        # Split data for federated learning with fewer clients
        print(f"[{datetime.now()}] MAIN: Splitting data for federated learning...")
        client_data = split_data_for_clients(grouped_train, num_clients, speaker_encoder)
        
        # Validate client data
        if not client_data or len(client_data) < num_clients:
            raise ValueError(f"Not enough data for {num_clients} clients. Only {len(client_data) if client_data else 0} clients can be created.")
        
        print(f"[{datetime.now()}] MAIN: Split data among {len(client_data)} clients")
        
        # Compute speaker embeddings for train set
        print(f"[{datetime.now()}] MAIN: Computing speaker embeddings for train set...")
        speaker_to_embedding = compute_speaker_embeddings(grouped_train, speaker_encoder)
        
        # Define client function for simulation
        def client_fn(context: Context):
            cid = context.node_config['partition-id']
            print(f"[client_fn] Got cid from context.node_config['partition-id']: {cid}")
            print(f"[{datetime.now()}] MAIN: Creating client {cid}")
            try:
                client_idx = int(cid)
                if client_idx >= len(client_data):
                    raise ValueError(f"Client ID {client_idx} is out of range. Only {len(client_data)} clients available.")
                train_loader, val_loader = client_data[client_idx]
                # Create new model instance for each client
                client_model = SENDModel(num_classes=num_classes).to(device)
                print(f"[{datetime.now()}] MAIN: Client {cid} created and ready")
                return SENDClient(
                    model=client_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    power_set_encoder=power_set_encoder,
                    speaker_encoder=speaker_encoder,
                    speaker_to_embedding=speaker_to_embedding
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
            losses = [v.get('train_loss') for v in round_info['client_metrics'].values() if 'train_loss' in v]
            ders = [v.get('der') for v in round_info['client_metrics'].values() if 'der' in v]
            round_info['mean_loss'] = sum(losses)/len(losses) if losses else None
            round_info['mean_der'] = sum(ders)/len(ders) if ders else None
            round_info['round'] = len(round_metrics)
            round_metrics.append(round_info)
            return {}

        # Use this aggregation function in strategy
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=num_clients,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            on_fit_config_fn=lambda _: {"epochs": epochs},
            on_evaluate_config_fn=lambda _: {"epochs": 1},
            initial_parameters=fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in model.state_dict().items()]
            ),
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        )
        
        # Calculate GPU resources
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus_per_client = max(1, num_gpus // num_clients) if num_gpus > 0 else 0
        print(f"Available GPUs: {num_gpus}, GPUs per client: {gpus_per_client}")
        
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            ray_init_args={
                "num_cpus": num_clients,
                "num_gpus": num_gpus,
                "include_dashboard": False,
                "ignore_reinit_error": True,
            },
            client_resources={
                "num_cpus": 1,
                "num_gpus": gpus_per_client
            }
        )
        
        # Final evaluation on test set
        print("\n==================== TESTING STARTED ====================\n")
        model.eval()
        test_loss = 0.0
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for features, speaker_embeddings, labels in test_loader:
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
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        print(f"Test predictions shape: {np.array(all_predictions).shape}, unique: {np.unique(all_predictions)}")
        print(f"Test labels shape: {np.array(all_labels).shape}, unique: {np.unique(all_labels)}")
        # Calculate final metrics
        test_loss = test_loss / len(test_loader)
        der = calculate_der(all_predictions, all_labels, power_set_encoder, speaker_id_list=speaker_id_list, debug=True)
        print(f"\n==================== TESTING FINISHED ====================\n")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final DER: {der:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final DER: {der:.4f}")

        # === LOG FINAL RESULTS TO FILE ===
        # Use actual experiment parameters, not hardcoded values
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dt_str_human = datetime.now().strftime("%Y-%m-%d-%H-%M")
        exp_tag = f"exp_{test_size}size_{epochs}epochs_{num_rounds}rounds_{num_clients}clients_{dt_str_human}"
        # Artifact directories for logs and plots
        artifact_logs_dir = os.path.join("out_artifacts", "logs", exp_tag)
        artifact_plots_dir = os.path.join("out_artifacts", "plots", exp_tag)
        os.makedirs(artifact_logs_dir, exist_ok=True)
        os.makedirs(artifact_plots_dir, exist_ok=True)
        # File paths for logs and metrics (simple names)
        exp_filename = "experiment.txt"
        exp_filepath = os.path.join(artifact_logs_dir, exp_filename)
        # Prepare lines for logging
        result_lines = [
            f"Experiment: {exp_tag}",
            f"Num records: {test_size}",
            f"Num clients: {num_clients}",
            f"Num epochs: {epochs}",
            f"Num rounds: {num_rounds}",
            f"Datetime: {dt_str}",
            f"Final Test Loss: {test_loss:.4f}",
            f"Final DER: {der:.4f}",
        ]
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
                round_labels = []
                for rnd, epoch_list in rounds.items():
                    if epoch_list:
                        last_epoch = epoch_list[-1]
                        round_train_loss.append(last_epoch.get('train_loss'))
                        round_der.append(last_epoch.get('der'))
                        round_labels.append(f"Round {rnd+1}")
                # Plot loss per round
                plt.figure(figsize=(8, 5))
                plt.plot(round_labels, round_train_loss, marker='o', label='Train Loss')
                plt.xlabel('Round')
                plt.ylabel('Loss')
                plt.title(f'Client {cid} Loss per Round')
                plt.tight_layout()
                plot_path = os.path.join(artifact_plots_dir, f"client{cid}_loss_per_round.png")
                plt.savefig(plot_path)
                plt.close()
                # Plot DER per round
                plt.figure(figsize=(8, 5))
                plt.plot(round_labels, round_der, marker='o', label='DER')
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