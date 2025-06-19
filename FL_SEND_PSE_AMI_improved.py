# Always import os only at the top-level to avoid shadowing.
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
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
import csv

# === FileHandler for logging ===
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# === Helper for metrics logging ===
def log_metrics_to_csv(filename, round_num, epoch, client_id, loss, der):
    header = ['round', 'epoch', 'client_id', 'loss', 'der']
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([round_num, epoch, client_id, loss, der])

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
        """Encode speaker labels into a single integer using power set encoding."""
        if len(speaker_labels) > self.max_speakers:
            raise ValueError(f"Number of speakers exceeds maximum ({self.max_speakers})")
        
        # Pad with zeros if necessary
        padded_labels = speaker_labels + [0] * (self.max_speakers - len(speaker_labels))
        return sum(label * (2 ** i) for i, label in enumerate(padded_labels))
    
    def decode(self, encoded_value: int) -> List[int]:
        """Decode an encoded value back into speaker labels."""
        binary = format(encoded_value, f'0{self.max_speakers}b')
        return [int(bit) for bit in binary]

class FSMNLayer(nn.Module):
    """Feedforward Sequential Memory Network layer."""
    def __init__(self, input_dim: int, hidden_dim: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.memory = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, current_dim = x.shape
        
        # Check and adapt input dimensions
        if current_dim != self.input_dim:
            x = nn.Linear(current_dim, self.input_dim)(x)
        
        # Linear projection
        h = self.linear(x)  # (batch_size, seq_len, hidden_dim)
        
        # Memory mechanism
        memory = torch.zeros_like(h)
        for i in range(seq_len):
            start_idx = max(0, i - self.stride)
            memory[:, i] = self.memory(x[:, start_idx:i+1].mean(dim=1))
        
        return h + memory

class SENDModel(nn.Module):
    """Speaker Embedding-aware Neural Diarization model with Power-Set Encoding."""
    def __init__(self, input_dim: int = 80, hidden_dim: int = 512, num_classes: int = 16):
        super().__init__()
        
        # Speech Encoder (FSMN)
        self.speech_encoder = nn.ModuleList([
            FSMNLayer(input_dim if i == 0 else hidden_dim, hidden_dim, stride=2**i)
            for i in range(8)  # 8 FSMN layers
        ])
        
        # Speaker Encoder (MLP)
        self.speaker_encoder = nn.Sequential(
            nn.Linear(192, hidden_dim),  # ECAPA-TDNN output size is 192
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # CI Scorer (Context-Independent)
        self.ci_scorer = nn.Linear(hidden_dim, 1)  # Dot product with speaker embeddings
        
        # CD Scorer (Context-Dependent)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.cd_scorer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Post-Net (FSMN)
        self.post_net = nn.ModuleList([
            FSMNLayer(hidden_dim, hidden_dim, stride=2**i)
            for i in range(6)  # 6 FSMN layers
        ])
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Adapter for combining CI and CD scores
        self.combine_adapter = None
        
    def forward(self, x: torch.Tensor, speaker_embeddings: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        # speaker_embeddings shape: (batch_size, num_speakers, 192)
        batch_size, seq_len, _ = x.shape
        num_speakers = speaker_embeddings.size(1)
        
        # Process audio features through Speech Encoder
        for fsmn in self.speech_encoder:
            x = fsmn(x)  # (batch_size, sequence_length, hidden_dim)
        
        # Process speaker embeddings
        speaker_features = self.speaker_encoder(speaker_embeddings)  # (batch_size, num_speakers, hidden_dim)
        
        # CI Scoring
        ci_scores = []
        for i in range(num_speakers):
            # Dot product between audio features and speaker embeddings
            score = torch.matmul(x, speaker_features[:, i].unsqueeze(-1)).squeeze(-1)  # (batch_size, sequence_length)
            ci_scores.append(score)
        ci_scores = torch.stack(ci_scores, dim=1)  # (batch_size, num_speakers, sequence_length)
        ci_scores = ci_scores.transpose(1, 2)  # (batch_size, sequence_length, num_speakers)
        
        # CD Scoring
        cd_scores = self.cd_scorer(x)  # (batch_size, sequence_length, hidden_dim)
        
        # Combine CI and CD scores
        combined = torch.cat([
            ci_scores,  # (batch_size, sequence_length, num_speakers)
            cd_scores  # (batch_size, sequence_length, hidden_dim)
        ], dim=2)  # (batch_size, sequence_length, num_speakers + hidden_dim)
        
        # Create adapter if it doesn't exist or dimensions have changed
        if self.combine_adapter is None or self.combine_adapter.in_features != combined.size(-1):
            self.combine_adapter = nn.Linear(combined.size(-1), self.post_net[0].input_dim).to(combined.device)
        
        # Adapt dimensions before Post-Net
        combined = self.combine_adapter(combined)
        
        # Process through Post-Net
        for fsmn in self.post_net:
            combined = fsmn(combined)
        
        # Final classification
        out = self.classifier(combined)  # (batch_size, sequence_length, num_classes)
        
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
        speaker_to_embedding: Dict[int, np.ndarray],
        log_suffix: str = ""
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
        self.log_suffix = log_suffix  # Store log_suffix for metrics file naming
        print(f"[{datetime.now()}] SENDClient: Initialization complete for client {id(self)}")
        print(f"[DEBUG] SENDClient: train_loader size: {len(self.train_loader)}")
        print(f"[DEBUG] SENDClient: val_loader size: {len(self.val_loader)}")
        if len(self.train_loader) == 0:
            print(f"[WARNING] SENDClient: train_loader is EMPTY for client {id(self)}!")
        if len(self.val_loader) == 0:
            print(f"[WARNING] SENDClient: val_loader is EMPTY for client {id(self)}!")
        # === Add per-client FileHandler for logging (do not remove existing handlers) ===
        client_log_file = os.path.join(log_dir, f"client_{id(self)}_log{log_suffix}.log")
        client_file_handler = logging.FileHandler(client_log_file, mode='a')
        client_file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        client_file_handler.setFormatter(formatter)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == client_log_file for h in logger.handlers):
            logger.addHandler(client_file_handler)
        # === Add StreamHandler to also log to console (if not present) ===
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        # Prevent log propagation to avoid duplicate logs in root logger
        logger.propagate = False
        
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
        round_num = config.get("round", 0)  # Передавайте round в config через on_fit_config_fn
        client_id = config.get("cid", id(self))
        start_time = time.time()
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
            # === Log metrics to CSV ===
            metrics_file = os.path.join(log_dir, f"client_metrics_{client_id}{self.log_suffix}.csv")
            log_metrics_to_csv(metrics_file, round_num, epoch+1, client_id, mean_loss, der)
        elapsed = time.time() - start_time
        print(f"[{datetime.now()}] SENDClient: Finished fit for client {id(self)}, total time: {elapsed:.2f} sec")
        print("=== CLIENT LOG: fit finished ===")
        print(f"=== CLIENT LOG: train_loader length: {len(self.train_loader)} ===")
        return self.get_parameters({}), len(self.train_loader), {"train_loss": mean_loss}
    
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
        return (
            float(val_loss / len(self.val_loader)),
            len(self.val_loader),
            {"val_loss": val_loss / len(self.val_loader), "der": der}
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
            # Используем frozenset для корректной работы pyannote
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
    print("MAIN STARTED")
    print(f"[{datetime.now()}] MAIN: Starting main()")
    try:
        # === Set experiment parameters here ===
        test_size = 10
        num_clients = 2
        num_rounds = 3
        num_epochs = 2
        run_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        # === Create log_suffix with all experiment parameters ===
        log_suffix = f"_{test_size}recs_{num_clients}clients_{num_epochs}epochs_{num_rounds}rounds_{run_datetime}"
        # === Create FileHandler for logging with correct log_suffix ===
        log_file = os.path.join(log_dir, f"experiment{log_suffix}.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        # Remove old handlers to avoid duplicate logs
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.addHandler(file_handler)
        
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
        dataset = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True)
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
        num_clients = 2  # reduce number of clients for testing
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
                client_model = SENDModel(num_classes=num_classes).to(device)
                print(f"[{datetime.now()}] MAIN: Client {cid} created and ready")
                return SENDClient(
                    model=client_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    power_set_encoder=power_set_encoder,
                    speaker_encoder=speaker_encoder,
                    speaker_to_embedding=speaker_to_embedding,
                    log_suffix=log_suffix
                ).to_client()
            except Exception as e:
                logger.error(f"Error creating client {cid}: {str(e)}")
                raise
        
        # Start federated learning with simulation
        print("Starting federated learning simulation...")
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            min_fit_clients=2,
            min_evaluate_clients=2,
            on_fit_config_fn=lambda server_round: {"epochs": num_epochs, "round": server_round, "cid": None},
            on_evaluate_config_fn=lambda server_round: {"epochs": 1, "round": server_round, "cid": None},
            initial_parameters=fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in model.state_dict().items()]
            ),
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

        # Print Ray/Flower client logs after simulation
        import glob
        def print_ray_logs():
            # Use global os module, do not import locally.
            ray_log_dir = "/tmp/ray/session_latest/logs/"
            if os.path.exists(ray_log_dir):
                log_files = glob.glob(os.path.join(ray_log_dir, "*.out"))
                if not log_files:
                    print("No Ray log files found.")
                for f in log_files:
                    print(f"\n===== {f} =====")
                    try:
                        with open(f, "r") as logf:
                            content = logf.read()[-5000:]
                            print(content)
                    except Exception as e:
                        print(f"Could not read {f}: {e}")
            else:
                print("Ray log directory not found.")

        # === Plotting function for metrics with log_suffix ===
        def plot_client_metrics(log_dir="logs", log_suffix=""):
            import pandas as pd
            import matplotlib.pyplot as plt
            import glob
            metric_files = glob.glob(os.path.join(log_dir, f"client_metrics_*{log_suffix}.csv"))
            for file in metric_files:
                df = pd.read_csv(file)
                client_id = df['client_id'].iloc[0]
                plt.figure(figsize=(10,5))
                plt.plot(df['epoch'], df['loss'], label='Loss')
                plt.plot(df['epoch'], df['der'], label='DER')
                plt.xlabel('Epoch')
                plt.title(f'Client {client_id}: Loss and DER per Epoch')
                plt.legend()
                plt.grid()
                plot_file = os.path.join(log_dir, f"client_{client_id}_metrics{log_suffix}.png")
                plt.savefig(plot_file)
                plt.close()
            print(f"Saved metrics plots for {len(metric_files)} clients in {log_dir}")

        # === Вызовите plot_client_metrics() после обучения, если хотите построить графики ===
        plot_client_metrics(log_dir, log_suffix)

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