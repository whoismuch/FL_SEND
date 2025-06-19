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
logger.info(f"Using device: {device}")

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
        speaker_to_embedding: Dict[int, np.ndarray]
    ):
        logger.info(f"[{datetime.now()}] SENDClient: Initializing client {id(self)}")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.power_set_encoder = power_set_encoder
        self.speaker_encoder = speaker_encoder
        self.speaker_to_embedding = speaker_to_embedding
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        logger.info(f"[{datetime.now()}] SENDClient: Initialization complete for client {id(self)}")
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        logger.info(f"[{datetime.now()}] SENDClient: Starting fit for client {id(self)}")
        self.set_parameters(parameters)
        self.model.train()
        epochs = config.get("epochs", 1)
        start_time = time.time()
        for epoch in range(epochs):
            train_loss = 0.0
            batch_losses = []
            all_predictions = []
            all_labels = []
            for batch_idx, (features, speaker_embeddings, labels) in enumerate(self.train_loader):
                if batch_idx == 0:
                    logger.info(f"[{datetime.now()}] SENDClient: First batch in fit for client {id(self)} (epoch {epoch+1}/{epochs})")
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
                    logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                if batch_idx == 0:
                    logger.info(f"Batch {batch_idx}, labels shape: {labels.shape}, unique labels: {torch.unique(labels)}")
                    logger.info(f"Batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {torch.unique(predictions)}")
            # Metrics per epoch
            mean_loss = np.mean(batch_losses)
            acc = (np.array(all_predictions) == np.array(all_labels)).mean()
            der = self.calculate_der(all_predictions, all_labels)
            logger.info(f"[DEBUG] Epoch {epoch+1}/{epochs} unique labels: {np.unique(all_labels)}")
            logger.info(f"[DEBUG] Epoch {epoch+1}/{epochs} unique predictions: {np.unique(all_predictions)}")
            logger.info(f"[{datetime.now()}] SENDClient: Epoch {epoch+1}/{epochs} summary for client {id(self)}: min_loss={min(batch_losses):.4f}, max_loss={max(batch_losses):.4f}, mean_loss={mean_loss:.4f}, acc={acc:.4f}, DER={der:.4f}")
        elapsed = time.time() - start_time
        logger.info(f"[{datetime.now()}] SENDClient: Finished fit for client {id(self)}, total time: {elapsed:.2f} sec")
        return self.get_parameters({}), len(self.train_loader), {"train_loss": mean_loss}
    
    def evaluate(self, parameters, config):
        logger.info(f"[{datetime.now()}] SENDClient: Starting evaluate for client {id(self)}")
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
                    logger.info(f"[{datetime.now()}] SENDClient: First batch in evaluate for client {id(self)}")
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
                    logger.info(f"Eval batch {batch_idx}, labels shape: {labels.shape}, unique labels: {np.unique(labels.cpu().numpy())}")
                    logger.info(f"Eval batch {batch_idx}, outputs shape: {outputs.shape}, unique preds: {np.unique(predictions.cpu().numpy())}")
        logger.info(f"[{datetime.now()}] SENDClient: Eval summary for client {id(self)}: min_loss={min(batch_losses):.4f}, max_loss={max(batch_losses):.4f}, mean_loss={np.mean(batch_losses):.4f}")
        elapsed = time.time() - start_time
        logger.info(f"[{datetime.now()}] SENDClient: Finished evaluate for client {id(self)}, total time: {elapsed:.2f} sec")
        der = self.calculate_der(all_predictions, all_labels)
        return (
            float(val_loss / len(self.val_loader)),
            len(self.val_loader),
            {"val_loss": val_loss / len(self.val_loader), "der": der}
        )
    
    def calculate_der(self, predictions: List[int], labels: List[int]) -> float:
        """Calculate Diarization Error Rate."""
        reference = Annotation()
        hypothesis = Annotation()
        valid_frames = 0
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if isinstance(label, str):
                if label == '-' or not label.isdigit():
                    continue
                label = int(label)
            if label == -100:
                continue  # skip padded frames
            valid_frames += 1
            # Convert power set encoded values back to speaker labels
            pred_speakers = self.power_set_encoder.decode(pred)
            true_speakers = self.power_set_encoder.decode(label)
            # Add segments to reference and hypothesis
            for speaker in true_speakers:
                if speaker == 1:
                    reference[Segment(i, i+1)] = f"speaker_{speaker}"
            for speaker in pred_speakers:
                if speaker == 1:
                    hypothesis[Segment(i, i+1)] = f"speaker_{speaker}"
        # Calculate DER
        metric = DiarizationErrorRate()
        der = metric(reference, hypothesis)
        logger.info(f"[DEBUG] DER calculation: valid frames used = {valid_frames}")
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
        logger.info(f"Client attempting to connect to {server_address}")
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client
        )
        logger.info("Client finished successfully")
    except Exception as e:
        logger.error(f"Client error: {str(e)}")

def main():
    print("MAIN STARTED")
    logger.info(f"[{datetime.now()}] MAIN: Starting main()")
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[{datetime.now()}] MAIN: Using device: {device}")
        
        # Initialize speaker encoder
        logger.info(f"[{datetime.now()}] MAIN: Initializing speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": device}
        ).to(device)
        logger.info(f"[{datetime.now()}] MAIN: Speaker encoder initialized successfully")
        
        # Load and preprocess data
        logger.info(f"[{datetime.now()}] MAIN: Loading AMI dataset...")
        dataset = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True)
        logger.info(f"[{datetime.now()}] MAIN: Dataset loaded successfully")
        
        # Take a small subset for testing
        test_size = 6
        logger.info(f"[{datetime.now()}] MAIN: Using subset of {test_size} samples for testing")
        
        # Group data by meeting ID for all splits
        logger.info(f"[{datetime.now()}] MAIN: Grouping data by meeting ID...")
        grouped_train = group_by_meeting(dataset["train"].select(range(test_size)))
        grouped_validation = group_by_meeting(dataset["validation"].select(range(test_size)))
        grouped_test = group_by_meeting(dataset["test"].select(range(test_size)))
        
        logger.info(f"[{datetime.now()}] MAIN: Grouped {len(grouped_train)} meetings from training set")
        logger.info(f"[{datetime.now()}] MAIN: Grouped {len(grouped_validation)} meetings from validation set")
        logger.info(f"[{datetime.now()}] MAIN: Grouped {len(grouped_test)} meetings from test set")
        
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
        num_speakers = len(speaker_ids)
        num_classes = 2 ** num_speakers
        logger.info(f"[{datetime.now()}] MAIN: Detected {num_speakers} unique speakers, num_classes={num_classes}")
        
        # Initialize Power Set Encoder
        logger.info(f"[{datetime.now()}] MAIN: Initializing Power Set Encoder...")
        power_set_encoder = PowerSetEncoder(max_speakers=num_speakers)
        
        # Create and train model
        logger.info(f"[{datetime.now()}] MAIN: Creating SEND model...")
        model = SENDModel(num_classes=num_classes).to(device)
        
        # Split data for federated learning with fewer clients
        logger.info(f"[{datetime.now()}] MAIN: Splitting data for federated learning...")
        num_clients = 2  # reduce number of clients for testing
        client_data = split_data_for_clients(grouped_train, num_clients, speaker_encoder)
        
        # Validate client data
        if not client_data or len(client_data) < num_clients:
            raise ValueError(f"Not enough data for {num_clients} clients. Only {len(client_data) if client_data else 0} clients can be created.")
        
        logger.info(f"[{datetime.now()}] MAIN: Split data among {len(client_data)} clients")
        
        # Compute speaker embeddings for train set
        logger.info(f"[{datetime.now()}] MAIN: Computing speaker embeddings for train set...")
        speaker_to_embedding = compute_speaker_embeddings(grouped_train, speaker_encoder)
        
        # Define client function for simulation
        def client_fn(context: Context):
            cid = context.node_config['partition-id']
            logger.info(f"[client_fn] Got cid from context.node_config['partition-id']: {cid}")
            logger.info(f"[{datetime.now()}] MAIN: Creating client {cid}")
            try:
                client_idx = int(cid)
                if client_idx >= len(client_data):
                    raise ValueError(f"Client ID {client_idx} is out of range. Only {len(client_data)} clients available.")
                train_loader, val_loader = client_data[client_idx]
                # Create new model instance for each client
                client_model = SENDModel(num_classes=num_classes).to(device)
                logger.info(f"[{datetime.now()}] MAIN: Client {cid} created and ready")
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
        logger.info("Starting federated learning simulation...")
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            min_fit_clients=2,
            min_evaluate_clients=2,
            on_fit_config_fn=lambda _: {"epochs": 2},
            on_evaluate_config_fn=lambda _: {"epochs": 1},
            initial_parameters=fl.common.ndarrays_to_parameters(
                [val.cpu().numpy() for _, val in model.state_dict().items()]
            ),
        )
        
        # Calculate GPU resources
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus_per_client = max(1, num_gpus // num_clients) if num_gpus > 0 else 0
        logger.info(f"Available GPUs: {num_gpus}, GPUs per client: {gpus_per_client}")
        
        # Start simulation
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=3),
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
        der = calculate_der(all_predictions, all_labels, power_set_encoder)
        print(f"\n==================== TESTING FINISHED ====================\n")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Final DER: {der:.4f}")
        logger.info(f"Final Test Loss: {test_loss:.4f}")
        logger.info(f"Final DER: {der:.4f}")

        # Print Ray/Flower client logs after simulation
        import glob
        import os
        def print_ray_logs():
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

        print('\n==================== RAY/CLIENT LOGS ====================\n')
        print_ray_logs()
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Cleaning up...")
        # Add any necessary cleanup code here
        raise
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        logger.info("Process completed.")

if __name__ == "__main__":
    main() 