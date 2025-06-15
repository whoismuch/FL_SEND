# FL-SEND-PSE: Federated Learning for Speaker Embedding-aware Neural Diarization with Power-Set Encoding
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
import logging
from sklearn.metrics import confusion_matrix
import json
from datetime import datetime
from data_processing import (
    split_data_for_clients, 
    extract_features, 
    simulate_overlapping_speech, 
    group_by_meeting,
    prepare_data_loaders,
    power_set_encoding,
    calculate_der
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.memory = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
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
        
        # Process through Post-Net
        for fsmn in self.post_net:
            combined = fsmn(combined)
        
        # Final classification
        out = self.classifier(combined)  # (batch_size, sequence_length, num_classes)
        
        return out

class OverlappingSpeechDataset(Dataset):
    """Dataset for overlapping speech diarization."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, speaker_encoder: EncoderClassifier):
        self.features = features
        self.labels = labels
        self.speaker_encoder = speaker_encoder
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Extract speaker embeddings
        with torch.no_grad():
            speaker_embedding = self.speaker_encoder.encode_batch(feature.unsqueeze(0))
            speaker_embedding = speaker_embedding.squeeze(0)
        
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
        speaker_encoder: EncoderClassifier
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.power_set_encoder = power_set_encoder
        self.speaker_encoder = speaker_encoder
        self.optimizer = optim.Adam(model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Train the model
        self.model.train()
        train_loss = 0.0
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Get speaker embeddings for all speakers
            with torch.no_grad():
                speaker_embeddings = []
                for i in range(self.power_set_encoder.max_speakers):
                    # Create a dummy audio segment for each speaker
                    dummy_audio = torch.zeros(1, 16000).to(self.device)  # 1 second of silence
                    embedding = self.speaker_encoder.encode_batch(dummy_audio)
                    speaker_embeddings.append(embedding)
                speaker_embeddings = torch.stack(speaker_embeddings, dim=1)  # (batch_size, num_speakers, 192)
                # Expand speaker embeddings to match batch size
                speaker_embeddings = speaker_embeddings.expand(features.size(0), -1, -1)
            
            self.optimizer.zero_grad()
            outputs = self.model(features, speaker_embeddings)
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
        
        return self.get_parameters({}), len(self.train_loader), {"train_loss": train_loss / len(self.train_loader)}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Evaluate the model
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Get speaker embeddings for all speakers
                speaker_embeddings = []
                for i in range(self.power_set_encoder.max_speakers):
                    dummy_audio = torch.zeros(1, 16000).to(self.device)
                    embedding = self.speaker_encoder.encode_batch(dummy_audio)
                    speaker_embeddings.append(embedding)
                speaker_embeddings = torch.stack(speaker_embeddings, dim=1)
                # Expand speaker embeddings to match batch size
                speaker_embeddings = speaker_embeddings.expand(features.size(0), -1, -1)
                
                outputs = self.model(features, speaker_embeddings)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate DER
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
        
        for i, (pred, label) in enumerate(zip(predictions, labels)):
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
    try:
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize speaker encoder
        logger.info("Initializing speaker encoder...")
        speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": device}
        ).to(device)
        logger.info("Speaker encoder initialized successfully")
        
        # Load and preprocess data
        logger.info("Loading AMI dataset...")
        dataset = load_dataset("edinburghcstr/ami", "ihm", trust_remote_code=True)
        logger.info("Dataset loaded successfully")
        
        # Take a small subset for testing
        test_size = 1000  # small number for quick testing
        logger.info(f"Using subset of {test_size} samples for testing")
        
        # Group data by meeting ID for all splits
        logger.info("Grouping data by meeting ID...")
        grouped_train = group_by_meeting(dataset["train"].select(range(test_size)))
        grouped_validation = group_by_meeting(dataset["validation"].select(range(test_size)))
        grouped_test = group_by_meeting(dataset["test"].select(range(test_size)))
        
        logger.info(f"Grouped {len(grouped_train)} meetings from training set")
        logger.info(f"Grouped {len(grouped_validation)} meetings from validation set")
        logger.info(f"Grouped {len(grouped_test)} meetings from test set")
        
        # Initialize Power Set Encoder
        logger.info("Initializing Power Set Encoder...")
        power_set_encoder = PowerSetEncoder(max_speakers=4)
        
        # Create and train model
        logger.info("Creating SEND model...")
        model = SENDModel().to(device)
        
        # Split data for federated learning with fewer clients
        logger.info("Splitting data for federated learning...")
        num_clients = 2  # reduce number of clients for testing
        client_data = split_data_for_clients(grouped_train, num_clients)
        
        # Validate client data
        if not client_data or len(client_data) < num_clients:
            raise ValueError(f"Not enough data for {num_clients} clients. Only {len(client_data) if client_data else 0} clients can be created.")
        
        logger.info(f"Split data among {len(client_data)} clients")
        
        # Define client function for simulation
        def client_fn(cid: str):
            """Create a client for the simulation."""
            try:
                client_idx = int(cid)
                if client_idx >= len(client_data):
                    raise ValueError(f"Client ID {client_idx} is out of range. Only {len(client_data)} clients available.")
                
                train_loader, val_loader = client_data[client_idx]
                # Create new model instance for each client
                client_model = SENDModel().to(device)
                return SENDClient(
                    model=client_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    power_set_encoder=power_set_encoder,
                    speaker_encoder=speaker_encoder
                )
            except Exception as e:
                logger.error(f"Error creating client {cid}: {str(e)}")
                raise
        
        # Start federated learning with simulation
        logger.info("Starting federated learning simulation...")
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=2,
            min_fit_clients=2,
            min_evaluate_clients=2,
            on_fit_config_fn=lambda _: {"epochs": 3},
            on_evaluate_config_fn=lambda _: {"epochs": 3},
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
        logger.info("Performing final evaluation on test set...")
        _, _, test_loader = prepare_data_loaders(grouped_train, grouped_validation, grouped_test, speaker_encoder)
        
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
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate final metrics
        test_loss = test_loss / len(test_loader)
        der = calculate_der(all_predictions, all_labels, power_set_encoder)
        
        logger.info(f"Final Test Loss: {test_loss:.4f}")
        logger.info(f"Final DER: {der:.4f}")
        
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