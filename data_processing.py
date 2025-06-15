import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa
from datasets import load_dataset
import logging
from speechbrain.pretrained import EncoderClassifier
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

logger = logging.getLogger(__name__)

def extract_features(audio: np.ndarray, sr: int = 16000, n_mels: int = 80) -> np.ndarray:
    """Extract log-mel spectrogram features from audio."""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        win_length=int(0.025 * sr),
        hop_length=int(0.01 * sr)
    )
    log_mel = librosa.power_to_db(mel_spec)
    return log_mel.T

def simulate_overlapping_speech(
    audio_segments: List[np.ndarray],
    speaker_labels: List[int],
    max_speakers: int = 4,
    duration: float = 10.0,
    sr: int = 16000
) -> Tuple[np.ndarray, List[int]]:
    """Simulate overlapping speech segments."""
    overlapping_segments = []
    power_set_labels = []
    
    for i in range(len(audio_segments)):
        for j in range(i + 1, len(audio_segments)):
            # Combine audio segments
            combined_audio = audio_segments[i] + audio_segments[j]
            
            # Normalize
            if np.max(np.abs(combined_audio)) > 0:
                combined_audio = combined_audio / np.max(np.abs(combined_audio))
            
            # Create power set encoded label
            speaker_label = [0] * max_speakers
            speaker_label[speaker_labels[i]] = 1
            speaker_label[speaker_labels[j]] = 1
            
            # Encode label
            encoded_label = sum(label * (2 ** i) for i, label in enumerate(speaker_label))
            
            overlapping_segments.append(combined_audio)
            power_set_labels.append(encoded_label)
    
    return np.array(overlapping_segments), power_set_labels

def split_data_for_clients(grouped_data, num_clients):
    """Split grouped data among clients."""
    # Convert grouped data to list of samples
    all_samples = []
    for meeting_id, samples in grouped_data.items():
        all_samples.extend(samples)
    
    # Shuffle samples
    np.random.shuffle(all_samples)
    
    # Split samples among clients
    samples_per_client = len(all_samples) // num_clients
    client_samples = [all_samples[i:i + samples_per_client] for i in range(0, len(all_samples), samples_per_client)]
    
    # Create data loaders for each client
    client_data = []
    for samples in client_samples[:num_clients]:  # Ensure we only use the requested number of clients
        # Create dataset for this client
        features = []
        labels = []
        for sample in samples:
            feature = extract_features(sample["audio"]["array"])
            label = power_set_encoding(sample["speaker_id"])
            features.append(feature)
            labels.append(label)
        
        # Split into train and validation
        train_size = int(0.8 * len(features))
        val_size = len(features) - train_size
        
        # Split features and labels
        train_features = features[:train_size]
        train_labels = labels[:train_size]
        val_features = features[train_size:]
        val_labels = labels[train_size:]
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(train_features, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(val_features, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.long)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        client_data.append((train_loader, val_loader))
    
    return client_data

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    der_scores: List[float],
    save_path: str
):
    """Plot training curves and save to file."""
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    
    # Plot DER
    plt.subplot(1, 2, 2)
    plt.plot(der_scores)
    plt.xlabel("Epoch")
    plt.ylabel("DER")
    plt.title("Diarization Error Rate")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: str
):
    """Plot confusion matrix and save to file."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()

def group_by_meeting(dataset_split):
    """Group dataset samples by meeting ID."""
    grouped = {}
    for sample in dataset_split:
        meeting_id = sample["meeting_id"]
        grouped.setdefault(meeting_id, []).append(sample)
    return grouped

def create_dataset_from_grouped(grouped_data, speaker_encoder):
    """Create dataset from grouped data."""
    features = []
    labels = []
    
    for meeting_id, samples in grouped_data.items():
        for sample in samples:
            # Extract features and labels
            feature = extract_features(sample["audio"]["array"])
            label = power_set_encoding(sample["speaker_id"])
            features.append(feature)
            labels.append(label)
    
    return OverlappingSpeechDataset(
        features=np.array(features),
        labels=np.array(labels),
        speaker_encoder=speaker_encoder
    )

def calculate_der(predictions, labels, power_set_encoder):
    """Calculate Diarization Error Rate."""
    reference = Annotation()
    hypothesis = Annotation()
    
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        # Convert power set encoded values back to speaker labels
        pred_speakers = power_set_encoder.decode(pred)
        true_speakers = power_set_encoder.decode(label)
        
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

def prepare_data_loaders(grouped_train, grouped_validation, grouped_test, speaker_encoder, batch_size=32):
    """Prepare data loaders for training, validation and testing."""
    # Create datasets
    train_dataset = create_dataset_from_grouped(grouped_train, speaker_encoder)
    val_dataset = create_dataset_from_grouped(grouped_validation, speaker_encoder)
    test_dataset = create_dataset_from_grouped(grouped_test, speaker_encoder)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def power_set_encoding(label):
    """Encodes speaker label into a single integer using power-set encoding."""
    if isinstance(label, (list, tuple)):
        return sum([l * (2 ** i) for i, l in enumerate(label) if l in [0, 1]])
    else:
        # If single label, create a list with one element
        return 2 ** label if label > 0 else 0

class OverlappingSpeechDataset(Dataset):
    """Dataset for overlapping speech diarization."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, speaker_encoder: EncoderClassifier):
        self.features = features
        self.labels = labels
        self.speaker_encoder = speaker_encoder
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple:
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Extract speaker embeddings
        with torch.no_grad():
            speaker_embedding = self.speaker_encoder.encode_batch(feature.unsqueeze(0))
            speaker_embedding = speaker_embedding.squeeze(0)
        
        return feature, speaker_embedding, label 