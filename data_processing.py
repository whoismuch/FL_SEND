import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa
from datasets import load_dataset
import logging

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

def split_data_for_clients(
    dataset,
    num_clients: int,
    batch_size: int = 32,
    val_split: float = 0.2
) -> List[Tuple[DataLoader, DataLoader]]:
    """Split dataset into client-specific train and validation loaders."""
    client_data = []
    
    # Group data by meeting ID
    meetings = {}
    for sample in dataset["train"]:
        meeting_id = sample["meeting_id"]
        if meeting_id not in meetings:
            meetings[meeting_id] = []
        meetings[meeting_id].append(sample)
    
    # Split meetings among clients
    meeting_ids = list(meetings.keys())
    np.random.shuffle(meeting_ids)
    meetings_per_client = len(meeting_ids) // num_clients
    
    for i in range(num_clients):
        client_meetings = meeting_ids[i * meetings_per_client:(i + 1) * meetings_per_client]
        
        # Collect all samples for this client
        client_samples = []
        for meeting_id in client_meetings:
            client_samples.extend(meetings[meeting_id])
        
        # Create dataset
        features = []
        labels = []
        for sample in client_samples:
            # Extract features
            audio = sample["audio"]["array"]
            feature = extract_features(audio)
            features.append(feature)
            
            # Get speaker label
            speaker_id = int(sample["speaker_id"].split("_")[1])
            labels.append(speaker_id)
        
        # Convert to numpy arrays
        features = np.array(features)
        labels = np.array(labels)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        
        # Split into train and validation
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
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