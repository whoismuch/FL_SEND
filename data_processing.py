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
    # Ensure minimum length for FFT
    min_length = 2048  # minimum length for FFT
    if len(audio) < min_length:
        audio = np.pad(audio, (0, min_length - len(audio)))
    
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
    sr: int = 16000,
    max_combinations: int = 1000  # Maximum number of combinations
) -> Tuple[np.ndarray, List[int]]:
    """Simulate overlapping speech segments.
    
    Args:
        audio_segments: List of audio segments
        speaker_labels: List of speaker labels corresponding to audio segments
        max_speakers: Maximum number of speakers
        duration: Target duration for each segment in seconds
        sr: Sample rate
        max_combinations: Maximum number of combinations to generate
        
    Returns:
        Tuple of (overlapping_segments, power_set_labels)
    """
    logger.info(f"Starting simulation with {len(audio_segments)} segments")
    
    # Limit segments for processing
    max_segments = min(len(audio_segments), 100)  # Reduce to 100 segments
    if len(audio_segments) > max_segments:
        logger.info(f"Limiting segments from {len(audio_segments)} to {max_segments}")
        indices = np.random.choice(len(audio_segments), max_segments, replace=False)
        audio_segments = [audio_segments[i] for i in indices]
        speaker_labels = [speaker_labels[i] for i in indices]
    
    overlapping_segments = []
    power_set_labels = []
    
    # Convert duration to samples
    target_length = int(duration * sr)
    
    # Create list of possible combinations
    combinations = []
    for i in range(len(audio_segments)):
        for j in range(i + 1, len(audio_segments)):
            combinations.append((i, j))
    
    # Limit combinations
    if len(combinations) > max_combinations:
        logger.info(f"Limiting combinations from {len(combinations)} to {max_combinations}")
        # Convert combinations list to array of indices
        combination_indices = np.random.choice(len(combinations), max_combinations, replace=False)
        combinations = [combinations[i] for i in combination_indices]
    
    logger.info(f"Processing {len(combinations)} combinations")
    processed = 0
    
    for i, j in combinations:
        try:
            # Get segments
            seg1 = audio_segments[i]
            seg2 = audio_segments[j]
            
            # Ensure both segments have the same length
            if len(seg1) > target_length:
                # If segment is longer than target, take a random slice
                start1 = np.random.randint(0, len(seg1) - target_length)
                seg1 = seg1[start1:start1 + target_length]
            elif len(seg1) < target_length:
                # If segment is shorter, pad with zeros
                pad_length = target_length - len(seg1)
                seg1 = np.pad(seg1, (0, pad_length))
            
            if len(seg2) > target_length:
                # If segment is longer than target, take a random slice
                start2 = np.random.randint(0, len(seg2) - target_length)
                seg2 = seg2[start2:start2 + target_length]
            elif len(seg2) < target_length:
                # If segment is shorter, pad with zeros
                pad_length = target_length - len(seg2)
                seg2 = np.pad(seg2, (0, pad_length))
            
            # Combine audio segments
            combined_audio = seg1 + seg2
            
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
            
            processed += 1
            if processed % 100 == 0:
                logger.info(f"Processed {processed}/{len(combinations)} combinations")
        
        except KeyboardInterrupt:
            logger.info("\nInterrupted during segment processing. Saving progress...")
            return np.array(overlapping_segments), power_set_labels
        except Exception as e:
            logger.error(f"Error processing segments {i} and {j}: {str(e)}")
            continue
    
    logger.info(f"Successfully created {len(overlapping_segments)} overlapping segments")
    return np.array(overlapping_segments), power_set_labels

def split_data_for_clients(grouped_data, num_clients, min_overlap_ratio=0.3):
    """Split grouped data among clients.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        num_clients: Number of clients to split data among
        min_overlap_ratio: Minimum ratio of overlapping samples to total samples
    """
    try:
        logger.info("Starting data processing for clients...")
        
        # Create speaker ID to index mapping
        speaker_to_idx = {}
        for meeting_id, samples in grouped_data.items():
            for sample in samples:
                speaker_id = sample["speaker_id"]
                if speaker_id not in speaker_to_idx:
                    speaker_to_idx[speaker_id] = len(speaker_to_idx)
        
        logger.info(f"Found {len(speaker_to_idx)} unique speakers: {speaker_to_idx}")
        
        # Convert grouped data to list of samples
        all_samples = []
        logger.info("Processing meetings for overlapping segments...")
        
        # Statistics by segment types
        total_meetings = len(grouped_data)
        total_original_segments = 0
        total_natural_overlaps = 0
        total_artificial_overlaps = 0
        
        for meeting_id, samples in grouped_data.items():
            try:
                logger.info(f"Processing meeting {meeting_id} with {len(samples)} samples")
                meeting_original_segments = 0
                meeting_natural_overlaps = 0
                
                # Sort samples by begin_time
                samples = sorted(samples, key=lambda x: x["begin_time"])
                
                # First, add original non-overlapping segments
                logger.info("Adding non-overlapping segments...")
                for sample in samples:
                    # Create power set encoded label for single speaker
                    max_speakers = len(speaker_to_idx)
                    speaker_label = [0] * max_speakers
                    speaker_idx = speaker_to_idx[sample["speaker_id"]]
                    speaker_label[speaker_idx] = 1
                    encoded_label = sum(label * (2 ** i) for i, label in enumerate(speaker_label))
                    
                    all_samples.append({
                        "audio": sample["audio"],
                        "speaker_id": encoded_label,
                        "begin_time": sample["begin_time"],
                        "end_time": sample["end_time"],
                        "is_overlap": False,
                        "is_artificial": False
                    })
                    meeting_original_segments += 1
                
                # Then, find real overlapping segments
                logger.info("Finding real overlapping segments...")
                overlapping_segments = []
                for i in range(len(samples)):
                    current = samples[i]
                    # Look for overlapping segments
                    for j in range(i + 1, len(samples)):
                        next_seg = samples[j]
                        # Check if segments overlap
                        if next_seg["begin_time"] < current["end_time"]:
                            logger.debug(f"Found overlap between segments {i} and {j}")
                            # Create overlapping segment
                            overlap_begin = max(current["begin_time"], next_seg["begin_time"])
                            overlap_end = min(current["end_time"], next_seg["end_time"])
                            
                            # Get audio segments
                            current_audio = current["audio"]["array"]
                            next_audio = next_seg["audio"]["array"]
                            
                            # Calculate overlap duration in samples
                            current_start = int((overlap_begin - current["begin_time"]) * 16000)  # assuming 16kHz
                            current_end = int((overlap_end - current["begin_time"]) * 16000)
                            next_start = int((overlap_begin - next_seg["begin_time"]) * 16000)
                            next_end = int((overlap_end - next_seg["begin_time"]) * 16000)
                            
                            # Extract overlapping portions
                            current_overlap = current_audio[current_start:current_end]
                            next_overlap = next_audio[next_start:next_end]
                            
                            # Ensure both segments have the same length
                            min_length = min(len(current_overlap), len(next_overlap))
                            current_overlap = current_overlap[:min_length]
                            next_overlap = next_overlap[:min_length]
                            
                            # Combine audio segments
                            combined_audio = current_overlap + next_overlap
                            
                            # Normalize
                            if np.max(np.abs(combined_audio)) > 0:
                                combined_audio = combined_audio / np.max(np.abs(combined_audio))
                            
                            # Create power set encoded label
                            max_speakers = len(speaker_to_idx)
                            speaker_label = [0] * max_speakers
                            current_speaker_idx = speaker_to_idx[current["speaker_id"]]
                            next_speaker_idx = speaker_to_idx[next_seg["speaker_id"]]
                            speaker_label[current_speaker_idx] = 1
                            speaker_label[next_speaker_idx] = 1
                            encoded_label = sum(label * (2 ** i) for i, label in enumerate(speaker_label))
                            
                            overlapping_segments.append({
                                "audio": {"array": combined_audio},
                                "speaker_id": encoded_label,
                                "begin_time": overlap_begin,
                                "end_time": overlap_end,
                                "is_overlap": True,
                                "is_artificial": False
                            })
                            meeting_natural_overlaps += 1
                
                # If we don't have enough overlapping segments, create artificial ones
                if len(overlapping_segments) < len(samples) * min_overlap_ratio:
                    logger.info("Creating artificial overlapping segments...")
                    # Group samples by speaker
                    speaker_segments = {}
                    for sample in samples:
                        speaker_id = sample["speaker_id"]
                        if speaker_id not in speaker_segments:
                            speaker_segments[speaker_id] = []
                        speaker_segments[speaker_id].append(sample["audio"]["array"])
                    
                    # Create artificial overlapping segments
                    audio_segments = []
                    speaker_labels = []
                    for speaker_id, segments in speaker_segments.items():
                        for segment in segments:
                            audio_segments.append(segment)
                            speaker_labels.append(speaker_to_idx[speaker_id])
                    
                    logger.info(f"Simulating overlapping speech with {len(audio_segments)} segments...")
                    # Simulate overlapping speech
                    artificial_segments, power_set_labels = simulate_overlapping_speech(
                        audio_segments=audio_segments,
                        speaker_labels=speaker_labels,
                        max_speakers=len(speaker_to_idx)
                    )
                    
                    # Add artificial overlapping segments
                    for segment, label in zip(artificial_segments, power_set_labels):
                        overlapping_segments.append({
                            "audio": {"array": segment},
                            "speaker_id": label,
                            "begin_time": 0,  # Artificial segments don't have real timestamps
                            "end_time": 0,
                            "is_overlap": True,
                            "is_artificial": True
                        })
                
                # Add overlapping segments
                all_samples.extend(overlapping_segments)
                
                # Update statistics
                total_original_segments += meeting_original_segments
                total_natural_overlaps += meeting_natural_overlaps
                total_artificial_overlaps += len(overlapping_segments) - meeting_natural_overlaps
                
                logger.info(f"Meeting {meeting_id} statistics:")
                logger.info(f"  - Original segments: {meeting_original_segments}")
                logger.info(f"  - Natural overlaps: {meeting_natural_overlaps}")
                logger.info(f"  - Artificial overlaps: {len(overlapping_segments) - meeting_natural_overlaps}")
            
            except KeyboardInterrupt:
                logger.info(f"\nInterrupted while processing meeting {meeting_id}. Saving progress...")
                break
            except Exception as e:
                logger.error(f"Error processing meeting {meeting_id}: {str(e)}")
                continue
        
        # Print overall statistics
        logger.info("\nOverall Dataset Statistics:")
        logger.info(f"Total meetings processed: {total_meetings}")
        logger.info(f"Total original segments: {total_original_segments}")
        logger.info(f"Total natural overlaps: {total_natural_overlaps}")
        logger.info(f"Total artificial overlaps: {total_artificial_overlaps}")
        logger.info(f"Total segments: {len(all_samples)}")
        logger.info(f"Natural overlap ratio: {total_natural_overlaps/total_original_segments:.2%}")
        logger.info(f"Artificial overlap ratio: {total_artificial_overlaps/total_original_segments:.2%}")
        
        # Balance the dataset
        logger.info("\nBalancing dataset...")
        non_overlap_samples = [s for s in all_samples if not s["is_overlap"]]
        natural_overlap_samples = [s for s in all_samples if s["is_overlap"] and not s["is_artificial"]]
        artificial_overlap_samples = [s for s in all_samples if s["is_overlap"] and s["is_artificial"]]
        
        # Ensure we have a good balance between overlapping and non-overlapping samples
        min_samples = min(len(non_overlap_samples), len(natural_overlap_samples) + len(artificial_overlap_samples))
        balanced_samples = non_overlap_samples[:min_samples] + natural_overlap_samples + artificial_overlap_samples[:min_samples - len(natural_overlap_samples)]
        
        # Shuffle samples
        np.random.shuffle(balanced_samples)
        
        logger.info("\nBalanced Dataset Statistics:")
        logger.info(f"Non-overlapping samples: {len(non_overlap_samples[:min_samples])}")
        logger.info(f"Natural overlapping samples: {len(natural_overlap_samples)}")
        logger.info(f"Artificial overlapping samples: {min(min_samples - len(natural_overlap_samples), len(artificial_overlap_samples))}")
        logger.info(f"Total balanced samples: {len(balanced_samples)}")
        
        # Split samples among clients
        logger.info(f"\nSplitting {len(balanced_samples)} samples among {num_clients} clients...")
        
        # Calculate samples per client and create client data
        samples_per_client = len(balanced_samples) // num_clients
        client_samples = [balanced_samples[i:i + samples_per_client] for i in range(0, len(balanced_samples), samples_per_client)]
        
        # Create data loaders for each client
        logger.info("Creating data loaders for clients...")
        client_data = []
        for client_id, samples in enumerate(client_samples[:num_clients]):
            try:
                logger.info(f"\nProcessing client {client_id} with {len(samples)} samples")
                client_non_overlap = len([s for s in samples if not s["is_overlap"]])
                client_natural_overlap = len([s for s in samples if s["is_overlap"] and not s["is_artificial"]])
                client_artificial_overlap = len([s for s in samples if s["is_overlap"] and s["is_artificial"]])
                logger.info(f"Client {client_id} sample distribution:")
                logger.info(f"  - Non-overlapping: {client_non_overlap}")
                logger.info(f"  - Natural overlaps: {client_natural_overlap}")
                logger.info(f"  - Artificial overlaps: {client_artificial_overlap}")
                # Инициализация списков для хранения признаков и меток
                raw_features = []
                raw_labels = []
                speaker_ids = []
                for sample in samples:
                    feature = extract_features(sample["audio"]["array"])
                    if feature.shape[0] == 0:
                        logger.warning(f"Sample with empty feature sequence detected, skipping.")
                        continue
                    raw_features.append(feature)
                    # frame-wise labels
                    label = np.full(feature.shape[0], sample["speaker_id"], dtype=np.int64)
                    raw_labels.append(label)
                    speaker_ids.append(sample["speaker_id"])
                if not raw_features:
                    logger.error(f"No valid features for client {client_id}, skipping client.")
                    continue
                max_len = max(f.shape[0] for f in raw_features)
                features_padded = []
                labels_padded = []
                for feature, label in zip(raw_features, raw_labels):
                    if feature.shape[0] < max_len:
                        pad_len = max_len - feature.shape[0]
                        feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                        label = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
                    features_padded.append(feature)
                    labels_padded.append(label)
                # Diagnostics: check that all sequence lengths are the same
                lengths = [f.shape[0] for f in features_padded]
                logger.info(f"All sequence lengths for client {client_id}: {set(lengths)}")
                if len(set(lengths)) != 1:
                    logger.error(f"Inhomogeneous sequence lengths for client {client_id}, skipping client.")
                    continue
                features = np.array(features_padded)
                labels = np.array(labels_padded)
                train_size = int(0.8 * len(features))
                val_size = len(features) - train_size
                train_features = features[:train_size]
                train_labels = labels[:train_size]
                val_features = features[train_size:]
                val_labels = labels[train_size:]
                train_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(train_features, dtype=torch.float32),
                    torch.tensor(train_labels, dtype=torch.long),
                    torch.tensor(speaker_ids[:train_size], dtype=torch.long)
                )
                val_dataset = torch.utils.data.TensorDataset(
                    torch.tensor(val_features, dtype=torch.float32),
                    torch.tensor(val_labels, dtype=torch.long),
                    torch.tensor(speaker_ids[train_size:], dtype=torch.long)
                )
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
                client_data.append((train_loader, val_loader))
                logger.info(f"Created data loaders for client {client_id}")
            except KeyboardInterrupt:
                logger.info(f"\nInterrupted while processing client {client_id}. Saving progress...")
                break
            except Exception as e:
                logger.error(f"Error processing client {client_id}: {str(e)}")
                continue
        
        if not client_data:
            raise ValueError("No client data was created successfully")
        
        logger.info("\nData processing completed successfully")
        return client_data
    
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user. Saving progress...")
        if 'client_data' in locals() and client_data:
            return client_data
        return []
    except Exception as e:
        logger.error(f"An error occurred during data processing: {str(e)}")
        return []

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
    """Create dataset from grouped data with proper padding.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        speaker_encoder: Speaker encoder model
        
    Returns:
        OverlappingSpeechDataset: Dataset with padded features
    """
    features = []
    labels = []
    raw_features = []
    raw_labels = []
    speaker_ids = []
    
    # Create speaker ID to index mapping
    speaker_to_idx = {}
    for meeting_id, samples in grouped_data.items():
        for sample in samples:
            speaker_id = sample["speaker_id"]
            if speaker_id not in speaker_to_idx:
                speaker_to_idx[speaker_id] = len(speaker_to_idx)
    
    logger.info(f"Created speaker ID mapping: {speaker_to_idx}")
    
    # First pass: extract features and find max length
    for meeting_id, samples in grouped_data.items():
        for sample in samples:
            # Extract features
            feature = extract_features(sample["audio"]["array"])
            raw_features.append(feature)
            # Convert string speaker ID to numeric index
            speaker_idx = speaker_to_idx[sample["speaker_id"]]
            label = power_set_encoding(speaker_idx)
            # frame-wise labels
            raw_labels.append(np.full(feature.shape[0], label, dtype=np.int64))
            speaker_ids.append(sample["speaker_id"])
    
    # Find max sequence length
    max_len = max(f.shape[0] for f in raw_features)
    logger.info(f"Max sequence length: {max_len}")
    
    # Second pass: pad all features to max length
    for feature, label in zip(raw_features, raw_labels):
        if feature.shape[0] < max_len:
            pad_len = max_len - feature.shape[0]
            feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
            label = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
        features.append(feature)
        labels.append(label)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    
    logger.info(f"Created dataset with shape: {features.shape}")
    
    return OverlappingSpeechDataset(
        features=features,
        labels=labels,
        speaker_ids=speaker_ids,
        speaker_to_embedding=compute_speaker_embeddings(grouped_data, speaker_encoder)
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

def prepare_data_loaders(grouped_train, grouped_validation, grouped_test, speaker_encoder, batch_size=4, speaker_to_embedding=None):
    """Prepare data loaders for training, validation and testing."""
    # Create datasets
    train_dataset = create_dataset_from_grouped(grouped_train, speaker_encoder)
    val_dataset = create_dataset_from_grouped(grouped_validation, speaker_encoder)
    test_dataset = create_dataset_from_grouped(grouped_test, speaker_encoder)
    
    # Create data loaders with collate function
    def collate_fn(batch):
        max_len = max(x[0].shape[0] for x in batch)
        features = []
        speaker_embeddings = []
        labels = []
        for feature, all_embeddings, label in batch:
            if feature.shape[0] < max_len:
                pad_len = max_len - feature.shape[0]
                feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
            features.append(feature)
            speaker_embeddings.append(all_embeddings)  # [num_speakers, 192]
            labels.append(label)
        features = torch.tensor(np.array(features), dtype=torch.float32)
        speaker_embeddings = torch.stack(speaker_embeddings)  # [batch, num_speakers, 192]
        labels = torch.tensor(labels, dtype=torch.long)
        return features, speaker_embeddings, labels
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    logger.info(f"Created data loaders with batch size {batch_size}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def power_set_encoding(label):
    """Encodes speaker label into a single integer using power-set encoding.
    
    Args:
        label: Either a single speaker index (int) or a list of speaker indices
        
    Returns:
        int: Encoded value representing the combination of speakers
        
    Example:
        For 4 speakers:
        - Single speaker: power_set_encoding(2) -> 4 (0100 in binary)
        - Multiple speakers: power_set_encoding([0, 2]) -> 5 (0101 in binary)
    """
    if isinstance(label, (list, tuple)):
        # For multiple speakers, set bits for each speaker
        return sum(2 ** i for i in label)
    else:
        # For single speaker, set bit for that speaker
        return 2 ** label

def demonstrate_power_set_encoding():
    """Demonstrates how power set encoding works for overlapping speech."""
    # Example with 4 speakers
    max_speakers = 4
    num_classes = 2 ** max_speakers
    
    print(f"Power Set Encoding for {max_speakers} speakers:")
    print(f"Total possible combinations: {num_classes}")
    print("\nExamples:")
    
    # Single speaker cases
    for i in range(max_speakers):
        encoded = power_set_encoding(i)
        binary = format(encoded, f'0{max_speakers}b')
        print(f"Speaker {i} only: {encoded} (binary: {binary})")
    
    # Multiple speaker cases
    examples = [
        [0, 1],  # Speakers 0 and 1
        [1, 2],  # Speakers 1 and 2
        [0, 2],  # Speakers 0 and 2
        [0, 1, 2]  # Speakers 0, 1, and 2
    ]
    
    print("\nOverlapping speech examples:")
    for speakers in examples:
        encoded = power_set_encoding(speakers)
        binary = format(encoded, f'0{max_speakers}b')
        print(f"Speakers {speakers}: {encoded} (binary: {binary})")

def compute_speaker_embeddings(grouped_data, speaker_encoder):
    """Вычисляет speaker embedding для каждого уникального speaker_id по его первому аудиосегменту."""
    speaker_to_audio = {}
    for meeting_id, samples in grouped_data.items():
        for sample in samples:
            sid = sample["speaker_id"]
            if sid not in speaker_to_audio:
                speaker_to_audio[sid] = []
            speaker_to_audio[sid].append(sample["audio"]["array"])
    speaker_to_embedding = {}
    for sid, audio_list in speaker_to_audio.items():
        audio = audio_list[0]
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, time]
        with torch.no_grad():
            emb = speaker_encoder.encode_batch(audio_tensor)
            emb = emb.squeeze(0).cpu()
        speaker_to_embedding[sid] = emb
    return speaker_to_embedding

class OverlappingSpeechDataset(Dataset):
    """Dataset for overlapping speech diarization."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, speaker_ids: list, speaker_to_embedding: dict):
        self.features = features
        self.labels = labels
        self.speaker_ids = speaker_ids
        self.speaker_to_embedding = speaker_to_embedding
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int) -> tuple:
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sid = self.speaker_ids[idx]
        # Для SEND-style: возвращаем embedding всех спикеров (матрицу)
        all_embeddings = torch.stack([self.speaker_to_embedding[s] for s in sorted(self.speaker_to_embedding.keys())])
        return feature, all_embeddings, label 