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
from dataset_statistics import print_function_start, print_function_end, print_data_loaders_info, print_dataset_statistics

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
    power_set_encoder,
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
            
            # Create power set encoded label using PowerSetEncoder
            active_speakers = [speaker_labels[i], speaker_labels[j]]
            encoded_label = power_set_encoder.encode(active_speakers)
            
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

def process_training_data(grouped_data, speaker_encoder, power_set_encoder):
    """Process training data and create samples with overlapping segments.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples (training data)
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder for encoding speaker combinations
    
    Returns:
        tuple: (all_samples, speaker_to_idx)
    """
    logger.info("Processing training data...")
    
    # Create speaker ID to index mapping from training data only
    speaker_to_idx = {}
    for meeting_id, samples in grouped_data.items():
        for sample in samples:
            speaker_id = sample["speaker_id"]
            if speaker_id not in speaker_to_idx:
                speaker_to_idx[speaker_id] = len(speaker_to_idx)
    
    logger.info(f"Found {len(speaker_to_idx)} unique speakers in training data: {speaker_to_idx}")
    
    # Convert grouped data to list of samples
    all_samples = []
    logger.info("Processing training meetings for overlapping segments...")
    
    # Statistics by segment types
    total_meetings = len(grouped_data)
    total_original_segments = 0
    total_natural_overlaps = 0
    
    for meeting_id, samples in grouped_data.items():
        try:
            logger.info(f"Processing training meeting {meeting_id} with {len(samples)} samples")
            meeting_original_segments = 0
            meeting_natural_overlaps = 0
            
            # Sort samples by begin_time
            samples = sorted(samples, key=lambda x: x["begin_time"])
            
            # First, add original non-overlapping segments
            logger.info("Adding non-overlapping segments...")
            for sample in samples:
                # Create power set encoded label for single speaker using PowerSetEncoder
                speaker_idx = speaker_to_idx[sample["speaker_id"]]
                encoded_label = power_set_encoder.encode([speaker_idx])
                
                all_samples.append({
                    "audio": sample["audio"],
                    "speaker_id": encoded_label,
                    "begin_time": sample["begin_time"],
                    "end_time": sample["end_time"],
                    "is_overlap": False
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
                        
                        # Create power set encoded label for overlapping speakers using PowerSetEncoder
                        current_speaker_idx = speaker_to_idx[current["speaker_id"]]
                        next_speaker_idx = speaker_to_idx[next_seg["speaker_id"]]
                        active_speakers = [current_speaker_idx, next_speaker_idx]
                        encoded_label = power_set_encoder.encode(active_speakers)
                        
                        overlapping_segments.append({
                            "audio": {"array": combined_audio},
                            "speaker_id": encoded_label,
                            "begin_time": overlap_begin,
                            "end_time": overlap_end,
                            "is_overlap": True
                        })
                        meeting_natural_overlaps += 1
            
            # Use only natural overlapping segments (no artificial ones)
            logger.info(f"Found {len(overlapping_segments)} natural overlapping segments")
            
            # Add overlapping segments
            all_samples.extend(overlapping_segments)
            
            # Update statistics
            total_original_segments += meeting_original_segments
            total_natural_overlaps += meeting_natural_overlaps
            
            logger.info(f"Training meeting {meeting_id} statistics:")
            logger.info(f"  - Original segments: {meeting_original_segments}")
            logger.info(f"  - Natural overlaps: {meeting_natural_overlaps}")
        
        except KeyboardInterrupt:
            logger.info(f"\nInterrupted while processing training meeting {meeting_id}. Saving progress...")
            break
        except Exception as e:
            logger.error(f"Error processing training meeting {meeting_id}: {str(e)}")
            continue
    
    # Print training statistics
    logger.info("\nTraining Dataset Statistics:")
    logger.info(f"Training meetings processed: {total_meetings}")
    logger.info(f"Training original segments: {total_original_segments}")
    logger.info(f"Training natural overlaps: {total_natural_overlaps}")
    logger.info(f"Training total segments: {len(all_samples)}")
    logger.info(f"Training natural overlap ratio: {total_natural_overlaps/total_original_segments:.2%}")
    
    return all_samples, speaker_to_idx


def process_validation_data(grouped_validation, speaker_encoder, power_set_encoder):
    """Process validation data and create samples with overlapping segments.
    
    Args:
        grouped_validation: Dictionary of meeting_id to samples (validation data)
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder for encoding speaker combinations
    
    Returns:
        tuple: (val_samples, val_speaker_to_idx)
    """
    logger.info("Processing validation data...")
    
    if not grouped_validation:
        logger.warning("No validation data provided - returning empty validation set")
        return [], {}
    
    # Create separate speaker ID to index mapping for validation data only
    val_speaker_to_idx = {}
    for meeting_id, samples in grouped_validation.items():
        for sample in samples:
            speaker_id = sample["speaker_id"]
            if speaker_id not in val_speaker_to_idx:
                val_speaker_to_idx[speaker_id] = len(val_speaker_to_idx)
    
    logger.info(f"Found {len(val_speaker_to_idx)} unique speakers in validation data: {val_speaker_to_idx}")
    
    val_samples = []
    val_total_meetings = len(grouped_validation)
    val_total_original_segments = 0
    val_total_natural_overlaps = 0
    
    for meeting_id, samples in grouped_validation.items():
        try:
            logger.info(f"Processing validation meeting {meeting_id} with {len(samples)} samples")
            meeting_original_segments = 0
            meeting_natural_overlaps = 0
            
            # Sort samples by begin_time
            samples = sorted(samples, key=lambda x: x["begin_time"])
            
            # Add original non-overlapping segments
            for sample in samples:
                speaker_idx = val_speaker_to_idx[sample["speaker_id"]]
                encoded_label = power_set_encoder.encode([speaker_idx])
                
                val_samples.append({
                    "audio": sample["audio"],
                    "speaker_id": encoded_label,
                    "begin_time": sample["begin_time"],
                    "end_time": sample["end_time"],
                    "is_overlap": False
                })
                meeting_original_segments += 1
            
            # Find real overlapping segments
            overlapping_segments = []
            for i in range(len(samples)):
                current = samples[i]
                for j in range(i + 1, len(samples)):
                    next_seg = samples[j]
                    if next_seg["begin_time"] < current["end_time"]:
                        overlap_begin = max(current["begin_time"], next_seg["begin_time"])
                        overlap_end = min(current["end_time"], next_seg["end_time"])
                        
                        current_audio = current["audio"]["array"]
                        next_audio = next_seg["audio"]["array"]
                        
                        current_start = int((overlap_begin - current["begin_time"]) * 16000)
                        current_end = int((overlap_end - current["begin_time"]) * 16000)
                        next_start = int((overlap_begin - next_seg["begin_time"]) * 16000)
                        next_end = int((overlap_end - next_seg["begin_time"]) * 16000)
                        
                        current_overlap = current_audio[current_start:current_end]
                        next_overlap = next_audio[next_start:next_end]
                        
                        min_length = min(len(current_overlap), len(next_overlap))
                        current_overlap = current_overlap[:min_length]
                        next_overlap = next_overlap[:min_length]
                        
                        combined_audio = current_overlap + next_overlap
                        
                        if np.max(np.abs(combined_audio)) > 0:
                            combined_audio = combined_audio / np.max(np.abs(combined_audio))
                        
                        current_speaker_idx = val_speaker_to_idx[current["speaker_id"]]
                        next_speaker_idx = val_speaker_to_idx[next_seg["speaker_id"]]
                        active_speakers = [current_speaker_idx, next_speaker_idx]
                        encoded_label = power_set_encoder.encode(active_speakers)
                        
                        overlapping_segments.append({
                            "audio": {"array": combined_audio},
                            "speaker_id": encoded_label,
                            "begin_time": overlap_begin,
                            "end_time": overlap_end,
                            "is_overlap": True
                        })
                        meeting_natural_overlaps += 1
            
            val_samples.extend(overlapping_segments)
            val_total_original_segments += meeting_original_segments
            val_total_natural_overlaps += meeting_natural_overlaps
            
        except Exception as e:
            logger.error(f"Error processing validation meeting {meeting_id}: {str(e)}")
            continue
    
    # Print validation statistics
    logger.info(f"Validation meetings processed: {val_total_meetings}")
    logger.info(f"Validation original segments: {val_total_original_segments}")
    logger.info(f"Validation natural overlaps: {val_total_natural_overlaps}")
    logger.info(f"Validation total segments: {len(val_samples)}")
    if val_total_original_segments > 0:
        logger.info(f"Validation natural overlap ratio: {val_total_natural_overlaps/val_total_original_segments:.2%}")
    else:
        logger.info("Validation natural overlap ratio: N/A (no original segments)")
    
    return val_samples, val_speaker_to_idx


def split_data_for_clients(grouped_data, grouped_validation, num_clients, speaker_encoder, power_set_encoder):
    """Split grouped data among clients.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples (training data)
        grouped_validation: Dictionary of meeting_id to samples (validation data)
        num_clients: Number of clients to split data among
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder for encoding speaker combinations
    """
    try:
        logger.info("Starting data processing for clients...")
        
        # Validate input data
        if not grouped_data:
            logger.error("No training data provided")
            return []
        
        logger.info(f"Training data: {len(grouped_data)} meetings")
        logger.info(f"Validation data: {len(grouped_validation)} meetings")
        
        # Process training and validation data separately
        all_samples, speaker_to_idx = process_training_data(grouped_data, speaker_encoder, power_set_encoder)
        val_samples, val_speaker_to_idx = process_validation_data(grouped_validation, speaker_encoder, power_set_encoder)
        
        # Log speaker separation
        logger.info(f"\nSpeaker Mapping Separation:")
        logger.info(f"Training speakers: {len(speaker_to_idx)} - {list(speaker_to_idx.keys())}")
        logger.info(f"Validation speakers: {len(val_speaker_to_idx)} - {list(val_speaker_to_idx.keys())}")
        
        # Check for speaker overlap between train and validation
        train_speakers = set(speaker_to_idx.keys())
        val_speakers = set(val_speaker_to_idx.keys())
        overlapping_speakers = train_speakers.intersection(val_speakers)
        
        if overlapping_speakers:
            logger.warning(f"WARNING: {len(overlapping_speakers)} speakers appear in both train and validation sets: {overlapping_speakers}")
            logger.warning("This may cause data leakage. Consider using different meetings for train/validation split.")
        else:
            logger.info("✓ No speaker overlap between train and validation sets - good separation!")
        
        # Balance the training dataset
        logger.info("\nBalancing training dataset...")
        non_overlap_samples = [s for s in all_samples if not s["is_overlap"]]
        natural_overlap_samples = [s for s in all_samples if s["is_overlap"]]
        
        # Use all available samples (no artificial balancing)
        balanced_samples = non_overlap_samples + natural_overlap_samples
        
        # Shuffle samples
        np.random.shuffle(balanced_samples)
        
        logger.info("\nTraining Dataset Statistics:")
        logger.info(f"Non-overlapping samples: {len(non_overlap_samples)}")
        logger.info(f"Natural overlapping samples: {len(natural_overlap_samples)}")
        logger.info(f"Total samples: {len(balanced_samples)}")
        
        # Split samples among clients
        logger.info(f"\nSplitting {len(balanced_samples)} samples among {num_clients} clients...")
        
        if len(balanced_samples) == 0:
            logger.error("No samples available for splitting - cannot create clients")
            return []
        
        # Calculate samples per client and create client data
        samples_per_client = len(balanced_samples) // num_clients
        if samples_per_client == 0:
            logger.error(f"Not enough samples ({len(balanced_samples)}) for {num_clients} clients")
            return []
        
        client_samples = [balanced_samples[i:i + samples_per_client] for i in range(0, len(balanced_samples), samples_per_client)]
        logger.info(f"Created {len(client_samples)} client sample groups")
        
        # Create data loaders for each client
        logger.info("Creating data loaders for clients...")
        client_data = []
        for client_id, samples in enumerate(client_samples[:num_clients]):
            try:
                logger.info(f"\nProcessing client {client_id} with {len(samples)} samples")
                client_non_overlap = len([s for s in samples if not s["is_overlap"]])
                client_natural_overlap = len([s for s in samples if s["is_overlap"]])
                logger.info(f"Client {client_id} sample distribution:")
                logger.info(f"  - Non-overlapping: {client_non_overlap}")
                logger.info(f"  - Natural overlaps: {client_natural_overlap}")
                # Initialize lists for storing features and labels
                raw_features = []
                raw_labels = []
                speaker_ids = []
                meeting_ids = []
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
                    # frame-wise meeting_ids
                    meeting_id = np.full(feature.shape[0], sample.get("meeting_id", f"client_{client_id}"), dtype=object)
                    meeting_ids.append(meeting_id)
                if not raw_features:
                    logger.error(f"No valid features for client {client_id}, skipping client.")
                    continue
                max_len = max(f.shape[0] for f in raw_features)
                features_padded = []
                labels_padded = []
                meeting_ids_padded = []
                for feature, label, meeting_id in zip(raw_features, raw_labels, meeting_ids):
                    if feature.shape[0] < max_len:
                        pad_len = max_len - feature.shape[0]
                        feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                        label = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
                        # Pad meeting_ids with None for padded frames
                        meeting_id = np.pad(meeting_id, (0, pad_len), mode='constant', constant_values=None)
                    features_padded.append(feature)
                    labels_padded.append(label)
                    meeting_ids_padded.append(meeting_id)
                # Diagnostics: check that all sequence lengths are the same
                lengths = [f.shape[0] for f in features_padded]
                logger.info(f"All sequence lengths for client {client_id}: {set(lengths)}")
                if len(set(lengths)) != 1:
                    logger.error(f"Inhomogeneous sequence lengths for client {client_id}, skipping client.")
                    continue
                
                # Convert to numpy arrays
                features = np.array(features_padded)
                labels = np.array(labels_padded)
                meeting_ids_array = np.array(meeting_ids_padded)
                
                # Process validation data for this client
                logger.info(f"Processing validation data for client {client_id}...")
                val_raw_features = []
                val_raw_labels = []
                val_speaker_ids = []
                val_meeting_ids = []
                
                if val_samples:
                    for sample in val_samples:
                        feature = extract_features(sample["audio"]["array"])
                        if feature.shape[0] == 0:
                            logger.warning(f"Validation sample with empty feature sequence detected, skipping.")
                            continue
                        val_raw_features.append(feature)
                        # frame-wise labels
                        label = np.full(feature.shape[0], sample["speaker_id"], dtype=np.int64)
                        val_raw_labels.append(label)
                        val_speaker_ids.append(sample["speaker_id"])
                        # frame-wise meeting_ids
                        meeting_id = np.full(feature.shape[0], sample.get("meeting_id", f"val_client_{client_id}"), dtype=object)
                        val_meeting_ids.append(meeting_id)
                
                if not val_raw_features:
                    logger.warning(f"No valid validation features for client {client_id}, using empty validation set.")
                    # Use training data shape as reference for empty validation arrays
                    val_features = np.array([]).reshape(0, features.shape[1], features.shape[2])
                    val_labels = np.array([]).reshape(0, labels.shape[1])
                    val_meeting_ids_array = np.array([]).reshape(0, meeting_ids_array.shape[1])
                else:
                    val_max_len = max(f.shape[0] for f in val_raw_features)
                    val_features_padded = []
                    val_labels_padded = []
                    val_meeting_ids_padded = []
                    for feature, label, meeting_id in zip(val_raw_features, val_raw_labels, val_meeting_ids):
                        if feature.shape[0] < val_max_len:
                            pad_len = val_max_len - feature.shape[0]
                            feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                            label = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
                            meeting_id = np.pad(meeting_id, (0, pad_len), mode='constant', constant_values=None)
                        val_features_padded.append(feature)
                        val_labels_padded.append(label)
                        val_meeting_ids_padded.append(meeting_id)
                    
                    val_features = np.array(val_features_padded)
                    val_labels = np.array(val_labels_padded)
                    val_meeting_ids_array = np.array(val_meeting_ids_padded)
                
                # Use all training samples for this client (no train/val split from training data)
                train_features = features
                train_labels = labels
                train_meeting_ids = meeting_ids_array
                
                # Create separate speaker embeddings for training and validation
                train_speaker_to_embedding = compute_speaker_embeddings(grouped_data, speaker_encoder)
                val_speaker_to_embedding = compute_speaker_embeddings(grouped_validation, speaker_encoder) if grouped_validation else {}
                
                train_dataset = OverlappingSpeechDataset(
                    features=train_features,
                    labels=train_labels,
                    meeting_ids=train_meeting_ids,
                    speaker_ids=speaker_ids,
                    speaker_to_embedding=train_speaker_to_embedding,
                    max_speakers=len(speaker_to_idx)
                )
                val_dataset = OverlappingSpeechDataset(
                    features=val_features,
                    labels=val_labels,
                    meeting_ids=val_meeting_ids_array,
                    speaker_ids=val_speaker_ids,
                    speaker_to_embedding=val_speaker_to_embedding,
                    max_speakers=len(val_speaker_to_idx)
                )
                # Create data loaders with collate function
                def collate_fn(batch):
                    max_len = max(x[0].shape[0] for x in batch)
                    features = []
                    speaker_embeddings = []
                    labels = []
                    meeting_ids = []
                    for feature, all_embeddings, label, meeting_id in batch:
                        if feature.shape[0] < max_len:
                            pad_len = max_len - feature.shape[0]
                            feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                            # Pad meeting_id array with None for padded frames
                            meeting_id = np.pad(meeting_id, (0, pad_len), mode='constant', constant_values=None)
                        features.append(feature)
                        speaker_embeddings.append(all_embeddings)  # [num_speakers, 192]
                        labels.append(label)
                        meeting_ids.append(meeting_id)
                    features = torch.tensor(np.array(features), dtype=torch.float32)
                    speaker_embeddings = torch.stack(speaker_embeddings).float()  # [batch, num_speakers, 192]
                    labels = torch.tensor(np.array(labels), dtype=torch.long)
                    return features, speaker_embeddings, labels, meeting_ids
                
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
                client_data.append((train_loader, val_loader))
                logger.info(f"Created data loaders for client {client_id}")
            except KeyboardInterrupt:
                logger.info(f"\nInterrupted while processing client {client_id}. Saving progress...")
                break
            except Exception as e:
                logger.error(f"Error processing client {client_id}: {str(e)}")
                continue
        
        if not client_data:
            logger.error("No client data was created successfully")
            logger.error(f"Expected {num_clients} clients, but only {len(client_data)} were created")
            return []
        
        logger.info(f"\nData processing completed successfully - created {len(client_data)} clients")
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

def create_dataset_from_grouped(grouped_data, speaker_encoder, power_set_encoder, N=4):
    """Create dataset from grouped data with proper padding and fixed N slots per recording.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder instance for encoding speaker combinations
        N: Maximum number of speaker slots per recording (fixed for PSE)
        
    Returns:
        Tuple of (features, labels, meeting_ids): Dataset with padded features and meeting IDs
    """
    # Log function start
    print_function_start("create_dataset_from_grouped", 
                        grouped_data_len=len(grouped_data), 
                        N=N)
    features = []
    labels = []
    meeting_ids = []
    raw_features = []
    raw_labels = []
    raw_meeting_ids = []
    speaker_ids = []
    
    # First pass: extract features and find max length
    for meeting_id, samples in grouped_data.items():
        # Create speaker-to-slot mapping for this recording (max N slots)
        meeting_speakers = list(set(sample["speaker_id"] for sample in samples))
        speaker_to_slot = {}
        for i, speaker_id in enumerate(meeting_speakers[:N]):  # Limit to N slots
            speaker_to_slot[speaker_id] = i
        
        logger.info(f"Meeting {meeting_id}: {len(meeting_speakers)} speakers mapped to slots {list(speaker_to_slot.values())}")
        
        for sample in samples:
            # Extract features
            feature = extract_features(sample["audio"]["array"])
            raw_features.append(feature)
            
            # Map speaker to slot (0 to N-1) and encode using PowerSetEncoder
            speaker_id = sample["speaker_id"]
            if speaker_id in speaker_to_slot:
                slot_idx = speaker_to_slot[speaker_id]
                label = power_set_encoder.encode([slot_idx])  # Single speaker in slot
            else:
                # Speaker not in top-N, assign to slot 0 (or handle differently)
                logger.warning(f"Speaker {speaker_id} not in top-{N} for meeting {meeting_id}, assigning to slot 0")
                label = power_set_encoder.encode([0])
            
            # frame-wise labels
            raw_labels.append(np.full(feature.shape[0], label, dtype=np.int64))
            # frame-wise meeting_ids
            raw_meeting_ids.append(np.full(feature.shape[0], meeting_id, dtype=object))
            speaker_ids.append(speaker_id)
    
    # Find max sequence length
    max_len = max(f.shape[0] for f in raw_features)
    logger.info(f"Max sequence length: {max_len}")
    
    # Second pass: pad all features to max length
    for feature, label, meeting_id_array in zip(raw_features, raw_labels, raw_meeting_ids):
        if feature.shape[0] < max_len:
            pad_len = max_len - feature.shape[0]
            feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
            label = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
            # Pad meeting_ids with None for padded frames
            meeting_id_array = np.pad(meeting_id_array, (0, pad_len), mode='constant', constant_values=None)
        features.append(feature)
        labels.append(label)
        meeting_ids.append(meeting_id_array)
    
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)
    meeting_ids = np.array(meeting_ids)
    
    # Use statistics function for dataset logging
    print_dataset_statistics(features, labels, meeting_ids, raw_features)
    
    # Log function completion
    print_function_end("create_dataset_from_grouped", 
                      f"Created dataset with {features.shape[0]} samples, {features.shape[1]} max frames, {features.shape[2]} mel-bands")
    
    return features, labels, meeting_ids

def calculate_der(predictions, labels, power_set_encoder, speaker_id_list=None, debug=True, frame_shift=0.01, uri=None):
    """Calculate Diarization Error Rate with detailed logging and correct multi-speaker segments.
    
    Args:
        predictions: List of predicted power-set encoded values
        labels: List of ground truth power-set encoded values
        power_set_encoder: PowerSetEncoder instance
        speaker_id_list: Optional list of speaker IDs
        debug: Whether to print debug information
        frame_shift: Time shift between frames in seconds (default: 0.01)
        uri: Optional URI for the recording (for Annotation)
    """
    from pyannote.core import Segment, Annotation
    from pyannote.metrics.diarization import DiarizationErrorRate
    reference = Annotation(uri=uri)
    hypothesis = Annotation(uri=uri)
    mismatches = 0
    unique_label_values = set()
    unique_pred_values = set()
    active_speakers_labels = []
    active_speakers_preds = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if isinstance(label, str):
            if label == '-' or not label.isdigit():
                continue
            label = int(label)
        if label == -100:
            continue
        
        # Decode to get indices of active speakers
        true_indices = set(power_set_encoder.decode(label))  # e.g., {0, 2}
        pred_indices = set(power_set_encoder.decode(pred))   # e.g., {1, 3}
        
        if speaker_id_list is None:
            # Create stable speaker ID list based on max_speakers
            speaker_id_list = list(range(power_set_encoder.max_speakers))
        
        # Limit indices to valid speaker indices (0 to max_speakers-1)
        max_valid_idx = power_set_encoder.max_speakers - 1
        true_indices = {idx for idx in true_indices if 0 <= idx <= max_valid_idx}
        pred_indices = {idx for idx in pred_indices if 0 <= idx <= max_valid_idx}
        
        # Create time segment for this frame using frame_shift
        t0 = i * frame_shift
        t1 = (i + 1) * frame_shift
        
        # REFERENCE: Add separate track for each active speaker
        for track_idx, idx in enumerate(true_indices):
            if idx < len(speaker_id_list):
                speaker_name = f"speaker_{speaker_id_list[idx]}"
            else:
                speaker_name = f"speaker_slot_{idx}"
            reference[Segment(t0, t1), track_idx] = speaker_name
        
        # HYPOTHESIS: Same approach - separate track for each active speaker
        for track_idx, idx in enumerate(pred_indices):
            if idx < len(speaker_id_list):
                speaker_name = f"speaker_{speaker_id_list[idx]}"
            else:
                speaker_name = f"speaker_slot_{idx}"
            hypothesis[Segment(t0, t1), track_idx] = speaker_name
        
        unique_label_values.add(label)
        unique_pred_values.add(pred)
        active_speakers_labels.append(len(true_indices))  # Count of active speakers
        active_speakers_preds.append(len(pred_indices))   # Count of active speakers
        
        if debug and mismatches < 10 and true_indices != pred_indices:
            print(f"[DER DEBUG] Frame {i}: label={label}, pred={pred}, true_indices={true_indices}, pred_indices={pred_indices}")
            mismatches += 1
    if debug:
        print(f"[DER DEBUG] Power set encoder max_speakers: {power_set_encoder.max_speakers}")
        print(f"[DER DEBUG] Valid speaker indices: 0 to {power_set_encoder.max_speakers - 1}")
        print(f"[DER DEBUG] Unique label values: {unique_label_values}")
        print(f"[DER DEBUG] Unique pred values: {unique_pred_values}")
        print(f"[DER DEBUG] Active speakers per frame (labels): min={min(active_speakers_labels)}, max={max(active_speakers_labels)}, mean={np.mean(active_speakers_labels):.2f}")
        print(f"[DER DEBUG] Active speakers per frame (preds): min={min(active_speakers_preds)}, max={max(active_speakers_preds)}, mean={np.mean(active_speakers_preds):.2f}")
        print(f"[DER DEBUG] Reference segments (first 10): {list(reference.itertracks(yield_label=True))[:10]}")
        print(f"[DER DEBUG] Hypothesis segments (first 10): {list(hypothesis.itertracks(yield_label=True))[:10]}")
        print(f"[DER DEBUG] Frame shift: {frame_shift}s, Total duration: {len(predictions) * frame_shift:.2f}s")
    metric = DiarizationErrorRate()
    der = metric(reference, hypothesis)
    print(f"[DER DEBUG] DER calculation: valid frames used = {len(predictions)}, DER = {der}")
    return der

def prepare_data_loaders(grouped_train, grouped_validation, grouped_test, speaker_encoder, power_set_encoder, batch_size=4, speaker_to_embedding=None, N=4):
    # Import and log function start
    print_function_start("prepare_data_loaders", 
                        grouped_train_len=len(grouped_train), 
                        grouped_validation_len=len(grouped_validation), 
                        grouped_test_len=len(grouped_test),
                        batch_size=batch_size, 
                        N=N)
    """Prepare data loaders for training, validation and testing."""
    # Create datasets
    train_features, train_labels, train_meeting_ids = create_dataset_from_grouped(grouped_train, speaker_encoder, power_set_encoder, N)
    val_features, val_labels, val_meeting_ids = create_dataset_from_grouped(grouped_validation, speaker_encoder, power_set_encoder, N)
    test_features, test_labels, test_meeting_ids = create_dataset_from_grouped(grouped_test, speaker_encoder, power_set_encoder, N)
    
    # Create speaker ID to index mapping for speaker_ids list
    speaker_to_idx = {}
    for meeting_id, samples in grouped_train.items():
        for sample in samples:
            speaker_id = sample["speaker_id"]
            if speaker_id not in speaker_to_idx:
                speaker_to_idx[speaker_id] = len(speaker_to_idx)
    
    # Create speaker_ids lists (one per sample)
    train_speaker_ids = []
    val_speaker_ids = []
    test_speaker_ids = []
    
    for meeting_id, samples in grouped_train.items():
        for sample in samples:
            train_speaker_ids.append(sample["speaker_id"])
    
    for meeting_id, samples in grouped_validation.items():
        for sample in samples:
            val_speaker_ids.append(sample["speaker_id"])
    
    for meeting_id, samples in grouped_test.items():
        for sample in samples:
            test_speaker_ids.append(sample["speaker_id"])
    
    # Create datasets
    train_dataset = OverlappingSpeechDataset(
        features=train_features,
        labels=train_labels,
        meeting_ids=train_meeting_ids,
        speaker_ids=train_speaker_ids,
        speaker_to_embedding=compute_speaker_embeddings(grouped_train, speaker_encoder),
        max_speakers=N
    )
    val_dataset = OverlappingSpeechDataset(
        features=val_features,
        labels=val_labels,
        meeting_ids=val_meeting_ids,
        speaker_ids=val_speaker_ids,
        speaker_to_embedding=compute_speaker_embeddings(grouped_validation, speaker_encoder),
        max_speakers=N
    )
    test_dataset = OverlappingSpeechDataset(
        features=test_features,
        labels=test_labels,
        meeting_ids=test_meeting_ids,
        speaker_ids=test_speaker_ids,
        speaker_to_embedding=compute_speaker_embeddings(grouped_test, speaker_encoder),
        max_speakers=N
    )
    
    # Create data loaders with collate function
    def collate_fn(batch):
        max_len = max(x[0].shape[0] for x in batch)
        features = []
        speaker_embeddings = []
        labels = []
        meeting_ids = []
        for feature, all_embeddings, label, meeting_id in batch:
            if feature.shape[0] < max_len:
                pad_len = max_len - feature.shape[0]
                feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                # Pad meeting_id array with None for padded frames
                meeting_id = np.pad(meeting_id, (0, pad_len), mode='constant', constant_values=None)
            features.append(feature)
            speaker_embeddings.append(all_embeddings)  # [num_speakers, 192]
            labels.append(label)
            meeting_ids.append(meeting_id)
        features = torch.tensor(np.array(features), dtype=torch.float32)
        speaker_embeddings = torch.stack(speaker_embeddings).float()  # [batch, num_speakers, 192]
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        return features, speaker_embeddings, labels, meeting_ids
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Import and use statistics function for logging
    print_data_loaders_info(train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, batch_size)
    
    # Log function completion
    print_function_end("prepare_data_loaders", 
                      f"Created 3 data loaders: train({len(train_loader)} batches), val({len(val_loader)} batches), test({len(test_loader)} batches)")

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
    """Computes speaker embedding for each unique speaker_id based on their first audio segment."""
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
            emb = emb.squeeze().cpu()
        speaker_to_embedding[sid] = emb
    return speaker_to_embedding

class OverlappingSpeechDataset(Dataset):
    """Dataset for overlapping speech diarization."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, meeting_ids: np.ndarray, speaker_ids: list, speaker_to_embedding: dict, max_speakers: int = 4):
        self.features = features
        self.labels = labels
        self.meeting_ids = meeting_ids
        self.speaker_ids = speaker_ids
        self.speaker_to_embedding = speaker_to_embedding
        self.max_speakers = max_speakers
        
        # Create stable mapping: slot -> speaker_id
        # This ensures consistent ordering across all samples
        ordered_spk_ids = sorted(self.speaker_to_embedding.keys())
        self.speaker_id_list = ordered_spk_ids[:max_speakers]  # Limit to max_speakers
        
        # Create embeddings matrix in the same order as speaker_id_list
        self.all_embeddings = torch.stack([
            self.speaker_to_embedding[sid] for sid in self.speaker_id_list
        ]).float()
        
        # Assertions for consistency
        assert len(self.speaker_id_list) <= max_speakers, \
            f"speaker_id_list length ({len(self.speaker_id_list)}) > max_speakers ({max_speakers})"
        assert self.all_embeddings.shape[0] == len(self.speaker_id_list), \
            f"embeddings rows ({self.all_embeddings.shape[0]}) != speaker_id_list length ({len(self.speaker_id_list)})"
        
        # Log the mapping for debugging
        print(f"[OverlappingSpeechDataset] Slot->Speaker mapping: {dict(enumerate(self.speaker_id_list))}")
        print(f"[OverlappingSpeechDataset] Embeddings shape: {self.all_embeddings.shape}")
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple:
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        meeting_id = self.meeting_ids[idx]
        sid = self.speaker_ids[idx]
        
        # Return the pre-computed embeddings matrix
        return feature, self.all_embeddings, label, meeting_id
    
    def get_speaker_id_list(self):
        """Get the stable speaker ID list for DER calculation."""
        return self.speaker_id_list 
    


