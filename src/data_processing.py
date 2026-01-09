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
import gc
import os
import warnings
from functools import partial
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.core import Segment, Annotation, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from dataset_statistics import print_function_start, print_function_end, print_data_loaders_info, print_dataset_statistics

# Suppress multiprocessing resource tracker warnings about leaked semaphores
# These warnings are harmless and occur when multiprocessing workers are not explicitly closed
# The warnings don't affect functionality and are common with libraries like librosa, numba, etc.
# Use comprehensive filtering to catch all variations of the warning
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
warnings.filterwarnings('ignore', message='.*resource_tracker.*')
warnings.filterwarnings('ignore', message='.*leaked semaphore.*')
# Also set environment variable to suppress at OS level
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'

logger = logging.getLogger(__name__)

def extract_features(audio: np.ndarray, sr: int = 16000, n_mels: int = 80) -> np.ndarray:
    """Extract log-mel spectrogram features from audio.
    
    Memory fix: Enforce float32 to prevent float64 from librosa (saves 50% memory).
    """
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
    # MEMORY FIX: Convert to float32 immediately (librosa returns float64 by default)
    # This saves 50% memory compared to float64
    return log_mel.T.astype(np.float32)

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


def split_data_for_clients(grouped_data, grouped_validation, num_clients, speaker_encoder, power_set_encoder, batch_size=4):
    """Split grouped data among clients.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples (training data)
        grouped_validation: Dictionary of meeting_id to samples (validation data)
        num_clients: Number of clients to split data among
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder for encoding speaker combinations
        batch_size: Batch size for data loaders (default: 4)
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
                # MEMORY FIX: Store variable-length arrays (no client-level padding)
                # Padding will be done per-batch in collate_fn to save memory
                raw_features = []
                raw_labels = []
                speaker_ids = []
                meeting_ids = []  # MEMORY FIX: Store one meeting_id per sample (not per-frame)
                for sample in samples:
                    feature = extract_features(sample["audio"]["array"])
                    if feature.shape[0] == 0:
                        logger.warning(f"Sample with empty feature sequence detected, skipping.")
                        continue
                    raw_features.append(feature)
                    # frame-wise labels (variable length)
                    label = np.full(feature.shape[0], sample["speaker_id"], dtype=np.int64)
                    raw_labels.append(label)
                    speaker_ids.append(sample["speaker_id"])
                    # MEMORY FIX: Store one meeting_id per sample (string/int) instead of per-frame object array
                    meeting_id_str = sample.get("meeting_id", f"client_{client_id}")
                    meeting_ids.append(meeting_id_str)
                
                if not raw_features:
                    logger.error(f"No valid features for client {client_id}, skipping client.")
                    continue
                
                # MEMORY FIX: Keep variable-length arrays (no padding at client level)
                # Padding will be done per-batch in collate_fn_overlapping_speech
                features = raw_features  # List of variable-length arrays
                labels = raw_labels  # List of variable-length arrays
                meeting_ids_array = np.array(meeting_ids, dtype=object)  # One ID per sample
                
                # Log sequence length statistics
                lengths = [f.shape[0] for f in features]
                logger.info(f"Client {client_id} variable-length sequences: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
                
                # Process validation data for this client
                logger.info(f"Processing validation data for client {client_id}...")
                val_raw_features = []
                val_raw_labels = []
                val_speaker_ids = []
                val_meeting_ids = []  # MEMORY FIX: Store one meeting_id per sample
                
                if val_samples:
                    for sample in val_samples:
                        feature = extract_features(sample["audio"]["array"])
                        if feature.shape[0] == 0:
                            logger.warning(f"Validation sample with empty feature sequence detected, skipping.")
                            continue
                        val_raw_features.append(feature)
                        # frame-wise labels (variable length)
                        label = np.full(feature.shape[0], sample["speaker_id"], dtype=np.int64)
                        val_raw_labels.append(label)
                        val_speaker_ids.append(sample["speaker_id"])
                        # MEMORY FIX: Store one meeting_id per sample (string/int) instead of per-frame object array
                        meeting_id_str = sample.get("meeting_id", f"val_client_{client_id}")
                        val_meeting_ids.append(meeting_id_str)
                
                if not val_raw_features:
                    logger.warning(f"No valid validation features for client {client_id}, using empty validation set.")
                    # MEMORY FIX: Empty lists for variable-length data
                    val_features = []
                    val_labels = []
                    val_meeting_ids_array = np.array([], dtype=object)
                else:
                    # MEMORY FIX: Keep variable-length arrays (no padding at client level)
                    val_features = val_raw_features  # List of variable-length arrays
                    val_labels = val_raw_labels  # List of variable-length arrays
                    val_meeting_ids_array = np.array(val_meeting_ids, dtype=object)  # One ID per sample
                    
                    # Log validation sequence length statistics
                    val_lengths = [f.shape[0] for f in val_features]
                    logger.info(f"Client {client_id} validation variable-length sequences: min={min(val_lengths)}, max={max(val_lengths)}, mean={np.mean(val_lengths):.1f}")
                
                # Use all training samples for this client (no train/val split from training data)
                train_features = features
                train_labels = labels
                train_meeting_ids = meeting_ids_array
                
                # Create meeting-specific embeddings and slot mappings
                # For FL, we need to create meeting_to_slot_speakers from grouped_data
                train_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_data, speaker_encoder)
                val_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_validation, speaker_encoder) if grouped_validation else {}
                
                # Create meeting_to_slot_speakers for train and val
                # Use same deterministic ordering as _process_meeting_with_overlaps
                train_meeting_to_slot_speakers = {}
                for meeting_id, samples in grouped_data.items():
                    speaker_durations = {}
                    speaker_first_appearance = {}
                    for idx, sample in enumerate(samples):
                        sid = sample["speaker_id"]
                        if isinstance(sid, list):
                            continue
                        duration = sample.get("end_time", 0) - sample.get("begin_time", 0)
                        if sid not in speaker_durations:
                            speaker_durations[sid] = 0.0
                            speaker_first_appearance[sid] = idx
                        speaker_durations[sid] += duration
                    meeting_speakers = sorted(
                        speaker_durations.keys(),
                        key=lambda sid: (-speaker_durations[sid], speaker_first_appearance[sid])
                    )
                    train_meeting_to_slot_speakers[meeting_id] = meeting_speakers[:len(speaker_to_idx)]
                
                val_meeting_to_slot_speakers = {}
                if grouped_validation:
                    for meeting_id, samples in grouped_validation.items():
                        speaker_durations = {}
                        speaker_first_appearance = {}
                        for idx, sample in enumerate(samples):
                            sid = sample["speaker_id"]
                            if isinstance(sid, list):
                                continue
                            duration = sample.get("end_time", 0) - sample.get("begin_time", 0)
                            if sid not in speaker_durations:
                                speaker_durations[sid] = 0.0
                                speaker_first_appearance[sid] = idx
                            speaker_durations[sid] += duration
                        meeting_speakers = sorted(
                            speaker_durations.keys(),
                            key=lambda sid: (-speaker_durations[sid], speaker_first_appearance[sid])
                        )
                        val_meeting_to_slot_speakers[meeting_id] = meeting_speakers[:len(val_speaker_to_idx)]
                
                train_dataset = OverlappingSpeechDataset(
                    features=train_features,
                    labels=train_labels,
                    meeting_ids=train_meeting_ids,
                    speaker_ids=speaker_ids,
                    meeting_to_speaker_embedding=train_meeting_to_speaker_embedding,
                    meeting_to_slot_speakers=train_meeting_to_slot_speakers,
                    max_speakers=len(speaker_to_idx)
                )
                val_dataset = OverlappingSpeechDataset(
                    features=val_features,
                    labels=val_labels,
                    meeting_ids=val_meeting_ids_array,
                    speaker_ids=val_speaker_ids,
                    meeting_to_speaker_embedding=val_meeting_to_speaker_embedding,
                    meeting_to_slot_speakers=val_meeting_to_slot_speakers,
                    max_speakers=len(val_speaker_to_idx)
                )
                # For FL clients: use num_workers=0, pin_memory=False, and persistent_workers=False to reduce RAM usage
                # This is critical for avoiding OOM in federated learning with Ray
                # D: In Ray workers, multiprocessing and pin_memory can cause memory issues
                num_workers = 0  # Disable multiprocessing to save RAM
                pin_memory = False  # D: Disable pin_memory for Ray workers (causes memory issues)
                use_persistent_workers = False  # Disable persistent workers to save RAM
                
                logger.info(f"Creating FL client DataLoaders with num_workers=0, pin_memory=False, persistent_workers=False (memory-optimized for Ray)")
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=batch_size, 
                    shuffle=True, 
                    collate_fn=collate_fn_overlapping_speech,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=use_persistent_workers
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size, 
                    shuffle=False, 
                    collate_fn=collate_fn_overlapping_speech,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=use_persistent_workers
                )
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

def _create_overlapping_segment(current, next_seg, sr=16000):
    """Create an overlapping audio segment from two overlapping segments.
    
    Args:
        current: First segment dict with 'audio', 'begin_time', 'end_time', 'speaker_id'
        next_seg: Second segment dict with 'audio', 'begin_time', 'end_time', 'speaker_id'
        sr: Sample rate (default: 16000)
        
    Returns:
        dict: Overlapping sample with combined audio and list of speaker_ids, or None if no valid overlap
    """
    # Check if segments overlap
    if not (next_seg["begin_time"] < current["end_time"] and next_seg["end_time"] > current["begin_time"]):
        return None
    
    # Calculate overlap boundaries
    overlap_begin = max(current["begin_time"], next_seg["begin_time"])
    overlap_end = min(current["end_time"], next_seg["end_time"])
    
    if overlap_end <= overlap_begin:
        return None
    
    # Get audio segments
    current_audio = current["audio"]["array"]
    next_audio = next_seg["audio"]["array"]
    
    # Calculate overlap duration in samples
    current_start = int((overlap_begin - current["begin_time"]) * sr)
    current_end = int((overlap_end - current["begin_time"]) * sr)
    next_start = int((overlap_begin - next_seg["begin_time"]) * sr)
    next_end = int((overlap_end - next_seg["begin_time"]) * sr)
    
    # Ensure indices are within bounds
    current_start = max(0, min(current_start, len(current_audio)))
    current_end = max(0, min(current_end, len(current_audio)))
    next_start = max(0, min(next_start, len(next_audio)))
    next_end = max(0, min(next_end, len(next_audio)))
    
    # Extract overlapping portions
    current_overlap = current_audio[current_start:current_end]
    next_overlap = next_audio[next_start:next_end]
    
    # Ensure both segments have the same length
    min_length = min(len(current_overlap), len(next_overlap))
    if min_length <= 0:
        return None
    
    current_overlap = current_overlap[:min_length]
    next_overlap = next_overlap[:min_length]
    
    # Combine audio segments
    combined_audio = current_overlap + next_overlap
    
    # Normalize
    if np.max(np.abs(combined_audio)) > 0:
        combined_audio = combined_audio / np.max(np.abs(combined_audio))
    
    # Skip if both segments have the same speaker (not a true overlap)
    if current["speaker_id"] == next_seg["speaker_id"]:
        return None
    
    # Create overlapping sample
    return {
        "audio": {"array": combined_audio},
        "speaker_id": [current["speaker_id"], next_seg["speaker_id"]],
        "begin_time": overlap_begin,
        "end_time": overlap_end
    }


def _process_meeting_with_overlaps(meeting_id, samples, N=4):
    """Process a single meeting to detect and create overlapping segments.
    
    Args:
        meeting_id: ID of the meeting
        samples: List of sample dicts for this meeting
        N: Maximum number of speaker slots
        
    Returns:
        tuple: (all_samples_info, meeting_original_count, meeting_overlaps_count, slot_speakers)
            where all_samples_info is a list of dicts with 'meeting_id', 'sample', 'speaker_to_slot', 'is_overlap'
            and slot_speakers is a list of speaker_id in slot order [spk_id_slot0, spk_id_slot1, ...]
    """
    # Create deterministic speaker-to-slot mapping based on total speech duration
    # This ensures consistent ordering across all samples from the same meeting
    speaker_durations = {}
    speaker_first_appearance = {}
    
    for idx, sample in enumerate(samples):
        sid = sample["speaker_id"]
        # Skip if speaker_id is a list (overlap segment)
        if isinstance(sid, list):
            continue
        
        duration = sample.get("end_time", 0) - sample.get("begin_time", 0)
        if sid not in speaker_durations:
            speaker_durations[sid] = 0.0
            speaker_first_appearance[sid] = idx
        speaker_durations[sid] += duration
    
    # Sort speakers by: 1) total duration (descending), 2) first appearance (ascending)
    # This creates a stable, deterministic ordering
    meeting_speakers = sorted(
        speaker_durations.keys(),
        key=lambda sid: (-speaker_durations[sid], speaker_first_appearance[sid])
    )
    
    # Take top N speakers and create slot mapping
    slot_speakers = meeting_speakers[:N]  # List of speaker_id in slot order
    speaker_to_slot = {sid: i for i, sid in enumerate(slot_speakers)}
    
    logger.info(f"Meeting {meeting_id}: {len(meeting_speakers)} speakers, selected top {len(slot_speakers)} for slots")
    logger.info(f"  Slot order: {slot_speakers}")
    logger.info(f"  Slot mapping: {speaker_to_slot}")
    
    # Sort samples by begin_time to detect overlaps
    sorted_samples = sorted(samples, key=lambda x: x["begin_time"])
    meeting_original = 0
    meeting_overlaps = 0
    all_samples_info = []
    
    # Add all original non-overlapping segments
    logger.info(f"  [OVERLAP PROCESSING] Meeting {meeting_id}: Adding {len(sorted_samples)} original segments...")
    for sample in sorted_samples:
        all_samples_info.append({
            'meeting_id': meeting_id,
            'sample': sample,
            'speaker_to_slot': speaker_to_slot,
            'is_overlap': False
        })
        meeting_original += 1
    
    # Find and create overlapping segments
    logger.info(f"  [OVERLAP PROCESSING] Meeting {meeting_id}: Detecting overlapping segments...")
    overlap_examples = []  # For logging examples
    same_speaker_overlaps = 0  # Count cases where same speaker overlaps with themselves
    for i in range(len(sorted_samples)):
        current = sorted_samples[i]
        for j in range(i + 1, len(sorted_samples)):
            next_seg = sorted_samples[j]
            
            # Check if segments overlap in time
            if next_seg["begin_time"] < current["end_time"] and next_seg["end_time"] > current["begin_time"]:
                # Check if same speaker (will be skipped in _create_overlapping_segment)
                if current["speaker_id"] == next_seg["speaker_id"]:
                    same_speaker_overlaps += 1
            
            overlap_sample = _create_overlapping_segment(current, next_seg)
            if overlap_sample is not None:
                all_samples_info.append({
                    'meeting_id': meeting_id,
                    'sample': overlap_sample,
                    'speaker_to_slot': speaker_to_slot,
                    'is_overlap': True
                })
                meeting_overlaps += 1
                
                # Save examples for logging (first 3)
                if len(overlap_examples) < 3:
                    overlap_examples.append({
                        'speakers': overlap_sample['speaker_id'],
                        'time': (overlap_sample['begin_time'], overlap_sample['end_time']),
                        'duration': overlap_sample['end_time'] - overlap_sample['begin_time']
                    })
    
    # Log examples of overlapping segments with labels
    if overlap_examples:
        logger.info(f"  [OVERLAP PROCESSING] Examples of overlapping segments for meeting {meeting_id}:")
        for idx, example in enumerate(overlap_examples, 1):
            speakers = example['speakers']
            time_range = example['time']
            duration = example['duration']
            # Determine slots for these speakers
            slot_indices = []
            for spk_id in speakers:
                if spk_id in speaker_to_slot:
                    slot_indices.append(speaker_to_slot[spk_id])
            logger.info(f"    Example {idx}: Speakers {speakers} -> Slots {slot_indices}, "
                       f"Time: {time_range[0]:.2f}s-{time_range[1]:.2f}s ({duration:.2f}s)")
    
    logger.info(f"  [OVERLAP PROCESSING] Meeting {meeting_id} statistics:")
    logger.info(f"    - Original segments: {meeting_original}")
    logger.info(f"    - Overlapping segments created: {meeting_overlaps}")
    if same_speaker_overlaps > 0:
        logger.info(f"    - Same-speaker overlaps skipped: {same_speaker_overlaps} (same speaker speaking in overlapping time segments - not true overlap)")
    if meeting_original > 0:
        logger.info(f"    - Overlap ratio: {meeting_overlaps/meeting_original:.2%}")
    
    if meeting_overlaps > 0:
        logger.info(f"    ✓ CONFIRMED: Found and processed {meeting_overlaps} overlapping segments for meeting {meeting_id}")
    
    return all_samples_info, meeting_original, meeting_overlaps, same_speaker_overlaps, slot_speakers


def _process_all_meetings_with_overlaps(grouped_data, N=4):
    """Process all meetings to detect and create overlapping segments.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        N: Maximum number of speaker slots
        
    Returns:
        tuple: (all_samples_info, total_original_segments, total_overlapping_segments, total_same_speaker_overlaps, meeting_to_slot_speakers)
            where meeting_to_slot_speakers is dict[meeting_id, list[speaker_id]] in slot order
    """
    logger.info("=" * 80)
    logger.info("OVERLAPPING SPEECH PROCESSING: ENABLED")
    logger.info("=" * 80)
    logger.info("Processing overlapping speech segments for each meeting...")
    
    total_meetings = len(grouped_data)
    total_original_segments = 0
    total_overlapping_segments = 0
    total_same_speaker_overlaps = 0
    all_samples_info = []
    meeting_to_slot_speakers = {}
    
    for meeting_id, samples in grouped_data.items():
        try:
            meeting_samples_info, meeting_original, meeting_overlaps, same_speaker_overlaps, slot_speakers = _process_meeting_with_overlaps(
                meeting_id, samples, N
            )
            all_samples_info.extend(meeting_samples_info)
            total_original_segments += meeting_original
            total_overlapping_segments += meeting_overlaps
            total_same_speaker_overlaps += same_speaker_overlaps
            meeting_to_slot_speakers[meeting_id] = slot_speakers
        except Exception as e:
            logger.error(f"Error processing meeting {meeting_id} for overlaps: {str(e)}")
            continue
    
    # Log overall statistics
    logger.info("=" * 80)
    logger.info("OVERLAPPING SPEECH PROCESSING: COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total meetings processed: {total_meetings}")
    logger.info(f"Total original segments: {total_original_segments}")
    logger.info(f"Total overlapping segments created: {total_overlapping_segments}")
    if total_same_speaker_overlaps > 0:
        logger.info(f"Total same-speaker overlaps skipped: {total_same_speaker_overlaps} (same speaker in overlapping time segments - not true overlap)")
    logger.info(f"Total segments (original + overlaps): {len(all_samples_info)}")
    if total_original_segments > 0:
        logger.info(f"Overall overlap ratio: {total_overlapping_segments/total_original_segments:.2%}")
    
    # Confirmation message
    if total_overlapping_segments > 0:
        logger.info("")
        logger.info("✓✓✓ CONFIRMATION: OVERLAPPING SPEECH SUCCESSFULLY PROCESSED ✓✓✓")
        logger.info(f"   - {total_overlapping_segments} overlapping segments were detected and added to the dataset")
        logger.info(f"   - These segments will be used for training with multi-speaker labels")
        logger.info("")
    else:
        logger.warning("")
        logger.warning("⚠ WARNING: No overlapping segments were found in the dataset")
        logger.warning("   - This may indicate that the dataset has no natural overlaps")
        logger.warning("   - Or there may be an issue with overlap detection")
        logger.warning("")
    logger.info("=" * 80)
    
    return all_samples_info, total_original_segments, total_overlapping_segments, total_same_speaker_overlaps, meeting_to_slot_speakers


def _find_max_sequence_length(all_samples_info, chunk_size=500):
    """First pass: extract features to find maximum sequence length and feature dimension.
    
    Args:
        all_samples_info: List of sample info dicts
        chunk_size: Number of samples to process at once
        
    Returns:
        tuple: (max_len, feature_dim)
    """
    logger.info("First pass: Extracting features in chunks to determine dimensions (checking ALL samples)...")
    max_len = 0
    feature_dim = None
    total_samples = len(all_samples_info)
    
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_info = all_samples_info[chunk_start:chunk_end]
        
        for item in chunk_info:
            feature = extract_features(item['sample']["audio"]["array"])
            if feature_dim is None:
                feature_dim = feature.shape[1]
            max_len = max(max_len, feature.shape[0])
            del feature
        
        # Progress update
        if chunk_start % (chunk_size * 10) == 0 or chunk_start == total_samples - chunk_size:
            logger.info(f"Dimension discovery progress: {chunk_start + len(chunk_info)}/{total_samples} samples checked, current max_len: {max_len}")
        
        # Force garbage collection periodically
        if chunk_start % (chunk_size * 20) == 0:
            gc.collect()
    
    gc.collect()
    return max_len, feature_dim


def _allocate_arrays(total_samples, max_len, feature_dim):
    """Pre-allocate numpy arrays and calculate memory requirements.
    
    Args:
        total_samples: Total number of samples
        max_len: Maximum sequence length
        feature_dim: Feature dimension
        
    Returns:
        tuple: (features, labels, meeting_ids, total_memory_gb)
    """
    # Calculate estimated memory requirements
    features_memory_gb = (total_samples * max_len * feature_dim * 4) / (1024**3)  # float32 = 4 bytes
    labels_memory_gb = (total_samples * max_len * 8) / (1024**3)  # int64 = 8 bytes
    # MEMORY FIX: meeting_ids is now 1D array (one per sample), not 2D (samples x max_len)
    meeting_ids_memory_gb = (total_samples * 8) / (1024**3)  # object pointer ~8 bytes per sample
    total_memory_gb = features_memory_gb + labels_memory_gb + meeting_ids_memory_gb
    
    logger.info(f"Estimated memory requirements:")
    logger.info(f"  Features array: {features_memory_gb:.2f} GB")
    logger.info(f"  Labels array: {labels_memory_gb:.2f} GB")
    logger.info(f"  Meeting IDs array: {meeting_ids_memory_gb:.2f} GB (1D: one per sample)")
    logger.info(f"  Total: {total_memory_gb:.2f} GB")
    
    # Warn if memory requirements are very high
    if total_memory_gb > 100:
        logger.warning(f"WARNING: Very high memory requirement ({total_memory_gb:.2f} GB)!")
        logger.warning(f"Consider reducing chunk_size or using a smaller dataset subset.")
        logger.warning(f"Proceeding anyway, but may fail if insufficient memory is available.")
    
    # Pre-allocate numpy arrays
    logger.info(f"Pre-allocating arrays for {total_samples} samples...")
    try:
        features = np.zeros((total_samples, max_len, feature_dim), dtype=np.float32)
        labels = np.full((total_samples, max_len), -100, dtype=np.int64)
        # MEMORY FIX: meeting_ids is now 1D array (one per sample), not 2D (samples x max_len)
        meeting_ids = np.empty(total_samples, dtype=object)
    except MemoryError as e:
        logger.error(f"Memory allocation failed! Required: {total_memory_gb:.2f} GB")
        logger.error(f"Try: 1) Reducing chunk_size, 2) Using smaller dataset, 3) Requesting more memory")
        raise
    
    return features, labels, meeting_ids, total_memory_gb


def _encode_speaker_label(speaker_id, speaker_to_slot, is_overlap, power_set_encoder, meeting_id, N):
    """Encode speaker label(s) using Power Set Encoding.
    
    Args:
        speaker_id: Single speaker ID (str/int) or list of speaker IDs for overlaps
        speaker_to_slot: Mapping from speaker_id to slot index
        is_overlap: Whether this is an overlapping segment
        power_set_encoder: PowerSetEncoder instance
        meeting_id: Meeting ID for logging
        N: Maximum number of speakers
        
    Returns:
        int: Encoded label value
    """
    if is_overlap:
        # This is an overlapping segment - speaker_id is a list
        if isinstance(speaker_id, list):
            slot_indices = []
            for spk_id in speaker_id:
                if spk_id in speaker_to_slot:
                    slot_indices.append(speaker_to_slot[spk_id])
                else:
                    logger.warning(f"Speaker {spk_id} in overlap not in top-{N} for meeting {meeting_id}, skipping")
            
            # Remove duplicates (shouldn't happen after fix in _create_overlapping_segment, but safety check)
            slot_indices = list(set(slot_indices))
            
            if len(slot_indices) > 0:
                # If only one unique speaker after deduplication, treat as non-overlap
                if len(slot_indices) == 1:
                    logger.warning(f"Overlap segment for meeting {meeting_id} has only one unique speaker after deduplication, treating as single speaker")
                    return power_set_encoder.encode(slot_indices)
                return power_set_encoder.encode(slot_indices)
            else:
                logger.warning(f"No valid speakers in overlap for meeting {meeting_id}, using slot 0")
                return power_set_encoder.encode([0])
        else:
            logger.warning(f"Overlap segment has non-list speaker_id: {speaker_id}")
            if speaker_id in speaker_to_slot:
                return power_set_encoder.encode([speaker_to_slot[speaker_id]])
            else:
                return power_set_encoder.encode([0])
    else:
        # This is a non-overlapping segment - speaker_id is a single value
        if isinstance(speaker_id, list):
            logger.warning(f"Non-overlap segment has list speaker_id: {speaker_id}, using first")
            speaker_id = speaker_id[0]
        
        if speaker_id in speaker_to_slot:
            return power_set_encoder.encode([speaker_to_slot[speaker_id]])
        else:
            logger.warning(f"Speaker {speaker_id} not in top-{N} for meeting {meeting_id}, assigning to slot 0")
            return power_set_encoder.encode([0])


def _extract_features_and_labels(all_samples_info, features, labels, meeting_ids, max_len, 
                                  power_set_encoder, N, chunk_size=500, max_sequence_length=None):
    """Second pass: extract features, create labels, and fill pre-allocated arrays.
    
    Args:
        all_samples_info: List of sample info dicts
        features: Pre-allocated features array
        labels: Pre-allocated labels array
        meeting_ids: Pre-allocated meeting_ids array
        max_len: Maximum sequence length
        power_set_encoder: PowerSetEncoder instance
        N: Maximum number of speakers
        chunk_size: Number of samples to process at once
        max_sequence_length: Optional maximum sequence length limit
        
    Returns:
        tuple: (speaker_ids, overlap_count, non_overlap_count)
    """
    logger.info(f"Second pass: Extracting and padding features (chunk_size={chunk_size})...")
    logger.info("Processing both original and overlapping segments...")
    
    total_samples = len(all_samples_info)
    speaker_ids = []
    overlap_count = 0
    non_overlap_count = 0
    
    # Track truncation statistics
    truncation_count = 0
    total_original_length = 0
    total_truncated_length = 0
    
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_info = all_samples_info[chunk_start:chunk_end]
        
        for local_idx, item in enumerate(chunk_info):
            global_idx = chunk_start + local_idx
            
            # Extract features
            feature = extract_features(item['sample']["audio"]["array"])
            speaker_to_slot = item['speaker_to_slot']
            meeting_id = item['meeting_id']
            sample = item['sample']
            is_overlap = item.get('is_overlap', False)
            
            original_length = feature.shape[0]
            total_original_length += original_length
            
            # Truncate if longer than max_len
            if feature.shape[0] > max_len:
                truncation_count += 1
                truncated_length = feature.shape[0] - max_len
                total_truncated_length += truncated_length
                if max_sequence_length is None:
                    logger.error(f"CRITICAL: Found longer sequence ({feature.shape[0]} > {max_len})!")
                    logger.error(f"This should not happen - we checked all samples. Truncating to {max_len}.")
                else:
                    if truncation_count <= 5:  # Log first 5 truncations
                        logger.warning(f"Truncating sequence from {feature.shape[0]} to {max_len} frames (meeting_id={meeting_id})")
                feature = feature[:max_len, :]
            else:
                total_truncated_length += 0
            
            # Encode speaker label
            speaker_id = sample["speaker_id"]
            label = _encode_speaker_label(speaker_id, speaker_to_slot, is_overlap, 
                                         power_set_encoder, meeting_id, N)
            
            if is_overlap:
                overlap_count += 1
                # Log first few examples of overlapping labels
                if overlap_count <= 5:
                    if isinstance(speaker_id, list):
                        slot_indices = [speaker_to_slot.get(spk_id, -1) for spk_id in speaker_id if spk_id in speaker_to_slot]
                        decoded = power_set_encoder.decode(label)
                        logger.info(f"  [LABEL EXAMPLE] Overlap segment {overlap_count}: "
                                   f"Speakers {speaker_id} -> Slots {slot_indices} -> "
                                   f"Encoded label: {label} -> Decoded slots: {decoded}")
            else:
                non_overlap_count += 1
            
            # Store speaker_id (convert list to tuple for hashing if needed)
            if isinstance(speaker_id, list):
                speaker_ids.append(tuple(speaker_id))
            else:
                speaker_ids.append(speaker_id)
            
            # Store feature and labels
            seq_len = feature.shape[0]
            features[global_idx, :seq_len, :] = feature
            labels[global_idx, :seq_len] = label
            # MEMORY FIX: Store one meeting_id per sample (not per-frame)
            meeting_ids[global_idx] = meeting_id
            
            # Free feature memory immediately
            del feature
            
            if (global_idx + 1) % 100 == 0 or global_idx == total_samples - 1:
                logger.info(f"Padding progress: {global_idx+1}/{total_samples} ({100*(global_idx+1)/total_samples:.1f}%)")
        
        # Force garbage collection after each chunk
        gc.collect()
    
    # Log final statistics
    logger.info("=" * 80)
    logger.info("FEATURE EXTRACTION: COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Non-overlapping segments processed: {non_overlap_count}")
    logger.info(f"Overlapping segments processed: {overlap_count}")
    logger.info(f"Total segments processed: {non_overlap_count + overlap_count}")
    if (non_overlap_count + overlap_count) > 0:
        logger.info(f"Overlap percentage in final dataset: {100*overlap_count/(non_overlap_count + overlap_count):.2f}%")
    
    # Log truncation statistics
    if max_sequence_length is not None:
        truncation_ratio = total_truncated_length / total_original_length if total_original_length > 0 else 0.0
        logger.info(f"Truncation statistics (max_sequence_length={max_sequence_length}):")
        logger.info(f"  - Samples truncated: {truncation_count}/{total_samples} ({100*truncation_count/total_samples:.2f}%)")
        logger.info(f"  - Total original length: {total_original_length:,} frames")
        logger.info(f"  - Total truncated length: {total_truncated_length:,} frames")
        logger.info(f"  - Truncation ratio: {truncation_ratio:.2%}")
        if truncation_ratio > 0.1:
            logger.warning(f"  ⚠️  High truncation ratio ({truncation_ratio:.2%})! Consider increasing max_sequence_length.")
    
    # Final confirmation with label statistics
    if overlap_count > 0:
        logger.info("")
        logger.info("✓✓✓ FINAL CONFIRMATION: OVERLAPPING SPEECH IN FINAL DATASET ✓✓✓")
        logger.info(f"   - {overlap_count} overlapping segments with multi-speaker labels are in the dataset")
        logger.info(f"   - These segments use Power Set Encoding for multiple active speakers")
        logger.info(f"   - The model will be trained on both single-speaker and multi-speaker segments")
        
        # Show statistics for overlapping segment labels
        overlap_labels = []
        for i in range(len(all_samples_info)):
            item = all_samples_info[i]
            if item.get('is_overlap', False):
                sample = item['sample']
                speaker_id = sample["speaker_id"]
                speaker_to_slot = item['speaker_to_slot']
                if isinstance(speaker_id, list):
                    slot_indices = [speaker_to_slot.get(spk_id) for spk_id in speaker_id if spk_id in speaker_to_slot]
                    if slot_indices:
                        label = power_set_encoder.encode(slot_indices)
                        overlap_labels.append((label, slot_indices, speaker_id))
        
        if overlap_labels:
            unique_labels = {}
            for label, slots, speakers in overlap_labels:
                if label not in unique_labels:
                    unique_labels[label] = {'count': 0, 'slots': slots, 'speakers': speakers}
                unique_labels[label]['count'] += 1
            
            logger.info(f"\n   Overlap label statistics:")
            logger.info(f"   - Unique overlap labels: {len(unique_labels)}")
            logger.info(f"   - Label distribution (first 10):")
            for label, info in sorted(unique_labels.items())[:10]:  # Show first 10
                decoded = power_set_encoder.decode(label)
                logger.info(f"     * Label {label}: Slots {info['slots']} (Speakers {info['speakers']}) "
                           f"-> Decoded: {decoded}, Count: {info['count']}")
            if len(unique_labels) > 10:
                logger.info(f"     ... and {len(unique_labels) - 10} more unique labels")
        
        logger.info("")
    logger.info("=" * 80)
    
    return speaker_ids, overlap_count, non_overlap_count


def create_dataset_from_grouped(grouped_data, speaker_encoder, power_set_encoder, N=4, chunk_size=500, max_sequence_length=None):
    """Create dataset from grouped data with proper padding and fixed N slots per recording.
    
    Memory-optimized version that processes data in chunks to reduce peak memory usage.
    NOW INCLUDES OVERLAPPING SPEECH PROCESSING: detects and processes natural overlapping segments.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        speaker_encoder: Speaker encoder model (not used directly, kept for API compatibility)
        power_set_encoder: PowerSetEncoder instance for encoding speaker combinations
        N: Maximum number of speaker slots per recording (fixed for PSE)
        chunk_size: Number of samples to process at once (default: 500)
        max_sequence_length: Optional maximum sequence length to truncate longer sequences (default: None = no limit)
        
    Returns:
        Tuple of (features, labels, meeting_ids, speaker_ids, meeting_to_slot_speakers): 
            Dataset with padded features, meeting IDs, speaker IDs, and slot speaker mapping
    """
    # Log function start
    print_function_start("create_dataset_from_grouped", 
                        grouped_data_len=len(grouped_data), 
                        N=N)
    
    # Step 1: Process all meetings to detect and create overlapping segments
    all_samples_info, total_original_segments, total_overlapping_segments, total_same_speaker_overlaps, meeting_to_slot_speakers = _process_all_meetings_with_overlaps(
        grouped_data, N
    )
    
    total_samples = len(all_samples_info)
    logger.info(f"Total samples to process (including overlaps): {total_samples}")
    
    # Step 2: First pass - find maximum sequence length and feature dimension
    max_len, feature_dim = _find_max_sequence_length(all_samples_info, chunk_size)
    
    # Apply max_sequence_length limit if specified
    original_max_len = max_len
    if max_sequence_length is not None and max_len > max_sequence_length:
        logger.warning(f"Truncating max sequence length from {max_len} to {max_sequence_length} to save memory")
        max_len = max_sequence_length
    
    logger.info(f"Final max sequence length: {max_len} (original: {original_max_len}), Feature dimension: {feature_dim}")
    
    # Step 3: Pre-allocate numpy arrays
    features, labels, meeting_ids, total_memory_gb = _allocate_arrays(total_samples, max_len, feature_dim)
    
    # Step 4: Second pass - extract features, create labels, and fill arrays
    speaker_ids, overlap_count, non_overlap_count = _extract_features_and_labels(
        all_samples_info, features, labels, meeting_ids, max_len,
        power_set_encoder, N, chunk_size, max_sequence_length
    )
    
    # Verify that speaker_ids length matches features length
    if len(speaker_ids) != len(features):
        logger.error(f"Mismatch: speaker_ids length ({len(speaker_ids)}) != features length ({len(features)})")
        raise ValueError(f"speaker_ids length ({len(speaker_ids)}) must match features length ({len(features)})")
    
    # Final logging
    logger.info("Padding complete.")
    logger.info(f"Features array created: shape {features.shape}, size {features.nbytes / (1024**3):.2f} GB")
    logger.info(f"Labels array created: shape {labels.shape}")
    logger.info(f"Meeting IDs array created: shape {meeting_ids.shape}")
    logger.info(f"Speaker IDs list length: {len(speaker_ids)}")
    
    # Use statistics function for dataset logging
    print_dataset_statistics(features, labels, meeting_ids, [])
    
    # Log function completion
    print_function_end("create_dataset_from_grouped", 
                      f"Created dataset with {features.shape[0]} samples, {features.shape[1]} max frames, {features.shape[2]} mel-bands")
    
    return features, labels, meeting_ids, speaker_ids, meeting_to_slot_speakers

def frames_to_annotation(active_speakers_per_frame, frame_shift, speaker_id_list, uri=None):
    """Convert frame-wise active speaker sets to pyannote Annotation by merging consecutive frames.
    
    This function efficiently merges consecutive frames with the same set of active speakers
    into single segments, reducing the number of segments from hundreds of thousands to thousands.
    
    Args:
        active_speakers_per_frame: List of sets of active speaker indices per frame
        frame_shift: Time shift between frames in seconds
        speaker_id_list: List of speaker IDs for naming
        uri: Optional URI for the Annotation
    
    Returns:
        pyannote.core.Annotation with merged segments
    """
    annotation = Annotation(uri=uri)
    
    if not active_speakers_per_frame:
        return annotation
    
    # Track current segments for each speaker
    # speaker_segments: dict[speaker_idx] -> (start_time, end_time)
    speaker_segments = {}
    
    for frame_idx, active_speakers in enumerate(active_speakers_per_frame):
        current_time = frame_idx * frame_shift
        next_time = (frame_idx + 1) * frame_shift
        
        # Get currently active speakers
        current_active = set(active_speakers)
        
        # Close segments for speakers that are no longer active
        for speaker_idx in list(speaker_segments.keys()):
            if speaker_idx not in current_active:
                # Close this segment
                start_time, _ = speaker_segments.pop(speaker_idx)
                if speaker_idx < len(speaker_id_list):
                    speaker_name = f"speaker_{speaker_id_list[speaker_idx]}"
                else:
                    speaker_name = f"speaker_slot_{speaker_idx}"
                # CRITICAL FIX: Use explicit track=speaker_idx to support overlapping speech correctly
                # Without track, overlapping segments with same time can overwrite each other
                annotation[Segment(start_time, current_time), speaker_idx] = speaker_name
        
        # Open/continue segments for currently active speakers
        for speaker_idx in current_active:
            if speaker_idx not in speaker_segments:
                # Start new segment
                speaker_segments[speaker_idx] = (current_time, next_time)
            else:
                # Continue existing segment (update end time)
                start_time, _ = speaker_segments[speaker_idx]
                speaker_segments[speaker_idx] = (start_time, next_time)
    
    # Close all remaining segments at the end
    final_time = len(active_speakers_per_frame) * frame_shift
    for speaker_idx, (start_time, _) in speaker_segments.items():
        if speaker_idx < len(speaker_id_list):
            speaker_name = f"speaker_{speaker_id_list[speaker_idx]}"
        else:
            speaker_name = f"speaker_slot_{speaker_idx}"
        # CRITICAL FIX: Use explicit track=speaker_idx to support overlapping speech correctly
        annotation[Segment(start_time, final_time), speaker_idx] = speaker_name
    
    return annotation

def calculate_der(predictions, labels, power_set_encoder, speaker_id_list=None, debug=True, frame_shift=0.01, der_frame_shift=None, uri=None, max_duration_seconds=1800, max_duration_warning=True):
    """Calculate Diarization Error Rate with detailed logging and optimized frame merging.
    
    This function now efficiently merges consecutive frames with the same speaker set,
    reducing segments from hundreds of thousands to thousands for better performance.
    
    Args:
        predictions: List of predicted power-set encoded values
        labels: List of ground truth power-set encoded values
        power_set_encoder: PowerSetEncoder instance
        speaker_id_list: Optional list of speaker IDs
        debug: Whether to print debug information
        frame_shift: Time shift between frames in seconds (default: 0.01) - used for model inference
        der_frame_shift: Time shift for DER evaluation (default: 0.05) - can be larger than frame_shift for speed
        uri: Optional URI for the recording (for Annotation)
        max_duration_seconds: Maximum recording duration in seconds before warning/skipping (default: 1800 = 30 min)
        max_duration_warning: If True, log warning for long recordings; if False, skip DER computation
    """
    
    # Use der_frame_shift for evaluation (can be larger than frame_shift for speed)
    if der_frame_shift is None:
        der_frame_shift = 0.05  # Default: 50ms for DER evaluation (5x larger than typical 10ms frame_shift)
    else:
        # FIX: Convert to float if it's a string (e.g., from config)
        try:
            der_frame_shift = float(der_frame_shift)
        except (ValueError, TypeError):
            logger.warning(f"Invalid der_frame_shift value: {der_frame_shift}, using default 0.05")
            der_frame_shift = 0.05
    
    # Calculate total duration
    total_duration = len(predictions) * frame_shift
    
    # Safeguard: Check if recording is too long
    if total_duration > max_duration_seconds:
        if max_duration_warning:
            logger.warning(
                f"[DER] Recording {uri or 'unknown'} is too long ({total_duration:.1f}s > {max_duration_seconds}s). "
                f"DER computation may be slow. Consider using der_frame_shift={der_frame_shift} or "
                f"limiting evaluation to first N seconds."
            )
        else:
            logger.warning(
                f"[DER] Skipping DER for recording {uri or 'unknown'} (duration {total_duration:.1f}s > {max_duration_seconds}s)"
            )
            return float('nan')
    
    # Prepare speaker_id_list
    if speaker_id_list is None:
        speaker_id_list = list(range(power_set_encoder.max_speakers))
    
    # Decode all frames to get active speaker sets
    active_speakers_labels = []
    active_speakers_preds = []
    unique_label_values = set()
    unique_pred_values = set()
    mismatches = 0
    valid_frames_count = 0
    
    # CRITICAL FIX: Create decode cache for performance (only 31 classes max)
    # This avoids repeated decoding of the same power-set values
    decode_cache = {}
    
    # Process frames with optional downsampling for DER evaluation
    # If der_frame_shift > frame_shift, we can downsample to speed up
    downsample_factor = max(1, int(round(der_frame_shift / frame_shift)))
    effective_frame_shift = frame_shift * downsample_factor
    
    # CRITICAL FIX: Process ALL downsampled frames to preserve temporal axis
    # Instead of skipping frames with continue, we use empty sets for invalid frames
    downsampled_indices = [i for i in range(len(predictions)) if i % downsample_factor == 0]
    
    for downsampled_idx, i in enumerate(downsampled_indices):
        pred, label = predictions[i], labels[i]
        
        # Handle invalid labels - use empty set instead of skipping
        if isinstance(label, str):
            if label == '-' or not label.isdigit():
                active_speakers_labels.append(set())
                active_speakers_preds.append(set())
                continue
            label = int(label)
        
        if label == -100:
            # CRITICAL FIX: Use empty set instead of continue to preserve temporal axis
            active_speakers_labels.append(set())
            active_speakers_preds.append(set())
            continue
        
        # Decode to get indices of active speakers (with caching)
        if label not in decode_cache:
            decode_cache[label] = set(power_set_encoder.decode(label))
        true_indices = decode_cache[label].copy()
        
        if pred not in decode_cache:
            decode_cache[pred] = set(power_set_encoder.decode(pred))
        pred_indices = decode_cache[pred].copy()
        
        # Limit indices to valid speaker indices (0 to max_speakers-1)
        max_valid_idx = power_set_encoder.max_speakers - 1
        true_indices = {idx for idx in true_indices if 0 <= idx <= max_valid_idx}
        pred_indices = {idx for idx in pred_indices if 0 <= idx <= max_valid_idx}
        
        active_speakers_labels.append(true_indices)
        active_speakers_preds.append(pred_indices)
        valid_frames_count += 1
        
        unique_label_values.add(label)
        unique_pred_values.add(pred)
        
        if debug and mismatches < 10 and true_indices != pred_indices:
            print(f"[DER DEBUG] Frame {i} (downsampled_idx {downsampled_idx}): label={label}, pred={pred}, true_indices={true_indices}, pred_indices={pred_indices}")
            mismatches += 1
    
    # Use optimized frames_to_annotation to merge consecutive frames
    reference = frames_to_annotation(active_speakers_labels, effective_frame_shift, speaker_id_list, uri=uri)
    hypothesis = frames_to_annotation(active_speakers_preds, effective_frame_shift, speaker_id_list, uri=uri)
    
    # Check if we have any valid frames
    if valid_frames_count == 0:
        logger.warning(f"[DER] No valid frames found for recording {uri or 'unknown'}. Returning NaN.")
        return float('nan')
    
    if debug:
        print(f"[DER DEBUG] Power set encoder max_speakers: {power_set_encoder.max_speakers}")
        print(f"[DER DEBUG] Valid speaker indices: 0 to {power_set_encoder.max_speakers - 1}")
        print(f"[DER DEBUG] Unique label values: {unique_label_values}")
        print(f"[DER DEBUG] Unique pred values: {unique_pred_values}")
        if active_speakers_labels:
            valid_labels = [s for s in active_speakers_labels if len(s) > 0]
            if valid_labels:
                print(f"[DER DEBUG] Active speakers per frame (labels): min={min(len(s) for s in valid_labels)}, max={max(len(s) for s in valid_labels)}, mean={np.mean([len(s) for s in valid_labels]):.2f}")
        if active_speakers_preds:
            valid_preds = [s for s in active_speakers_preds if len(s) > 0]
            if valid_preds:
                print(f"[DER DEBUG] Active speakers per frame (preds): min={min(len(s) for s in valid_preds)}, max={max(len(s) for s in valid_preds)}, mean={np.mean([len(s) for s in valid_preds]):.2f}")
        print(f"[DER DEBUG] Reference segments count: {len(reference)}, first 10: {list(reference.itertracks(yield_label=True))[:10]}")
        print(f"[DER DEBUG] Hypothesis segments count: {len(hypothesis)}, first 10: {list(hypothesis.itertracks(yield_label=True))[:10]}")
        print(f"[DER DEBUG] Frame shift: {frame_shift}s, DER frame shift: {effective_frame_shift}s, Total duration: {total_duration:.2f}s")
        print(f"[DER DEBUG] Downsample factor: {downsample_factor}, Total downsampled frames: {len(active_speakers_labels)}, Valid frames: {valid_frames_count}")
    
    # Create metric with explicit UEM (Universal Evaluation Map)
    metric = DiarizationErrorRate()
    
    # Add explicit UEM if we have duration information
    if total_duration > 0:
        uem = Timeline([Segment(0, total_duration)])
        der = metric(reference, hypothesis, uem=uem)
    else:
        der = metric(reference, hypothesis)
    
    print(f"[DER DEBUG] DER calculation: total downsampled frames = {len(active_speakers_labels)}, valid frames = {valid_frames_count}, DER = {der}")
    return der

def collate_fn_overlapping_speech(batch, enable_bucketing=True):
    """Collate function for overlapping speech dataset. Must be at module level for multiprocessing.
    
    MEMORY FIX: Handles variable-length features and meeting_ids (one per sample, not per-frame).
    Padding is done per-batch here to save memory compared to client-level padding.
    
    Args:
        batch: List of tuples (feature, speaker_embeddings, label, meeting_id)
        enable_bucketing: If True, sort batch by sequence length to reduce padding ratio
    """
    # Optional: Sort by sequence length to reduce padding (bucketing)
    if enable_bucketing and len(batch) > 1:
        # Sort by feature length (descending) to group similar-length sequences
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    max_len = max(x[0].shape[0] for x in batch)
    features = []
    speaker_embeddings = []
    labels = []
    meeting_ids = []  # MEMORY FIX: Store one meeting_id per sample (not per-frame)
    total_original_length = 0
    total_padded_length = 0
    
    for feature, all_embeddings, label, meeting_id in batch:
        original_len = feature.shape[0]
        total_original_length += original_len
        
        # Convert to numpy if needed (from torch tensor)
        if isinstance(feature, torch.Tensor):
            feature = feature.cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()
        
        # MEMORY FIX: Pad only per-batch (not client-level)
        if feature.shape[0] < max_len:
            pad_len = max_len - feature.shape[0]
            feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
            label = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
            total_padded_length += pad_len
        else:
            total_padded_length += 0
        
        features.append(feature)
        speaker_embeddings.append(all_embeddings)  # [num_speakers, 192]
        labels.append(label)
        # MEMORY FIX: Store one meeting_id per sample (string/int), not per-frame array
        meeting_ids.append(meeting_id)
    
    features = torch.tensor(np.array(features), dtype=torch.float32)
    speaker_embeddings = torch.stack(speaker_embeddings).float()  # [batch, num_speakers, 192]
    labels = torch.tensor(np.array(labels), dtype=torch.long)
    # MEMORY FIX: meeting_ids is a list of strings/ints (one per sample)
    # Convert to numpy array for compatibility, but keep as 1D array
    meeting_ids = np.array(meeting_ids, dtype=object)
    
    # Log padding ratio (only for first few batches to avoid spam)
    if not hasattr(collate_fn_overlapping_speech, '_batch_count'):
        collate_fn_overlapping_speech._batch_count = 0
    collate_fn_overlapping_speech._batch_count += 1
    
    if collate_fn_overlapping_speech._batch_count <= 5:
        padding_ratio = total_padded_length / (total_original_length + total_padded_length) if (total_original_length + total_padded_length) > 0 else 0.0
        logger.debug(f"Collate batch {collate_fn_overlapping_speech._batch_count}: padding_ratio={padding_ratio:.2%}, max_len={max_len}, avg_len={total_original_length/len(batch):.1f}")
    
    return features, speaker_embeddings, labels, meeting_ids

def prepare_data_loaders(grouped_train, grouped_validation, grouped_test, speaker_encoder, power_set_encoder, batch_size=4, speaker_to_embedding=None, N=4, chunk_size=500, max_sequence_length=None, enable_persistent_workers=False, enable_bucketing=True):
    # Import and log function start
    print_function_start("prepare_data_loaders", 
                        grouped_train_len=len(grouped_train), 
                        grouped_validation_len=len(grouped_validation), 
                        grouped_test_len=len(grouped_test),
                        batch_size=batch_size, 
                        N=N)
    """Prepare data loaders for training, validation and testing.
    
    Args:
        chunk_size: Number of samples to process at once during dataset creation (default: 500)
                   Smaller values use less memory but are slower.
        max_sequence_length: Optional maximum sequence length to truncate longer sequences (default: None = no limit)
                            Use this to limit memory usage when working with large datasets.
    """
    # Create datasets with memory-optimized chunk processing
    logger.info(f"Creating datasets with chunk_size={chunk_size} for memory optimization...")
    if max_sequence_length is not None:
        logger.info(f"Using max_sequence_length={max_sequence_length} to limit memory usage")
    
    # Create datasets - now returns speaker_ids and meeting_to_slot_speakers
    train_features, train_labels, train_meeting_ids, train_speaker_ids, train_meeting_to_slot_speakers = create_dataset_from_grouped(
        grouped_train, speaker_encoder, power_set_encoder, N, chunk_size=chunk_size, max_sequence_length=max_sequence_length
    )
    val_features, val_labels, val_meeting_ids, val_speaker_ids, val_meeting_to_slot_speakers = create_dataset_from_grouped(
        grouped_validation, speaker_encoder, power_set_encoder, N, chunk_size=chunk_size, max_sequence_length=max_sequence_length
    )
    test_features, test_labels, test_meeting_ids, test_speaker_ids, test_meeting_to_slot_speakers = create_dataset_from_grouped(
        grouped_test, speaker_encoder, power_set_encoder, N, chunk_size=chunk_size, max_sequence_length=max_sequence_length
    )
    
    # Compute meeting-specific embeddings for each dataset
    logger.info("Computing meeting-specific speaker embeddings...")
    train_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_train, speaker_encoder)
    val_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_validation, speaker_encoder) if grouped_validation else {}
    test_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_test, speaker_encoder) if grouped_test else {}
    
    # Create datasets with meeting-specific embeddings and slot mappings
    train_dataset = OverlappingSpeechDataset(
        features=train_features,
        labels=train_labels,
        meeting_ids=train_meeting_ids,
        speaker_ids=train_speaker_ids,
        meeting_to_speaker_embedding=train_meeting_to_speaker_embedding,
        meeting_to_slot_speakers=train_meeting_to_slot_speakers,
        max_speakers=N
    )
    val_dataset = OverlappingSpeechDataset(
        features=val_features,
        labels=val_labels,
        meeting_ids=val_meeting_ids,
        speaker_ids=val_speaker_ids,
        meeting_to_speaker_embedding=val_meeting_to_speaker_embedding,
        meeting_to_slot_speakers=val_meeting_to_slot_speakers,
        max_speakers=N
    )
    test_dataset = OverlappingSpeechDataset(
        features=test_features,
        labels=test_labels,
        meeting_ids=test_meeting_ids,
        speaker_ids=test_speaker_ids,
        meeting_to_speaker_embedding=test_meeting_to_speaker_embedding,
        meeting_to_slot_speakers=test_meeting_to_slot_speakers,
        max_speakers=N
    )
    
    # Optimize DataLoader with num_workers and pin_memory for faster data loading
    # Use module-level collate_fn for multiprocessing compatibility
    num_workers = min(8, os.cpu_count() or 1)  # Use up to 8 workers, but not more than available CPUs
    pin_memory = torch.cuda.is_available()  # Pin memory only if CUDA is available
    # persistent_workers can cause semaphore leaks if not properly closed
    # DISABLED by default to avoid resource leaks - the performance gain is minimal (~1-2 sec per epoch)
    # Enable only if you have many epochs and proper cleanup is working
    use_persistent_workers = enable_persistent_workers and num_workers > 0  # Only enable if explicitly requested
    
    # Create collate function with bucketing enabled
    collate_fn = partial(collate_fn_overlapping_speech, enable_bucketing=enable_bucketing)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers  # Keep workers alive between epochs (can cause leaks if not closed properly)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers
    )
    
    # Import and use statistics function for logging
    print_data_loaders_info(train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, batch_size)
    
    # Log function completion
    print_function_end("prepare_data_loaders", 
                      f"Created 3 data loaders: train({len(train_loader)} batches), val({len(val_loader)} batches), test({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader

def build_train_dataset_for_grouped_meetings(
    grouped_train_subset: dict,
    speaker_encoder,
    power_set_encoder,
    max_speakers: int,
    chunk_size: int,
    max_sequence_length: int,
):
    """Build train dataset for a subset of meetings (for one FL client).
    
    This function builds only the train dataset for a client's subset of meetings.
    Use build_eval_dataset() to build shared val/test datasets once globally.
    
    Args:
        grouped_train_subset: Dictionary of meeting_id to samples (subset for one client)
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder instance
        max_speakers: Maximum number of speaker slots (N)
        chunk_size: Number of samples to process at once
        max_sequence_length: Maximum sequence length to truncate longer sequences
        
    Returns:
        OverlappingSpeechDataset: Train dataset for the client
    """
    logger.info(f"Building train dataset for FL client...")
    logger.info(f"  Train meetings: {len(grouped_train_subset)}")
    logger.info(f"  Max speakers: {max_speakers}, chunk_size: {chunk_size}, max_sequence_length: {max_sequence_length}")
    
    # Create dataset using the same logic as prepare_data_loaders
    train_features, train_labels, train_meeting_ids, train_speaker_ids, train_meeting_to_slot_speakers = create_dataset_from_grouped(
        grouped_train_subset, speaker_encoder, power_set_encoder, max_speakers, 
        chunk_size=chunk_size, max_sequence_length=max_sequence_length
    )
    
    # Compute meeting-specific embeddings for train subset
    logger.info(f"Computing meeting-specific speaker embeddings for {len(grouped_train_subset)} train meetings...")
    train_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_train_subset, speaker_encoder)
    
    # Create train dataset with meeting-specific embeddings and slot mappings
    train_dataset = OverlappingSpeechDataset(
        features=train_features,
        labels=train_labels,
        meeting_ids=train_meeting_ids,
        speaker_ids=train_speaker_ids,
        meeting_to_speaker_embedding=train_meeting_to_speaker_embedding,
        meeting_to_slot_speakers=train_meeting_to_slot_speakers,
        max_speakers=max_speakers
    )
    
    logger.info(f"✓ Built train dataset: {len(train_dataset)} samples")
    
    return train_dataset


def build_eval_dataset(
    grouped_eval: dict,
    speaker_encoder,
    power_set_encoder,
    max_speakers: int,
    chunk_size: int,
    max_sequence_length: int,
    dataset_name: str = "eval",
):
    """Build evaluation dataset (val or test) for shared use across all FL clients.
    
    This function builds a single eval dataset that can be safely shared across all clients.
    OverlappingSpeechDataset is read-only in __getitem__, so it's safe for concurrent access.
    
    Args:
        grouped_eval: Dictionary of meeting_id to samples (validation or test data)
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder instance
        max_speakers: Maximum number of speaker slots (N)
        chunk_size: Number of samples to process at once
        max_sequence_length: Maximum sequence length to truncate longer sequences
        dataset_name: Name for logging (e.g., "val" or "test")
        
    Returns:
        OverlappingSpeechDataset: Evaluation dataset (safe to share across clients)
    """
    if not grouped_eval:
        logger.warning(f"Empty {dataset_name} dataset provided, returning empty dataset")
        # Return empty dataset structure
        return OverlappingSpeechDataset(
            features=[],
            labels=[],
            meeting_ids=[],
            speaker_ids=[],
            meeting_to_speaker_embedding={},
            meeting_to_slot_speakers={},
            max_speakers=max_speakers
        )
    
    logger.info(f"Building {dataset_name} dataset (shared across all clients)...")
    logger.info(f"  {dataset_name.capitalize()} meetings: {len(grouped_eval)}")
    logger.info(f"  Max speakers: {max_speakers}, chunk_size: {chunk_size}, max_sequence_length: {max_sequence_length}")
    
    # Create dataset using the same logic as prepare_data_loaders
    eval_features, eval_labels, eval_meeting_ids, eval_speaker_ids, eval_meeting_to_slot_speakers = create_dataset_from_grouped(
        grouped_eval, speaker_encoder, power_set_encoder, max_speakers,
        chunk_size=chunk_size, max_sequence_length=max_sequence_length
    )
    
    # Compute meeting-specific embeddings for eval dataset
    logger.info(f"Computing meeting-specific speaker embeddings for {len(grouped_eval)} {dataset_name} meetings...")
    eval_meeting_to_speaker_embedding = compute_speaker_embeddings(grouped_eval, speaker_encoder)
    
    # Create eval dataset with meeting-specific embeddings and slot mappings
    eval_dataset = OverlappingSpeechDataset(
        features=eval_features,
        labels=eval_labels,
        meeting_ids=eval_meeting_ids,
        speaker_ids=eval_speaker_ids,
        meeting_to_speaker_embedding=eval_meeting_to_speaker_embedding,
        meeting_to_slot_speakers=eval_meeting_to_slot_speakers,
        max_speakers=max_speakers
    )
    
    logger.info(f"✓ Built {dataset_name} dataset: {len(eval_dataset)} samples (safe to share across clients)")
    
    return eval_dataset


def build_meeting_datasets(
    grouped_train_subset: dict,
    grouped_val: dict,
    grouped_test: dict,
    speaker_encoder,
    power_set_encoder,
    max_speakers: int,
    chunk_size: int,
    max_sequence_length: int,
):
    """Build datasets for a subset of meetings (for FL clients).
    
    DEPRECATED: This function is kept for backward compatibility but is inefficient.
    Use build_train_dataset_for_grouped_meetings() and build_eval_dataset() instead
    to avoid rebuilding val/test datasets for each client.
    
    Args:
        grouped_train_subset: Dictionary of meeting_id to samples (subset for one client)
        grouped_val: Dictionary of meeting_id to samples (validation, shared across clients)
        grouped_test: Dictionary of meeting_id to samples (test, shared across clients)
        speaker_encoder: Speaker encoder model
        power_set_encoder: PowerSetEncoder instance
        max_speakers: Maximum number of speaker slots (N)
        chunk_size: Number of samples to process at once
        max_sequence_length: Maximum sequence length to truncate longer sequences
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) - OverlappingSpeechDataset objects
    """
    logger.warning("build_meeting_datasets is deprecated. Use build_train_dataset_for_grouped_meetings() and build_eval_dataset() instead.")
    
    train_dataset = build_train_dataset_for_grouped_meetings(
        grouped_train_subset, speaker_encoder, power_set_encoder, max_speakers,
        chunk_size, max_sequence_length
    )
    val_dataset = build_eval_dataset(
        grouped_val, speaker_encoder, power_set_encoder, max_speakers,
        chunk_size, max_sequence_length, dataset_name="val"
    )
    test_dataset = build_eval_dataset(
        grouped_test, speaker_encoder, power_set_encoder, max_speakers,
        chunk_size, max_sequence_length, dataset_name="test"
    )
    
    return train_dataset, val_dataset, test_dataset

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

def compute_speaker_embeddings(grouped_data, speaker_encoder, target_sr=16000, min_duration=0.5, target_duration=2.0):
    """Computes speaker embedding for each unique (meeting_id, speaker_id) pair.
    
    Returns meeting-specific embeddings: dict[meeting_id, dict[speaker_id, torch.Tensor]]
    Each embedding is computed from the longest non-overlap segment (or accumulated segments) of that speaker.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        speaker_encoder: Speaker encoder model (expects 16kHz audio)
        target_sr: Target sampling rate (default: 16000 Hz)
        min_duration: Minimum segment duration in seconds (default: 0.5s)
        target_duration: Target duration for accumulated segments in seconds (default: 2.0s)
    """
    # librosa is already imported at module level
    
    meeting_to_speaker_embedding = {}
    
    # Check sampling rates in first 20 samples
    sample_rates = []
    for meeting_id, samples in list(grouped_data.items())[:1]:  # Check first meeting
        for sample in samples[:20]:  # First 20 samples
            if "audio" in sample and "sampling_rate" in sample["audio"]:
                sample_rates.append(sample["audio"]["sampling_rate"])
            elif "audio" in sample and hasattr(sample["audio"], "get"):
                sample_rates.append(sample["audio"].get("sampling_rate", None))
    
    if sample_rates:
        unique_srs = list(set(sr for sr in sample_rates if sr is not None))
        logger.info(f"Sample rates found in first 20 samples: {unique_srs}")
        if len(unique_srs) > 0 and unique_srs[0] != target_sr:
            logger.warning(f"⚠️  WARNING: Audio sampling rate is {unique_srs[0]} Hz, not {target_sr} Hz!")
            logger.warning(f"   Will resample to {target_sr} Hz for speaker embeddings.")
    
    for meeting_id, samples in grouped_data.items():
        meeting_to_speaker_embedding[meeting_id] = {}
        
        # Group samples by speaker_id for this meeting with duration info
        speaker_to_segments = {}
        for sample in samples:
            sid = sample["speaker_id"]
            # Skip if speaker_id is a list (overlap segment) - we only want non-overlap segments
            if isinstance(sid, list):
                continue
            
            audio_array = sample["audio"]["array"]
            sampling_rate = sample["audio"].get("sampling_rate", target_sr)
            
            # Calculate duration
            duration = len(audio_array) / sampling_rate if sampling_rate > 0 else 0.0
            
            if sid not in speaker_to_segments:
                speaker_to_segments[sid] = []
            
            speaker_to_segments[sid].append({
                "audio": audio_array,
                "sampling_rate": sampling_rate,
                "duration": duration,
                "begin_time": sample.get("begin_time", 0),
                "end_time": sample.get("end_time", duration)
            })
        
        # Compute embedding for each speaker in this meeting
        for sid, segments in speaker_to_segments.items():
            if len(segments) == 0:
                logger.warning(f"Meeting {meeting_id}: No audio segments for speaker {sid}")
                continue
            
            # Strategy 1: Find longest segment
            longest_segment = max(segments, key=lambda s: s["duration"])
            
            # Strategy 2: If longest is too short, accumulate segments up to target_duration
            if longest_segment["duration"] < min_duration:
                logger.warning(f"Meeting {meeting_id}, speaker {sid}: Longest segment is only {longest_segment['duration']:.2f}s (< {min_duration}s). Using it anyway.")
                selected_segment = longest_segment
            elif longest_segment["duration"] < target_duration:
                # Try to accumulate segments
                sorted_segments = sorted(segments, key=lambda s: s["duration"], reverse=True)
                accumulated_audio = []
                accumulated_duration = 0.0
                accumulated_sr = sorted_segments[0]["sampling_rate"]
                
                for seg in sorted_segments:
                    if accumulated_duration >= target_duration:
                        break
                    accumulated_audio.append(seg["audio"])
                    accumulated_duration += seg["duration"]
                    accumulated_sr = seg["sampling_rate"]  # Assume same SR
                
                if len(accumulated_audio) > 1 and accumulated_duration >= min_duration:
                    # Concatenate accumulated segments
                    selected_audio = np.concatenate(accumulated_audio)
                    selected_sr = accumulated_sr
                    logger.info(f"Meeting {meeting_id}, speaker {sid}: Accumulated {len(accumulated_audio)} segments to {accumulated_duration:.2f}s")
                else:
                    # Use longest segment
                    selected_audio = longest_segment["audio"]
                    selected_sr = longest_segment["sampling_rate"]
            else:
                # Use longest segment (it's long enough)
                selected_audio = longest_segment["audio"]
                selected_sr = longest_segment["sampling_rate"]
            
            # Resample to target_sr if needed
            if selected_sr != target_sr:
                logger.info(f"Meeting {meeting_id}, speaker {sid}: Resampling from {selected_sr} Hz to {target_sr} Hz")
                selected_audio = librosa.resample(selected_audio, orig_sr=selected_sr, target_sr=target_sr)
            
            # Ensure minimum length (at least 0.1s)
            min_samples = int(target_sr * 0.1)
            if len(selected_audio) < min_samples:
                logger.warning(f"Meeting {meeting_id}, speaker {sid}: Audio too short ({len(selected_audio)} samples), padding to {min_samples}")
                selected_audio = np.pad(selected_audio, (0, min_samples - len(selected_audio)), mode='constant')
            
            # Convert to tensor and compute embedding
            audio_tensor = torch.tensor(selected_audio, dtype=torch.float32).unsqueeze(0)  # [1, time]
            with torch.no_grad():
                emb = speaker_encoder.encode_batch(audio_tensor)
                emb = emb.squeeze().cpu()
            meeting_to_speaker_embedding[meeting_id][sid] = emb
    
    logger.info(f"Computed embeddings for {len(meeting_to_speaker_embedding)} meetings")
    total_speakers = sum(len(speakers) for speakers in meeting_to_speaker_embedding.values())
    logger.info(f"Total (meeting, speaker) pairs: {total_speakers}")
    
    return meeting_to_speaker_embedding

class OverlappingSpeechDataset(Dataset):
    """Dataset for overlapping speech diarization.
    
    MEMORY FIX: Supports both variable-length (list of arrays) and padded (numpy array) features.
    Variable-length features save memory by avoiding client-level padding.
    
    Now uses meeting-specific embeddings in slot order matching the labels.
    
    THREAD-SAFETY: This dataset is read-only and safe for concurrent access across multiple FL clients.
    The __getitem__ method only reads data and creates new tensors, never modifies internal state.
    The only mutable state (_missing_meetings, _missing_embeddings) is for logging only and doesn't
    affect data access. Multiple clients can safely share the same OverlappingSpeechDataset instance
    for evaluation (val/test datasets).
    """
    def __init__(self, features, labels, meeting_ids, speaker_ids: list, 
                 meeting_to_speaker_embedding: dict, meeting_to_slot_speakers: dict, 
                 max_speakers: int = 4):
        # MEMORY FIX: Accept both list (variable-length) and numpy array (padded) for features/labels
        self.features = features
        self.labels = labels
        self.meeting_ids = meeting_ids
        self.speaker_ids = speaker_ids
        self.meeting_to_speaker_embedding = meeting_to_speaker_embedding
        self.meeting_to_slot_speakers = meeting_to_slot_speakers
        self.max_speakers = max_speakers
        
        # Verify that all arrays/lists have the same length
        if len(features) != len(labels) or len(features) != len(meeting_ids) or len(features) != len(speaker_ids):
            raise ValueError(
                f"Mismatch in dataset lengths: features={len(features)}, labels={len(labels)}, "
                f"meeting_ids={len(meeting_ids)}, speaker_ids={len(speaker_ids)}"
            )
        
        # Get embedding dimension from first available embedding
        emb_dim = None
        for meeting_id, speaker_embeddings in meeting_to_speaker_embedding.items():
            if speaker_embeddings:
                first_emb = next(iter(speaker_embeddings.values()))
                emb_dim = first_emb.shape[0]
                break
        
        if emb_dim is None:
            raise ValueError("No embeddings found in meeting_to_speaker_embedding")
        
        self.emb_dim = emb_dim
        
        # Log the structure for debugging
        logger.info(f"[OverlappingSpeechDataset] Dataset size: {len(features)} samples")
        logger.info(f"[OverlappingSpeechDataset] Max speakers: {max_speakers}, Embedding dim: {emb_dim}")
        logger.info(f"[OverlappingSpeechDataset] Meetings with embeddings: {len(meeting_to_speaker_embedding)}")
        logger.info(f"[OverlappingSpeechDataset] Meetings with slot mappings: {len(meeting_to_slot_speakers)}")
        
        # Log example slot mappings
        example_meetings = list(meeting_to_slot_speakers.keys())[:3]
        for mid in example_meetings:
            slot_speakers = meeting_to_slot_speakers[mid]
            logger.info(f"[OverlappingSpeechDataset] Example meeting {mid}: slot_speakers={slot_speakers}")
        
        # Initialize tracking for missing embeddings warnings
        self._missing_meetings = set()
        self._missing_embeddings = set()
        self._missing_speaker_embeddings = []
    
    def get_missing_embeddings_summary(self):
        """Get summary of missing embeddings warnings."""
        return {
            'missing_meetings_in_slot_mapping': len(self._missing_meetings),
            'missing_meetings_in_embeddings': len(self._missing_embeddings),
            'missing_speaker_embeddings': len(self._missing_speaker_embeddings)
        }
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple:
        # MEMORY FIX: Handle both variable-length (list) and padded (numpy array) features
        if isinstance(self.features, list):
            # Variable-length: features[idx] is already a numpy array
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
        else:
            # Padded: features is a numpy array, slice it
            feature = torch.tensor(self.features[idx], dtype=torch.float32)
        
        if isinstance(self.labels, list):
            # Variable-length: labels[idx] is already a numpy array
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            # Padded: labels is a numpy array, slice it
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # MEMORY FIX: meeting_ids is now a 1D array of strings/ints (one per sample)
        meeting_id_raw = self.meeting_ids[idx]
        
        # Convert meeting_id to hashable type (string or int)
        # Handle numpy arrays and other types
        if isinstance(meeting_id_raw, np.ndarray):
            # If it's a numpy array, extract the scalar value
            meeting_id = meeting_id_raw.item() if meeting_id_raw.size == 1 else str(meeting_id_raw)
        elif isinstance(meeting_id_raw, (list, tuple)):
            # If it's a list/tuple, convert to string
            meeting_id = str(meeting_id_raw[0]) if len(meeting_id_raw) > 0 else str(meeting_id_raw)
        else:
            # Already a string, int, or other hashable type
            meeting_id = meeting_id_raw
        
        # Check bounds for speaker_ids
        if idx >= len(self.speaker_ids):
            logger.error(f"Index {idx} out of range for speaker_ids (length: {len(self.speaker_ids)})")
            logger.error(f"Features length: {len(self.features)}, Labels length: {len(self.labels)}, Meeting IDs length: {len(self.meeting_ids)}")
            raise IndexError(f"speaker_ids index {idx} out of range (length: {len(self.speaker_ids)})")
        
        # Get slot_speakers for this meeting (in slot order: slot 0, 1, 2, ...)
        slot_speakers = self.meeting_to_slot_speakers.get(meeting_id, [])
        if not slot_speakers:
            # Track missing meetings for summary
            if not hasattr(self, '_missing_meetings'):
                self._missing_meetings = set()
            self._missing_meetings.add(meeting_id)
            if len(self._missing_meetings) <= 10:  # Log first 10
                logger.warning(f"Meeting {meeting_id} not found in meeting_to_slot_speakers. Using empty slot list.")
        
        # Get meeting-specific embeddings
        meeting_embeddings = self.meeting_to_speaker_embedding.get(meeting_id, {})
        if not meeting_embeddings:
            # Track missing embeddings for summary
            if not hasattr(self, '_missing_embeddings'):
                self._missing_embeddings = set()
            self._missing_embeddings.add(meeting_id)
            if len(self._missing_embeddings) <= 10:  # Log first 10
                logger.warning(f"Meeting {meeting_id} not found in meeting_to_speaker_embedding. Using zero embeddings.")
        
        # Build speaker_embeddings tensor in slot order: [emb_slot0, emb_slot1, ..., emb_slotN-1]
        speaker_embeddings_list = []
        missing_embedding_count = 0
        for slot_idx in range(self.max_speakers):
            if slot_idx < len(slot_speakers):
                speaker_id = slot_speakers[slot_idx]
                if speaker_id in meeting_embeddings:
                    speaker_embeddings_list.append(meeting_embeddings[speaker_id])
                else:
                    missing_embedding_count += 1
                    # Track missing speaker embeddings for summary
                    if not hasattr(self, '_missing_speaker_embeddings'):
                        self._missing_speaker_embeddings = []
                    self._missing_speaker_embeddings.append((meeting_id, speaker_id, slot_idx))
                    if len(self._missing_speaker_embeddings) <= 20:  # Log first 20
                        logger.warning(f"Missing embedding for meeting {meeting_id}, speaker {speaker_id} at slot {slot_idx}. Using zeros.")
                    speaker_embeddings_list.append(torch.zeros(self.emb_dim))
            else:
                # Pad with zeros if fewer speakers than max_speakers
                speaker_embeddings_list.append(torch.zeros(self.emb_dim))
        
        # Stack to create [max_speakers, emb_dim] tensor
        speaker_embeddings = torch.stack(speaker_embeddings_list).float()
        
        return feature, speaker_embeddings, label, meeting_id
    
    def get_speaker_id_list(self):
        """Get the stable speaker ID list for DER calculation.
        
        Returns a list of speaker IDs for backward compatibility.
        Note: This is now meeting-specific, so this method may not be accurate.
        """
        # For backward compatibility, return first meeting's slot speakers
        if self.meeting_to_slot_speakers:
            first_meeting = next(iter(self.meeting_to_slot_speakers.keys()))
            return self.meeting_to_slot_speakers[first_meeting]
        return [] 
    


