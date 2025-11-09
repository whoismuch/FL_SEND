"""
Statistics module for FL-SEND diarization project.
Contains functions for analyzing and reporting dataset statistics.
"""

from datetime import datetime
from typing import Dict, List, Any


def find_overlapping_segments(samples):
    """
    Find overlapping audio segments in a meeting.
    
    Args:
        samples: List of audio samples for a meeting
        
    Returns:
        List of overlapping segments with speaker IDs and timestamps
    """
    overlapping_segments = []
    
    # Sort samples by begin_time
    sorted_samples = sorted(samples, key=lambda x: x["begin_time"])
    
    # Find overlapping segments
    for i in range(len(sorted_samples)):
        current = sorted_samples[i]
        for j in range(i + 1, len(sorted_samples)):
            next_seg = sorted_samples[j]
            
            # Check if segments overlap
            if next_seg["begin_time"] < current["end_time"]:
                overlap_begin = max(current["begin_time"], next_seg["begin_time"])
                overlap_end = min(current["end_time"], next_seg["end_time"])
                
                overlapping_segments.append({
                    "speaker_1": current["speaker_id"],
                    "speaker_2": next_seg["speaker_id"],
                    "begin_time": overlap_begin,
                    "end_time": overlap_end,
                    "duration": overlap_end - overlap_begin
                })
    
    return overlapping_segments


def find_max_concurrent_speakers(samples):
    """
    Find the maximum number of concurrent speakers in a meeting.
    
    Args:
        samples: List of audio samples for a meeting
        
    Returns:
        int: Maximum number of concurrent speakers
    """
    if not samples:
        return 0
    
    # Create events for speaker start/end times
    events = []
    for sample in samples:
        events.append((sample["begin_time"], "start", sample["speaker_id"]))
        events.append((sample["end_time"], "end", sample["speaker_id"]))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    # Track concurrent speakers
    active_speakers = set()
    max_concurrent = 0
    
    for time, event_type, speaker_id in events:
        if event_type == "start":
            active_speakers.add(speaker_id)
        else:  # "end"
            active_speakers.discard(speaker_id)
        
        max_concurrent = max(max_concurrent, len(active_speakers))
    
    return max_concurrent


def print_meeting_statistics(grouped_train: Dict, grouped_validation: Dict, grouped_test: Dict) -> None:
    """
    Print detailed statistics for meetings in train, validation, and test sets.
    
    Args:
        grouped_train: Dictionary of meeting_id to samples for training set
        grouped_validation: Dictionary of meeting_id to samples for validation set
        grouped_test: Dictionary of meeting_id to samples for test set
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== MEETING STATISTICS ==========================================")
    
    # Training set statistics
    print(f"\n[{datetime.now()}] STATS: Training set meetings:")
    train_audio_counts = []
    train_overlap_counts = []
    train_max_concurrent = []
    for meeting_id, samples in grouped_train.items():
        audio_count = len(samples)
        train_audio_counts.append(audio_count)
        
        # Find overlapping segments
        overlapping_segments = find_overlapping_segments(samples)
        train_overlap_counts.append(len(overlapping_segments))
        
        # Find maximum concurrent speakers
        max_concurrent = find_max_concurrent_speakers(samples)
        train_max_concurrent.append(max_concurrent)
        
        print(f"  Meeting {meeting_id}: {audio_count} audio segments, {len(overlapping_segments)} overlapping segments, max {max_concurrent} concurrent speakers")
        
        # Print first 5 overlapping segments
        if overlapping_segments:
            print(f"    First 5 overlapping segments:")
            for i, overlap in enumerate(overlapping_segments[:5]):
                print(f"      {i+1}. Speakers {overlap['speaker_1']} & {overlap['speaker_2']}: "
                      f"{overlap['begin_time']:.2f}s - {overlap['end_time']:.2f}s "
                      f"(duration: {overlap['duration']:.2f}s)")
    
    # Validation set statistics
    print(f"\n[{datetime.now()}] STATS: Validation set meetings:")
    val_audio_counts = []
    val_overlap_counts = []
    val_max_concurrent = []
    for meeting_id, samples in grouped_validation.items():
        audio_count = len(samples)
        val_audio_counts.append(audio_count)
        
        # Find overlapping segments
        overlapping_segments = find_overlapping_segments(samples)
        val_overlap_counts.append(len(overlapping_segments))
        
        # Find maximum concurrent speakers
        max_concurrent = find_max_concurrent_speakers(samples)
        val_max_concurrent.append(max_concurrent)
        
        print(f"  Meeting {meeting_id}: {audio_count} audio segments, {len(overlapping_segments)} overlapping segments, max {max_concurrent} concurrent speakers")
        
        # Print first 5 overlapping segments
        if overlapping_segments:
            print(f"    First 5 overlapping segments:")
            for i, overlap in enumerate(overlapping_segments[:5]):
                print(f"      {i+1}. Speakers {overlap['speaker_1']} & {overlap['speaker_2']}: "
                      f"{overlap['begin_time']:.2f}s - {overlap['end_time']:.2f}s "
                      f"(duration: {overlap['duration']:.2f}s)")
    
    # Test set statistics
    print(f"\n[{datetime.now()}] STATS: Test set meetings:")
    test_audio_counts = []
    test_overlap_counts = []
    test_max_concurrent = []
    for meeting_id, samples in grouped_test.items():
        audio_count = len(samples)
        test_audio_counts.append(audio_count)
        
        # Find overlapping segments
        overlapping_segments = find_overlapping_segments(samples)
        test_overlap_counts.append(len(overlapping_segments))
        
        # Find maximum concurrent speakers
        max_concurrent = find_max_concurrent_speakers(samples)
        test_max_concurrent.append(max_concurrent)
        
        print(f"  Meeting {meeting_id}: {audio_count} audio segments, {len(overlapping_segments)} overlapping segments, max {max_concurrent} concurrent speakers")
        
        # Print first 5 overlapping segments
        if overlapping_segments:
            print(f"    First 5 overlapping segments:")
            for i, overlap in enumerate(overlapping_segments[:5]):
                print(f"      {i+1}. Speakers {overlap['speaker_1']} & {overlap['speaker_2']}: "
                      f"{overlap['begin_time']:.2f}s - {overlap['end_time']:.2f}s "
                      f"(duration: {overlap['duration']:.2f}s)")
    
    # Summary statistics
    all_counts = train_audio_counts + val_audio_counts + test_audio_counts
    all_overlap_counts = train_overlap_counts + val_overlap_counts + test_overlap_counts
    all_max_concurrent = train_max_concurrent + val_max_concurrent + test_max_concurrent
    print(f"\n[{datetime.now()}] STATS: ========================================== SUMMARY STATISTICS ==========================================")
    print(f"Total meetings: {len(all_counts)}")
    print(f"Total audio segments: {sum(all_counts)}")
    print(f"Total overlapping segments: {sum(all_overlap_counts)}")
    
    # Analysis of concurrent speakers
    print(f"\n CONCURRENT SPEAKERS ANALYSIS:")
    print(f"  - Maximum number of concurrent speakers: {max(all_max_concurrent)}")
    print(f"  - Average maximum number: {sum(all_max_concurrent)/len(all_max_concurrent):.1f}")
    print(f"  - Median maximum number: {sorted(all_max_concurrent)[len(all_max_concurrent)//2]}")
    
    # Distribution of max concurrent speakers
    concurrent_distribution = {}
    for max_conc in all_max_concurrent:
        concurrent_distribution[max_conc] = concurrent_distribution.get(max_conc, 0) + 1
    
    print(f"  - Distribution of maximum concurrent speakers:")
    for max_conc in sorted(concurrent_distribution.keys()):
        count = concurrent_distribution[max_conc]
        percentage = (count / len(all_max_concurrent)) * 100
        print(f"    {max_conc} speakers: {count} meetings ({percentage:.1f}%)")
    
    # Per-split statistics
    print(f"\nTraining set: {len(train_audio_counts)} meetings, {sum(train_audio_counts)} segments, {sum(train_overlap_counts)} overlaps (avg: {sum(train_audio_counts)/len(train_audio_counts):.1f} segments, {sum(train_overlap_counts)/len(train_overlap_counts):.1f} overlaps)")
    print(f"Validation set: {len(val_audio_counts)} meetings, {sum(val_audio_counts)} segments, {sum(val_overlap_counts)} overlaps (avg: {sum(val_audio_counts)/len(val_audio_counts):.1f} segments, {sum(val_overlap_counts)/len(val_overlap_counts):.1f} overlaps)")
    print(f"Test set: {len(test_audio_counts)} meetings, {sum(test_audio_counts)} segments, {sum(test_overlap_counts)} overlaps (avg: {sum(test_audio_counts)/len(test_audio_counts):.1f} segments, {sum(test_overlap_counts)/len(test_overlap_counts):.1f} overlaps)")


def print_dataset_overview(dataset_name: str, total_samples: int, test_size: int) -> None:
    """
    Print basic dataset overview information.
    
    Args:
        dataset_name: Name of the dataset
        total_samples: Total number of samples in the dataset
        test_size: Number of samples used for testing
    """
    print(f"[{datetime.now()}] STATS: Dataset: {dataset_name}")
    print(f"[{datetime.now()}] STATS: Total samples: {total_samples}")
    print(f"[{datetime.now()}] STATS: Using subset of {test_size} samples for testing")


def print_grouping_results(grouped_train: Dict, grouped_validation: Dict, grouped_test: Dict) -> None:
    """
    Print results of data grouping by meeting ID.
    
    Args:
        grouped_train: Dictionary of meeting_id to samples for training set
        grouped_validation: Dictionary of meeting_id to samples for validation set
        grouped_test: Dictionary of meeting_id to samples for test set
    """
    print(f"[{datetime.now()}] STATS: ========================================== DATA GROUPING ==========================================")
    print(f"[{datetime.now()}] STATS: Grouped {len(grouped_train)} meetings from training set")
    print(f"[{datetime.now()}] STATS: Grouped {len(grouped_validation)} meetings from validation set")
    print(f"[{datetime.now()}] STATS: Grouped {len(grouped_test)} meetings from test set")


def print_experiment_config(num_clients: int, num_rounds: int, num_epochs: int, test_size: int) -> None:
    """
    Print experiment configuration parameters.
    
    Args:
        num_clients: Number of clients in federated learning
        num_rounds: Number of communication rounds
        num_epochs: Number of training epochs per round
        test_size: Number of samples used for testing
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== EXPERIMENT CONFIGURATION ==========================================")
    print(f"Number of clients: {num_clients}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Number of epochs per round: {num_epochs}")
    print(f"Test size: {test_size}")


def print_training_progress(round_num: int, epoch_num: int, train_loss: float, val_loss: float, der: float) -> None:
    """
    Print training progress information.
    
    Args:
        round_num: Current round number
        epoch_num: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
        der: Diarization Error Rate
    """
    print(f"[{datetime.now()}] Round {round_num}, Epoch {epoch_num}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, DER: {der:.4f}")


def print_final_results(final_der: float, best_der: float, total_time: float) -> None:
    """
    Print final experiment results.
    
    Args:
        final_der: Final DER score
        best_der: Best DER score achieved
        total_time: Total experiment time in seconds
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== FINAL RESULTS ==========================================")
    print(f"Final DER: {final_der:.4f}")
    print(f"Best DER: {best_der:.4f}")
    print(f"Total experiment time: {total_time:.2f} seconds")


def analyze_speaker_distribution(grouped_data: Dict) -> None:
    """
    Analyze and print speaker distribution across meetings.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== SPEAKER DISTRIBUTION ANALYSIS ==========================================")
    
    all_speakers = set()
    meeting_speaker_counts = []
    
    for meeting_id, samples in grouped_data.items():
        meeting_speakers = set(sample["speaker_id"] for sample in samples)
        all_speakers.update(meeting_speakers)
        meeting_speaker_counts.append(len(meeting_speakers))
        print(f"  Meeting {meeting_id}: {len(meeting_speakers)} unique speakers")
    
    print(f"\nTotal unique speakers across all meetings: {len(all_speakers)}")
    print(f"Speakers per meeting:")
    print(f"  Min: {min(meeting_speaker_counts)}")
    print(f"  Max: {max(meeting_speaker_counts)}")
    print(f"  Mean: {sum(meeting_speaker_counts)/len(meeting_speaker_counts):.1f}")
    print(f"  Median: {sorted(meeting_speaker_counts)[len(meeting_speaker_counts)//2]}")


def print_data_loading_info(dataset_name: str) -> None:
    """
    Print information about dataset loading.
    
    Args:
        dataset_name: Name of the dataset being loaded
    """
    print(f"[{datetime.now()}] STATS: ========================================== DATASET LOADING ==========================================")
    print(f"[{datetime.now()}] STATS: Loading dataset: {dataset_name}")
    print(f"[{datetime.now()}] STATS: Dataset loaded successfully")


def print_function_start(function_name: str, **kwargs) -> None:
    """
    Print function start with parameters.
    
    Args:
        function_name: Name of the function
        **kwargs: Function parameters
    """
    params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    print(f"[{datetime.now()}] STATS: STARTING FUNCTION: {function_name}({params_str})")


def print_function_end(function_name: str, result_summary: str = "") -> None:
    """
    Print function end with optional result summary.
    
    Args:
        function_name: Name of the function
        result_summary: Optional summary of results
    """
    if result_summary:
        print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: {function_name} - {result_summary}")
    else:
        print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: {function_name}")


def print_dataset_statistics(features, labels, meeting_ids, raw_features) -> None:
    """
    Print detailed dataset statistics.
    
    Args:
        features: Feature tensor
        labels: Label tensor
        meeting_ids: Meeting ID tensor
        raw_features: Raw feature list for length analysis
    """
    import numpy as np
    
    print(f"[{datetime.now()}] STATS: ========================================== DATASET STATISTICS ==========================================")
    print(f"[{datetime.now()}] STATS: STARTING FUNCTION: print_dataset_statistics")
    
    print(f"[{datetime.now()}] STATS: Dataset size (number of samples): {features.shape[0]}")
    print(f"[{datetime.now()}] STATS: Feature shape (samples, frames, mel-bands): {features.shape}")
    print(f"[{datetime.now()}] STATS: Label shape: {labels.shape}")
    print(f"[{datetime.now()}] STATS: Meeting IDs shape: {meeting_ids.shape}")
    
    # Frame size statistics - compute from padded features if raw_features not available
    if raw_features and len(raw_features) > 0:
        frame_sizes = [f.shape[0] for f in raw_features]
        print(f"[{datetime.now()}] STATS: Frame size (frames per sample): min={np.min(frame_sizes)}, max={np.max(frame_sizes)}, mean={np.mean(frame_sizes):.1f}")
        
        # Audio segment length distribution
        segment_lengths = [f.shape[0] for f in raw_features]
        print(f"[{datetime.now()}] STATS: Audio segment length distribution: min={np.min(segment_lengths)}, max={np.max(segment_lengths)}, mean={np.mean(segment_lengths):.1f}, median={np.median(segment_lengths)}")
    else:
        # Compute from padded features by finding actual lengths (non-padded parts)
        # Labels with -100 indicate padding, so we can find actual lengths
        actual_lengths = []
        for i in range(features.shape[0]):
            # Find first padding position in labels (value -100)
            label_row = labels[i]
            actual_len = np.where(label_row == -100)[0]
            if len(actual_len) > 0:
                actual_lengths.append(actual_len[0])
            else:
                actual_lengths.append(features.shape[1])  # Full length if no padding
        
        if len(actual_lengths) > 0:
            actual_lengths = np.array(actual_lengths)
            print(f"[{datetime.now()}] STATS: Frame size (frames per sample): min={np.min(actual_lengths)}, max={np.max(actual_lengths)}, mean={np.mean(actual_lengths):.1f}")
            print(f"[{datetime.now()}] STATS: Audio segment length distribution: min={np.min(actual_lengths)}, max={np.max(actual_lengths)}, mean={np.mean(actual_lengths):.1f}, median={np.median(actual_lengths)}")
        else:
            print(f"[{datetime.now()}] STATS: Frame size: Unable to compute (no valid data)")
    
    # Data types
    print(f"[{datetime.now()}] STATS: Feature dtype: {features.dtype}, Label dtype: {labels.dtype}")
    
    # Example sample analysis
    print(f"[{datetime.now()}] STATS: Example feature[0] shape: {features[0].shape}, min={features[0].min():.2f}, max={features[0].max():.2f}")
    print(f"[{datetime.now()}] STATS: Example label[0] shape: {labels[0].shape}, values: {np.unique(labels[0])}")
    print(f"[{datetime.now()}] STATS: Unique label values in dataset: {np.unique(labels)}")
    
    print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: print_dataset_statistics")


def print_data_loaders_info(train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader, batch_size: int) -> None:
    """
    Print detailed information about created data loaders and sample analysis.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        batch_size: Batch size used
    """
    print(f"[{datetime.now()}] STATS: ========================================== DATA LOADERS CREATION ==========================================")
    print(f"[{datetime.now()}] STATS: STARTING FUNCTION: print_data_loaders_info")
    
    print(f"[{datetime.now()}] STATS: Created data loaders with batch size {batch_size}")
    print(f"[{datetime.now()}] STATS: Training set: {len(train_dataset)} samples | {len(train_loader)} batches | {len(train_dataset) * train_dataset[0][0].shape[0]} frames")
    print(f"[{datetime.now()}] STATS: Validation set: {len(val_dataset)} samples | {len(val_loader)} batches | {len(val_dataset) * val_dataset[0][0].shape[0]} frames")
    print(f"[{datetime.now()}] STATS: Test set: {len(test_dataset)} samples | {len(test_loader)} batches | {len(test_dataset) * test_dataset[0][0].shape[0]} frames")

    # Example of a single sample from train_dataset
    feature, all_embeddings, label, meeting_id = train_dataset[0]
    print(f"[{datetime.now()}] STATS: === EXAMPLE TRAIN SAMPLE ===")
    print(f"[{datetime.now()}] STATS: Feature shape: {feature.shape}, dtype: {feature.dtype}")
    print(f"[{datetime.now()}] STATS: Feature (first frame): {feature[0]}")
    print(f"[{datetime.now()}] STATS: Speaker embeddings shape: {all_embeddings.shape}, dtype: {all_embeddings.dtype}")
    print(f"[{datetime.now()}] STATS: Label shape: {label.shape}, dtype: {label.dtype}")
    print(f"[{datetime.now()}] STATS: Label (first 10 frames): {label[:10]}")
    print(f"[{datetime.now()}] STATS: Meeting ID shape: {meeting_id.shape}, dtype: {meeting_id.dtype}")
    print(f"[{datetime.now()}] STATS: Meeting ID (first 10 frames): {meeting_id[:10]}")
    print(f"[{datetime.now()}] STATS: Sample = audio segment (feature matrix), batch = group of samples, frame = row in the feature matrix (one time step)")
    print(f"[{datetime.now()}] STATS: Frames are NOT independent: the model takes their sequence/context into account")
    
    print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: print_data_loaders_info")


def print_power_set_encoder_examples(power_set_encoder) -> None:
    """
    Print examples of PowerSetEncoder encoding and decoding operations.
    
    Args:
        power_set_encoder: PowerSetEncoder instance to demonstrate
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== POWER SET ENCODER EXAMPLES ==========================================")
    print(f"[{datetime.now()}] STATS: STARTING FUNCTION: print_power_set_encoder_examples")
    
    max_speakers = power_set_encoder.max_speakers
    max_overlap = power_set_encoder.max_overlap
    num_classes = power_set_encoder.num_classes
    
    print(f"[{datetime.now()}] STATS: PowerSetEncoder Configuration:")
    print(f"[{datetime.now()}] STATS:   - Max speakers: {max_speakers}")
    print(f"[{datetime.now()}] STATS:   - Max overlap: {max_overlap}")
    print(f"[{datetime.now()}] STATS:   - Number of classes: {num_classes}")
    print(f"[{datetime.now()}] STATS:   - Formula: C(K,N) = Σ(k=0 to {max_overlap}) C({max_speakers},k)")
    
    # Test cases for encoding/decoding
    test_cases = [
        [],  # No speakers
        [0],  # Single speaker
        [1],  # Single speaker (different)
        [0, 1],  # Two speakers
        [0, 2],  # Two speakers (non-consecutive)
        [1, 2],  # Two speakers
        [0, 1, 2],  # Three speakers
    ]
    
    # Add more test cases if max_overlap allows
    if max_overlap >= 4 and max_speakers >= 4:
        test_cases.append([0, 1, 2, 3])  # Four speakers
    
    print(f"\n[{datetime.now()}] STATS: Encoding/Decoding Examples:")
    print(f"[{datetime.now()}] STATS: {'Speaker IDs':<15} {'Encoded':<8} {'Decoded':<15} {'Match':<5}")
    print(f"[{datetime.now()}] STATS: {'-'*15} {'-'*8} {'-'*15} {'-'*5}")
    
    for speaker_ids in test_cases:
        if len(speaker_ids) > max_overlap:
            continue
        if max(speaker_ids) >= max_speakers if speaker_ids else False:
            continue
            
        try:
            # Encode
            encoded = power_set_encoder.encode(speaker_ids)
            
            # Decode
            decoded = power_set_encoder.decode(encoded)
            
            # Check if encoding/decoding is correct
            match = "✓" if set(speaker_ids) == set(decoded) else "✗"
            
            speaker_str = str(speaker_ids) if speaker_ids else "[]"
            decoded_str = str(decoded) if decoded else "[]"
            
            print(f"[{datetime.now()}] STATS: {speaker_str:<15} {encoded:<8} {decoded_str:<15} {match:<5}")
        except ValueError as e:
            speaker_str = str(speaker_ids) if speaker_ids else "[]"
            print(f"[{datetime.now()}] STATS: {speaker_str:<15} {'ERROR':<8} {'ERROR':<15} {'✗':<5} ({str(e)})")
    
    # Show all possible combinations
    print(f"\n[{datetime.now()}] STATS: All Possible Speaker Combinations:")
    print(f"[{datetime.now()}] STATS: {'Combination':<15} {'Encoded':<8} {'Description':<25}")
    print(f"[{datetime.now()}] STATS: {'-'*15} {'-'*8} {'-'*25}")
    
    for i in range(num_classes):
        decoded = power_set_encoder.decode(i)
        
        if not decoded:
            description = "No speakers"
        elif len(decoded) == 1:
            description = f"Speaker {decoded[0]} only"
        elif len(decoded) == max_overlap:
            description = f"Max overlap ({len(decoded)} speakers)"
        else:
            description = f"{len(decoded)} speakers: {decoded}"
        
        decoded_str = str(decoded) if decoded else "[]"
        print(f"[{datetime.now()}] STATS: {decoded_str:<15} {i:<8} {description:<25}")
    
    # Demonstrate edge cases
    print(f"\n[{datetime.now()}] STATS: Edge Cases:")
    print(f"[{datetime.now()}] STATS:   - Empty list [] encodes to 0 (no speakers active)")
    print(f"[{datetime.now()}] STATS:   - Single speaker [0] encodes to 1")
    print(f"[{datetime.now()}] STATS:   - Max overlap combination encodes to {num_classes - 1}")
    
    # Show overlap limitations
    print(f"\n[{datetime.now()}] STATS: Overlap Limitations:")
    print(f"[{datetime.now()}] STATS:   - Maximum {max_overlap} speakers can be active simultaneously")
    print(f"[{datetime.now()}] STATS:   - Total combinations: {num_classes}")
    
    # Calculate and show the formula breakdown
    from math import comb
    print(f"\n[{datetime.now()}] STATS: Formula Breakdown C(K,N) = Σ(k=0 to {max_overlap}) C({max_speakers},k):")
    total = 0
    for k in range(max_overlap + 1):
        combinations_k = comb(max_speakers, k)
        total += combinations_k
        print(f"[{datetime.now()}] STATS:   - C({max_speakers},{k}) = {combinations_k} combinations with {k} speakers")
    print(f"[{datetime.now()}] STATS:   - Total: {total} classes")
    
    # Show encoding and decoding formulas with examples
    print(f"\n[{datetime.now()}] STATS: Encoding/Decoding Formulas:")
    print(f"[{datetime.now()}] STATS: ")
    print(f"[{datetime.now()}] STATS: ENCODING: S → class_id")
    print(f"[{datetime.now()}] STATS:   1. Sort speaker IDs: S' = sorted(S)")
    print(f"[{datetime.now()}] STATS:   2. Find lexicographic index of S'")
    print(f"[{datetime.now()}] STATS: ")
    print(f"[{datetime.now()}] STATS: DECODING: class_id → S")
    print(f"[{datetime.now()}] STATS:   1. Find k: Σ(j=0 to k-1) C({max_speakers},j) ≤ class_id < Σ(j=0 to k) C({max_speakers},j)")
    print(f"[{datetime.now()}] STATS:   2. Reconstruct combination from remaining index")
    print(f"[{datetime.now()}] STATS: ")
    print(f"[{datetime.now()}] STATS: EXAMPLES WITH CALCULATIONS:")
    
    # Show calculation examples
    examples = [
        ([], "Empty set"),
        ([0], "Single speaker"),
        ([0, 1], "Two speakers"),
    ]
    
    if max_overlap >= 3 and max_speakers >= 3:
        examples.append(([0, 1, 2], "Three speakers"))
    
    for speaker_set, description in examples:
        if len(speaker_set) > max_overlap:
            continue
            
        encoded = power_set_encoder.encode(speaker_set)
        decoded = power_set_encoder.decode(encoded)
        
        print(f"[{datetime.now()}] STATS: ")
        print(f"[{datetime.now()}] STATS: Example: {speaker_set} ({description})")
        print(f"[{datetime.now()}] STATS:   Encoded: {encoded}")
        print(f"[{datetime.now()}] STATS:   Decoded: {decoded}")
        
        # Show calculation breakdown for simple cases
        print(f"[{datetime.now()}] STATS:   Calculation:")
        if not speaker_set:
            print(f"[{datetime.now()}] STATS:     Empty set → index 0")
        elif len(speaker_set) == 1:
            s = speaker_set[0]
            # Single speaker: index = C(n,0) + s = 1 + s
            offset = comb(max_speakers, 0)  # C(n,0) = 1
            print(f"[{datetime.now()}] STATS:     Speaker {s}: offset {offset} + speaker_id {s} = {offset + s}")
        elif len(speaker_set) == 2:
            s1, s2 = sorted(speaker_set)
            # Two speakers: index = C(n,0) + C(n,1) + combination_index
            offset = comb(max_speakers, 0) + comb(max_speakers, 1)  # C(n,0) + C(n,1)
            # Combination index for [s1, s2] in lexicographic order
            combo_idx = 0
            for i in range(s1):
                combo_idx += comb(max_speakers - i - 1, 2 - 1)  # C(n-i-1, 1)
            combo_idx += s2 - s1 - 1  # Position within the s1-th group
            print(f"[{datetime.now()}] STATS:     Speakers [{s1},{s2}]: offset {offset} + combo_index {combo_idx} = {offset + combo_idx}")
        elif len(speaker_set) == 3:
            s1, s2, s3 = sorted(speaker_set)
            # Three speakers: index = C(n,0) + C(n,1) + C(n,2) + combination_index
            offset = comb(max_speakers, 0) + comb(max_speakers, 1) + comb(max_speakers, 2)
            # Combination index for [s1, s2, s3] in lexicographic order
            combo_idx = 0
            for i in range(s1):
                combo_idx += comb(max_speakers - i - 1, 3 - 1)  # C(n-i-1, 2)
            for j in range(s1 + 1, s2):
                combo_idx += comb(max_speakers - j - 1, 3 - 2)  # C(n-j-1, 1)
            combo_idx += s3 - s2 - 1  # Position within the [s1,s2] group
            print(f"[{datetime.now()}] STATS:     Speakers [{s1},{s2},{s3}]: offset {offset} + combo_index {combo_idx} = {offset + combo_idx}")
    
    print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: print_power_set_encoder_examples")


def print_send_model_statistics(model) -> None:
    """
    Print basic statistics about SENDModel architecture and parameters.
    
    Args:
        model: SENDModel instance to analyze
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== SEND MODEL STATISTICS ==========================================")
    print(f"[{datetime.now()}] STATS: STARTING FUNCTION: print_send_model_statistics")
    
    # Get model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[{datetime.now()}] STATS: Model Architecture:")
    print(f"[{datetime.now()}] STATS:   - Model type: SENDModel")
    print(f"[{datetime.now()}] STATS:   - Total parameters: {total_params:,}")
    print(f"[{datetime.now()}] STATS:   - Trainable parameters: {trainable_params:,}")
    print(f"[{datetime.now()}] STATS:   - Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model configuration
    print(f"[{datetime.now()}] STATS: Model Configuration:")
    
    # Extract configuration from model structure
    input_dim = None
    hidden_dim = None
    num_classes = None
    dropout_p = None
    
    # Try to get input_dim and hidden_dim from FSMNLayer
    try:
        first_fsmn = model.speech_encoder[0][0]
        if hasattr(first_fsmn, 'input_dim'):
            input_dim = first_fsmn.input_dim
        if hasattr(first_fsmn, 'hidden_dim'):
            hidden_dim = first_fsmn.hidden_dim
    except:
        pass
    
    # Try to get num_classes from output layer
    try:
        # Get from classifier's last layer
        if hasattr(model, 'classifier'):
            last_layer = model.classifier[-1]
            if hasattr(last_layer, 'out_features'):
                num_classes = last_layer.out_features
    except:
        pass
    
    # Try to get dropout_p from dropout layers
    try:
        import torch.nn as nn
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                dropout_p = module.p
                break
    except:
        pass
    
    print(f"[{datetime.now()}] STATS:   - Input dimension: {input_dim if input_dim else 'Unknown'}")
    print(f"[{datetime.now()}] STATS:   - Hidden dimension: {hidden_dim if hidden_dim else 'Unknown'}")
    print(f"[{datetime.now()}] STATS:   - Number of classes: {num_classes if num_classes else 'Unknown'}")
    print(f"[{datetime.now()}] STATS:   - Dropout probability: {dropout_p if dropout_p else 'Unknown'}")
    
    # Layer information
    print(f"[{datetime.now()}] STATS: Layer Structure:")
    layer_count = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_count += 1
            param_count = sum(p.numel() for p in module.parameters())
            print(f"[{datetime.now()}] STATS:   - {name}: {type(module).__name__} ({param_count:,} params)")
    
    print(f"[{datetime.now()}] STATS:   - Total layers: {layer_count}")
    
    # Memory estimation
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"[{datetime.now()}] STATS: Memory Estimation:")
    print(f"[{datetime.now()}] STATS:   - Model size: ~{param_size_mb:.1f} MB")
    print(f"[{datetime.now()}] STATS:   - Training memory: ~{param_size_mb * 3:.1f} MB (including gradients and optimizer)")
    
    print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: print_send_model_statistics")


def print_client_split_statistics(client_data: List, num_clients: int, grouped_train: Dict) -> None:
    """
    Print detailed statistics about client data split for federated learning.
    
    Args:
        client_data: List containing client data split results (each element is a tuple of (train_loader, val_loader))
        num_clients: Number of clients in federated learning
        grouped_train: Original grouped training data for comparison
    """
    print(f"\n[{datetime.now()}] STATS: ========================================== CLIENT DATA SPLIT STATISTICS ==========================================")
    print(f"[{datetime.now()}] STATS: STARTING FUNCTION: print_client_split_statistics")
    
    if not client_data:
        print(f"[{datetime.now()}] STATS: ERROR: No client data provided")
        return
    
    # Basic split information
    print(f"[{datetime.now()}] STATS: Federated Learning Configuration:")
    print(f"[{datetime.now()}] STATS:   - Number of clients: {num_clients}")
    print(f"[{datetime.now()}] STATS:   - Split strategy: meeting_based")
    print(f"[{datetime.now()}] STATS:   - Total meetings in training set: {len(grouped_train)}")
    print(f"[{datetime.now()}] STATS:   - Actual clients created: {len(client_data)}")
    
    # Client-specific statistics
    print(f"\n[{datetime.now()}] STATS: Client Data Distribution:")
    print(f"[{datetime.now()}] STATS: {'Client':<10} {'Train Batches':<15} {'Val Batches':<12} {'Train Samples':<15} {'Val Samples':<12}")
    print(f"[{datetime.now()}] STATS: {'-'*10} {'-'*15} {'-'*12} {'-'*15} {'-'*12}")
    
    total_train_samples = 0
    total_val_samples = 0
    clients_with_data = 0
    
    for client_id, (train_loader, val_loader) in enumerate(client_data):
        if train_loader is None or val_loader is None:
            print(f"[{datetime.now()}] STATS: {client_id:<10} {'MISSING':<15} {'MISSING':<12} {'MISSING':<15} {'MISSING':<12}")
            continue
        
        # Extract information from data loaders
        train_batches = len(train_loader)
        val_batches = len(val_loader)
        
        # Calculate approximate number of samples (assuming batch_size=4)
        train_samples = train_batches * 4  # Approximate, actual batch size may vary
        val_samples = val_batches * 4      # Approximate, actual batch size may vary
        
        total_train_samples += train_samples
        total_val_samples += val_samples
        
        if train_batches > 0 or val_batches > 0:
            clients_with_data += 1
        
        print(f"[{datetime.now()}] STATS: {client_id:<10} {train_batches:<15} {val_batches:<12} {train_samples:<15} {val_samples:<12}")
    
    # Summary statistics
    print(f"\n[{datetime.now()}] STATS: Split Summary:")
    print(f"[{datetime.now()}] STATS:   - Total train samples (approx): {total_train_samples:,}")
    print(f"[{datetime.now()}] STATS:   - Total validation samples (approx): {total_val_samples:,}")
    print(f"[{datetime.now()}] STATS:   - Total samples (approx): {total_train_samples + total_val_samples:,}")
    print(f"[{datetime.now()}] STATS:   - Clients with data: {clients_with_data}/{len(client_data)}")
    
    # Balance analysis
    if len(client_data) > 0:
        client_train_samples = []
        client_val_samples = []
        client_total_samples = []
        
        for train_loader, val_loader in client_data:
            if train_loader is not None and val_loader is not None:
                train_samples = len(train_loader) * 4  # Approximate
                val_samples = len(val_loader) * 4      # Approximate
                total_samples = train_samples + val_samples
                
                client_train_samples.append(train_samples)
                client_val_samples.append(val_samples)
                client_total_samples.append(total_samples)
        
        if client_total_samples:
            import numpy as np
            print(f"\n[{datetime.now()}] STATS: Balance Analysis:")
            print(f"[{datetime.now()}] STATS:   - Train samples per client: min={min(client_train_samples)}, max={max(client_train_samples)}, mean={np.mean(client_train_samples):.1f}, std={np.std(client_train_samples):.1f}")
            print(f"[{datetime.now()}] STATS:   - Val samples per client: min={min(client_val_samples)}, max={max(client_val_samples)}, mean={np.mean(client_val_samples):.1f}, std={np.std(client_val_samples):.1f}")
            print(f"[{datetime.now()}] STATS:   - Total samples per client: min={min(client_total_samples)}, max={max(client_total_samples)}, mean={np.mean(client_total_samples):.1f}, std={np.std(client_total_samples):.1f}")
            
            # Coefficient of variation (lower is better for balance)
            cv_train = np.std(client_train_samples) / np.mean(client_train_samples) if np.mean(client_train_samples) > 0 else 0
            cv_val = np.std(client_val_samples) / np.mean(client_val_samples) if np.mean(client_val_samples) > 0 else 0
            cv_total = np.std(client_total_samples) / np.mean(client_total_samples) if np.mean(client_total_samples) > 0 else 0
            
            print(f"[{datetime.now()}] STATS:   - Balance coefficients (lower=better): train={cv_train:.3f}, val={cv_val:.3f}, total={cv_total:.3f}")
    
    # Data loader analysis
    print(f"\n[{datetime.now()}] STATS: Data Loader Analysis:")
    print(f"[{datetime.now()}] STATS:   - Each client has train_loader and val_loader")
    print(f"[{datetime.now()}] STATS:   - Batch size: 4 (approximate)")
    print(f"[{datetime.now()}] STATS:   - Data loaders include collate function for padding")
    print(f"[{datetime.now()}] STATS:   - Features are padded to max sequence length in batch")
    
    # Data quality checks
    print(f"\n[{datetime.now()}] STATS: Data Quality Checks:")
    empty_clients = 0
    for train_loader, val_loader in client_data:
        if train_loader is None or val_loader is None:
            empty_clients += 1
        elif len(train_loader) == 0 and len(val_loader) == 0:
            empty_clients += 1
    
    print(f"[{datetime.now()}] STATS:   - Empty clients: {empty_clients}/{len(client_data)}")
    print(f"[{datetime.now()}] STATS:   - Clients with data: {len(client_data) - empty_clients}/{len(client_data)}")
    
    if empty_clients > 0:
        print(f"[{datetime.now()}] STATS:   - WARNING: Some clients have no data!")
    
    # Training readiness check
    print(f"\n[{datetime.now()}] STATS: Training Readiness:")
    ready_clients = 0
    for client_id, (train_loader, val_loader) in enumerate(client_data):
        if train_loader is not None and val_loader is not None:
            if len(train_loader) > 0 and len(val_loader) > 0:
                ready_clients += 1
                print(f"[{datetime.now()}] STATS:   - Client {client_id}: Ready for training ({len(train_loader)} train batches, {len(val_loader)} val batches)")
            else:
                print(f"[{datetime.now()}] STATS:   - Client {client_id}: Not ready (empty loaders)")
        else:
            print(f"[{datetime.now()}] STATS:   - Client {client_id}: Not ready (missing loaders)")
    
    print(f"[{datetime.now()}] STATS:   - Clients ready for training: {ready_clients}/{len(client_data)}")
    
    # Recommendations
    print(f"\n[{datetime.now()}] STATS: Recommendations:")
    if 'cv_total' in locals() and cv_total > 0.3:
        print(f"[{datetime.now()}] STATS:   - Consider rebalancing: high sample count variation (CV={cv_total:.3f})")
    if empty_clients > 0:
        print(f"[{datetime.now()}] STATS:   - Fix empty clients before training")
    if ready_clients < len(client_data):
        print(f"[{datetime.now()}] STATS:   - Ensure all clients have valid data loaders before federated training")
    
    print(f"[{datetime.now()}] STATS: COMPLETED FUNCTION: print_client_split_statistics")
