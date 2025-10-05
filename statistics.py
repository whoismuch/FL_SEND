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
    print(f"\n[{datetime.now()}] MAIN: ========================================== MEETING STATISTICS ==========================================")
    
    # Training set statistics
    print(f"\n[{datetime.now()}] MAIN: Training set meetings:")
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
    print(f"\n[{datetime.now()}] MAIN: Validation set meetings:")
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
    print(f"\n[{datetime.now()}] MAIN: Test set meetings:")
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
    print(f"\n[{datetime.now()}] MAIN: ========================================== SUMMARY STATISTICS ==========================================")
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
    print(f"[{datetime.now()}] MAIN: Dataset: {dataset_name}")
    print(f"[{datetime.now()}] MAIN: Total samples: {total_samples}")
    print(f"[{datetime.now()}] MAIN: Using subset of {test_size} samples for testing")


def print_grouping_results(grouped_train: Dict, grouped_validation: Dict, grouped_test: Dict) -> None:
    """
    Print results of data grouping by meeting ID.
    
    Args:
        grouped_train: Dictionary of meeting_id to samples for training set
        grouped_validation: Dictionary of meeting_id to samples for validation set
        grouped_test: Dictionary of meeting_id to samples for test set
    """
    print(f"[{datetime.now()}] MAIN: ========================================== DATA GROUPING ==========================================")
    print(f"[{datetime.now()}] MAIN: Grouped {len(grouped_train)} meetings from training set")
    print(f"[{datetime.now()}] MAIN: Grouped {len(grouped_validation)} meetings from validation set")
    print(f"[{datetime.now()}] MAIN: Grouped {len(grouped_test)} meetings from test set")


def print_experiment_config(num_clients: int, num_rounds: int, num_epochs: int, test_size: int) -> None:
    """
    Print experiment configuration parameters.
    
    Args:
        num_clients: Number of clients in federated learning
        num_rounds: Number of communication rounds
        num_epochs: Number of training epochs per round
        test_size: Number of samples used for testing
    """
    print(f"\n[{datetime.now()}] MAIN: ========================================== EXPERIMENT CONFIGURATION ==========================================")
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
    print(f"\n[{datetime.now()}] MAIN: ========================================== FINAL RESULTS ==========================================")
    print(f"Final DER: {final_der:.4f}")
    print(f"Best DER: {best_der:.4f}")
    print(f"Total experiment time: {total_time:.2f} seconds")


def analyze_speaker_distribution(grouped_data: Dict) -> None:
    """
    Analyze and print speaker distribution across meetings.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
    """
    print(f"\n[{datetime.now()}] MAIN: ========================================== SPEAKER DISTRIBUTION ANALYSIS ==========================================")
    
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
    print(f"[{datetime.now()}] MAIN: ========================================== DATASET LOADING ==========================================")
    print(f"[{datetime.now()}] MAIN: Loading dataset: {dataset_name}")
    print(f"[{datetime.now()}] MAIN: Dataset loaded successfully")
