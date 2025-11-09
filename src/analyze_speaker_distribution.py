#!/usr/bin/env python3
"""
Script for analyzing speaker distribution in a dataset.

Shows:
- How many meetings have N speakers (total number of unique speakers)
- Maximum number of simultaneously active speakers (K)
- Overlap distribution
- Recommendations for choosing N and K
"""

import sys
import os
from collections import Counter
from datasets import load_dataset
from datetime import datetime

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import group_by_meeting


class TeeOutput:
    """Class to write output to both console and file."""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def find_max_concurrent_speakers(samples):
    """
    Finds the maximum number of simultaneously active speakers in a meeting.
    
    Args:
        samples: List of audio samples for a meeting
        
    Returns:
        int: Maximum number of simultaneously active speakers
    """
    if not samples:
        return 0
    
    # Create events for start and end of each speaker's speech
    events = []
    for sample in samples:
        events.append((sample["begin_time"], "start", sample["speaker_id"]))
        events.append((sample["end_time"], "end", sample["speaker_id"]))
    
    # Sort events by time
    events.sort(key=lambda x: x[0])
    
    # Track active speakers
    active_speakers = set()
    max_concurrent = 0
    
    for time, event_type, speaker_id in events:
        if event_type == "start":
            active_speakers.add(speaker_id)
        else:  # "end"
            active_speakers.discard(speaker_id)
        
        max_concurrent = max(max_concurrent, len(active_speakers))
    
    return max_concurrent


def analyze_speaker_distribution(dataset_name="edinburghcstr/ami", analyze_all_splits=True, max_meetings=None, output_dir="logs"):
    """
    Analyzes speaker distribution in a dataset.
    
    Args:
        dataset_name: Dataset name
        analyze_all_splits: If True, analyze all splits (train, validation, test). If False, analyze only train.
        max_meetings: Maximum number of meetings to analyze per split (None = all)
        output_dir: Directory to save output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"speaker_distribution_analysis_{timestamp}.txt")
    
    # Redirect output to both console and file
    tee = TeeOutput(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("="*80)
        print("📊 SPEAKER DISTRIBUTION ANALYSIS")
        print("="*80)
        print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output file: {output_file}\n")
    
        # Load dataset
        print(f"\n📥 Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, "ihm")
        
        # Determine which splits to analyze
        if analyze_all_splits:
            splits_to_analyze = ['train', 'validation', 'test']
            print(f"   Analyzing all splits: {', '.join(splits_to_analyze)}")
        else:
            splits_to_analyze = ['train']
            print(f"   Analyzing only train split")
        
        # Collect data from all splits
        all_grouped_data = {}
        split_info = {}
        
        for split in splits_to_analyze:
            if split not in dataset:
                print(f"   ⚠️  Warning: Split '{split}' not found in dataset, skipping...")
                continue
                
            split_data = dataset[split]
            
            if max_meetings:
                split_data = split_data.select(range(min(max_meetings, len(split_data))))
                print(f"   {split}: Limited to {len(split_data)} samples")
            
            # Group by meetings
            grouped_data = group_by_meeting(split_data)
            all_grouped_data[split] = grouped_data
            split_info[split] = {
                'num_samples': len(split_data),
                'num_meetings': len(grouped_data)
            }
            print(f"   {split}: {len(grouped_data)} meetings from {len(split_data)} samples")
        
        # Combine all meetings for overall analysis
        print(f"\n🏢 Combining all splits for overall analysis...")
        combined_grouped_data = {}
        for split, grouped_data in all_grouped_data.items():
            combined_grouped_data.update(grouped_data)
        
        total_meetings_all = len(combined_grouped_data)
        print(f"   Total unique meetings across all splits: {total_meetings_all}")
        
        # Use combined data for main analysis
        grouped_data = combined_grouped_data
        
        # Analyze per-split statistics first
        if analyze_all_splits and len(splits_to_analyze) > 1:
            print(f"\n📊 PER-SPLIT STATISTICS:")
            for split in splits_to_analyze:
                if split not in all_grouped_data:
                    continue
                split_grouped = all_grouped_data[split]
                split_speaker_counts = []
                split_concurrent_counts = []
                
                for meeting_id, samples in split_grouped.items():
                    speakers = set(sample["speaker_id"] for sample in samples)
                    split_speaker_counts.append(len(speakers))
                    split_concurrent_counts.append(find_max_concurrent_speakers(samples))
                
                print(f"\n   {split.upper()} split:")
                print(f"      Meetings: {len(split_grouped)}")
                if split_speaker_counts:
                    print(f"      Speakers: min={min(split_speaker_counts)}, max={max(split_speaker_counts)}, "
                          f"avg={sum(split_speaker_counts)/len(split_speaker_counts):.2f}")
                if split_concurrent_counts:
                    print(f"      Max overlap: min={min(split_concurrent_counts)}, max={max(split_concurrent_counts)}, "
                          f"avg={sum(split_concurrent_counts)/len(split_concurrent_counts):.2f}")
        
        # Analyze number of speakers in each meeting (OVERALL)
        print(f"\n📊 OVERALL SPEAKER COUNT ANALYSIS (All Splits Combined):")
        speaker_counts = []
        speaker_count_distribution = Counter()
        
        for meeting_id, samples in grouped_data.items():
            speakers = set(sample["speaker_id"] for sample in samples)
            num_speakers = len(speakers)
            speaker_counts.append(num_speakers)
            speaker_count_distribution[num_speakers] += 1
        
        # Statistics
        print(f"\n   Overall Statistics:")
        print(f"   - Minimum speakers: {min(speaker_counts)}")
        print(f"   - Maximum speakers: {max(speaker_counts)}")
        print(f"   - Average number of speakers: {sum(speaker_counts) / len(speaker_counts):.2f}")
        print(f"   - Median number of speakers: {sorted(speaker_counts)[len(speaker_counts) // 2]}")
        
        # Distribution
        print(f"\n   Distribution by number of speakers:")
        print(f"   {'# Speakers':<20} {'# Meetings':<20} {'Percentage':<20}")
        print(f"   {'-'*60}")
        
        total_meetings = len(grouped_data)
        for num_speakers in sorted(speaker_count_distribution.keys()):
            count = speaker_count_distribution[num_speakers]
            percentage = 100 * count / total_meetings
            print(f"   {num_speakers:<20} {count:<20} {percentage:>6.2f}%")
        
        # Visualization
        print(f"\n   Visualization (each '#' = 1%):")
        for num_speakers in sorted(speaker_count_distribution.keys()):
            count = speaker_count_distribution[num_speakers]
            percentage = 100 * count / total_meetings
            bar = '#' * int(percentage)
            print(f"   {num_speakers} speakers: {bar} {percentage:.1f}%")
    
        # Analyze maximum number of simultaneously active speakers (K) - OVERALL
        print(f"\n📊 OVERALL MAXIMUM OVERLAP ANALYSIS (K) - All Splits Combined:")
        concurrent_counts = []
        concurrent_distribution = Counter()
        
        for meeting_id, samples in grouped_data.items():
            max_concurrent = find_max_concurrent_speakers(samples)
            concurrent_counts.append(max_concurrent)
            concurrent_distribution[max_concurrent] += 1
        
        # Statistics on overlaps
        print(f"\n   Overall Maximum Overlap Statistics:")
        print(f"   - Minimum simultaneously active speakers: {min(concurrent_counts)}")
        print(f"   - Maximum simultaneously active speakers: {max(concurrent_counts)}")
        print(f"   - Average maximum overlap: {sum(concurrent_counts) / len(concurrent_counts):.2f}")
        print(f"   - Median maximum overlap: {sorted(concurrent_counts)[len(concurrent_counts) // 2]}")
        
        # Distribution by overlaps
        print(f"\n   Distribution by maximum overlap:")
        print(f"   {'Max Overlap':<20} {'# Meetings':<20} {'Percentage':<20}")
        print(f"   {'-'*60}")
        
        for max_concurrent in sorted(concurrent_distribution.keys()):
            count = concurrent_distribution[max_concurrent]
            percentage = 100 * count / total_meetings
            print(f"   {max_concurrent:<20} {count:<20} {percentage:>6.2f}%")
        
        # Visualization of overlaps
        print(f"\n   Overlap visualization (each '#' = 1%):")
        for max_concurrent in sorted(concurrent_distribution.keys()):
            count = concurrent_distribution[max_concurrent]
            percentage = 100 * count / total_meetings
            bar = '#' * int(percentage)
            print(f"   {max_concurrent} simultaneously: {bar} {percentage:.1f}%")
        
        max_concurrent_overall = max(concurrent_counts)
    
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        max_speakers = max(speaker_counts)
        
        print(f"\n   1. Current limitation N=4:")
        meetings_with_4_or_less = sum(count for n, count in speaker_count_distribution.items() if n <= 4)
        meetings_with_more_than_4 = total_meetings - meetings_with_4_or_less
        percentage_covered = 100 * meetings_with_4_or_less / total_meetings
        percentage_lost = 100 * meetings_with_more_than_4 / total_meetings
        
        print(f"      - Covers {meetings_with_4_or_less}/{total_meetings} meetings ({percentage_covered:.1f}%)")
        print(f"      - Loses information about {meetings_with_more_than_4} meetings ({percentage_lost:.1f}%)")
        
        if meetings_with_more_than_4 > 0:
            print(f"      ⚠️  WARNING: {meetings_with_more_than_4} meetings have >4 speakers!")
            print(f"         These meetings will be processed with information loss.")
            if analyze_all_splits:
                # Check which splits have meetings with >4 speakers
                for split in splits_to_analyze:
                    if split not in all_grouped_data:
                        continue
                    split_meetings_with_more_than_4 = sum(
                        1 for samples in all_grouped_data[split].values()
                        if len(set(s["speaker_id"] for s in samples)) > 4
                    )
                    if split_meetings_with_more_than_4 > 0:
                        print(f"         - {split} split: {split_meetings_with_more_than_4} meetings with >4 speakers")
        
        print(f"\n   2. Recommended value for N (based on ALL splits):")
        print(f"      - Minimum: N = {max_speakers} (covers all meetings across all splits)")
        print(f"      - This ensures coverage for train, validation, AND test sets")
        
        print(f"\n   3. Recommended value for K (based on ALL splits):")
        print(f"      - Minimum: K = {max_concurrent_overall} (covers all overlaps across all splits)")
        print(f"      - This ensures coverage for train, validation, AND test sets")
        
        # Coverage analysis for different K values
        print(f"\n   4. Coverage for different K values:")
        print(f"      {'K':<10} {'Meetings Covered':<20} {'Percentage':<20}")
        print(f"      {'-'*50}")
        
        for k in range(1, max_concurrent_overall + 2):
            meetings_covered = sum(count for max_conc, count in concurrent_distribution.items() if max_conc <= k)
            percentage_covered = 100 * meetings_covered / total_meetings
            marker = " ← Recommended" if k == max_concurrent_overall else ""
            print(f"      {k:<10} {meetings_covered}/{total_meetings:<15} {percentage_covered:>6.2f}%{marker}")
    
        # Calculate number of classes for different (N, K) combinations
        print(f"\n   5. Number of classes for different (N, K) combinations:")
        print(f"      {'N':<6} {'K':<6} {'Classes':<15} {'Note':<30}")
        print(f"      {'-'*60}")
        
        from math import comb
        
        # Show recommended combination
        recommended_n = max_speakers
        recommended_k = max_concurrent_overall
        recommended_classes = sum(comb(recommended_n, k) for k in range(recommended_k + 1))
        print(f"      {recommended_n:<6} {recommended_k:<6} {recommended_classes:<15} {'← Recommended (full coverage)':<30}")
        
        # Show alternative options
        for n in range(max(1, recommended_n - 1), min(recommended_n + 2, 9)):
            for k in range(max(1, recommended_k - 1), min(n + 1, recommended_k + 2)):
                if n == recommended_n and k == recommended_k:
                    continue  # Already shown
                num_classes = sum(comb(n, k_val) for k_val in range(k + 1))
                note = ""
                if k < recommended_k:
                    meetings_lost = sum(count for max_conc, count in concurrent_distribution.items() if max_conc > k)
                    note = f"Loses {meetings_lost} meetings"
                print(f"      {n:<6} {k:<6} {num_classes:<15} {note:<30}")
    
        # Detailed information about meetings with >4 speakers
        if meetings_with_more_than_4 > 0:
            print(f"\n   6. Meetings with >4 speakers:")
            for meeting_id, samples in grouped_data.items():
                speakers = set(sample["speaker_id"] for sample in samples)
                num_speakers = len(speakers)
                if num_speakers > 4:
                    max_conc = find_max_concurrent_speakers(samples)
                    print(f"      - {meeting_id}: {num_speakers} speakers, max overlap: {max_conc}")
        
        # Detailed information about meetings with high overlap
        meetings_with_high_overlap = [mid for mid, samples in grouped_data.items() 
                                       if find_max_concurrent_speakers(samples) > 3]
        if meetings_with_high_overlap:
            print(f"\n   7. Meetings with overlap >3:")
            for meeting_id in meetings_with_high_overlap:
                samples = grouped_data[meeting_id]
                speakers = set(sample["speaker_id"] for sample in samples)
                num_speakers = len(speakers)
                max_conc = find_max_concurrent_speakers(samples)
                print(f"      - {meeting_id}: {num_speakers} speakers, max overlap: {max_conc}")
    
        # Final recommendation
        print(f"\n✅ FINAL RECOMMENDATION:")
        from math import comb
        
        recommended_num_classes = sum(comb(recommended_n, k) for k in range(recommended_k + 1))
        
        print(f"\n   Recommended parameters:")
        print(f"   - N = {recommended_n} (maximum number of speakers per meeting)")
        print(f"   - K = {recommended_k} (maximum overlap)")
        print(f"   - Number of classes = {recommended_num_classes}")
        
        if max_speakers <= 4 and max_concurrent_overall <= 3:
            print(f"\n   Current values N=4, K=3 are suitable for all meetings.")
            print(f"   No need to change parameters.")
        else:
            print(f"\n   This will require:")
            print(f"   - Recreate PowerSetEncoder with max_speakers={recommended_n}, max_overlap={recommended_k}")
            print(f"   - Recreate model with num_classes={recommended_num_classes}")
            print(f"   - Retrain the model")
            print(f"   - This will cover 100% of meetings without information loss")
        
        # Alternative options
        if recommended_num_classes > 64:
            print(f"\n   ⚠️  WARNING: Number of classes ({recommended_num_classes}) > 64!")
            print(f"   Consider alternative options:")
            
            # Option 1: Reduce K
            if recommended_k > 2:
                alt_k = 2
                alt_classes = sum(comb(recommended_n, k) for k in range(alt_k + 1))
                meetings_lost = sum(count for max_conc, count in concurrent_distribution.items() if max_conc > alt_k)
                print(f"   - Option 1: N={recommended_n}, K={alt_k} → {alt_classes} classes")
                print(f"     Loses {meetings_lost} meetings with overlap >{alt_k}")
            
            # Option 2: Reduce N (if possible)
            if recommended_n > 4:
                alt_n = 4
                alt_k = min(alt_n, recommended_k)
                alt_classes = sum(comb(alt_n, k) for k in range(alt_k + 1))
                meetings_lost = sum(count for n, count in speaker_count_distribution.items() if n > alt_n)
                print(f"   - Option 2: N={alt_n}, K={alt_k} → {alt_classes} classes")
                print(f"     Loses {meetings_lost} meetings with >{alt_n} speakers")
        
        print(f"\n{'='*80}")
        print(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*80}\n")
        
        result = {
            'max_speakers': max_speakers,
            'max_concurrent': max_concurrent_overall,
            'speaker_distribution': dict(speaker_count_distribution),
            'concurrent_distribution': dict(concurrent_distribution),
            'total_meetings': total_meetings,
            'meetings_with_more_than_4': meetings_with_more_than_4,
            'recommended_n': recommended_n,
            'recommended_k': recommended_k,
            'recommended_num_classes': recommended_num_classes,
            'output_file': output_file
        }
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        tee.close()
        print(f"Results saved to: {output_file}")
    
    return result


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze speaker distribution in dataset')
    parser.add_argument('--dataset', type=str, default='edinburghcstr/ami',
                        help='Dataset name (default: edinburghcstr/ami)')
    parser.add_argument('--train-only', action='store_true',
                        help='Analyze only train split (default: analyze all splits)')
    parser.add_argument('--max-meetings', type=int, default=None,
                        help='Maximum number of meetings to analyze per split (default: all)')
    parser.add_argument('--output-dir', type=str, default='statistics',
                        help='Output directory for results file (default: statistics)')
    
    args = parser.parse_args()
    
    analyze_speaker_distribution(
        dataset_name=args.dataset,
        analyze_all_splits=not args.train_only,
        max_meetings=args.max_meetings,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

