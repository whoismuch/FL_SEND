#!/usr/bin/env python3
"""
Test script to verify that the DER calculation fix works correctly for overlapping speakers.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import calculate_der
from FL_SEND_PSE_AMI_improved import PowerSetEncoder
from pyannote.core import Segment, Annotation

def test_overlapping_speakers():
    """Test DER calculation with overlapping speakers."""
    print("Testing DER calculation with overlapping speakers...")
    
    # Create a simple power set encoder for 3 speakers
    power_set_encoder = PowerSetEncoder(max_speakers=3)
    speaker_id_list = [0, 1, 2]
    
    # Test case: overlapping speakers
    # Frame 0: speaker 0 only (encoded as 1 = 001)
    # Frame 1: speakers 0 and 1 (encoded as 3 = 011) 
    # Frame 2: speaker 2 only (encoded as 4 = 100)
    # Frame 3: speakers 1 and 2 (encoded as 6 = 110)
    
    predictions = [1, 3, 4, 6]  # Power set encoded predictions
    labels = [1, 3, 4, 6]       # Perfect predictions for testing
    
    print(f"Predictions: {predictions}")
    print(f"Labels: {labels}")
    print(f"Speaker ID list: {speaker_id_list}")
    
    # Calculate DER
    der = calculate_der(predictions, labels, power_set_encoder, speaker_id_list, debug=True)
    
    print(f"\nCalculated DER: {der}")
    
    # For perfect predictions, DER should be 0.0
    if der == 0.0:
        print("✓ Test PASSED: Perfect predictions result in DER = 0.0")
    else:
        print(f"✗ Test FAILED: Expected DER = 0.0, got {der}")
    
    # Test with some errors
    print("\n" + "="*50)
    print("Testing with prediction errors...")
    
    # Introduce some errors
    predictions_with_errors = [1, 2, 4, 5]  # Some wrong predictions
    der_with_errors = calculate_der(predictions_with_errors, labels, power_set_encoder, speaker_id_list, debug=True)
    
    print(f"DER with errors: {der_with_errors}")
    
    if der_with_errors > 0.0:
        print("✓ Test PASSED: Prediction errors result in DER > 0.0")
    else:
        print("✗ Test FAILED: Expected DER > 0.0 for prediction errors")

def test_annotation_structure():
    """Test that annotations are created correctly with separate segments."""
    print("\n" + "="*50)
    print("Testing annotation structure...")
    
    power_set_encoder = PowerSetEncoder(max_speakers=3)
    speaker_id_list = [0, 1, 2]
    
    # Test with overlapping speakers
    predictions = [3]  # speakers 0 and 1 (encoded as 3 = 011)
    labels = [3]
    
    # Create annotations manually to inspect structure
    from pyannote.core import Segment, Annotation
    
    reference = Annotation()
    hypothesis = Annotation()
    
    # Decode the power set encoded value
    true_indices = set(power_set_encoder.decode(labels[0]))
    pred_indices = set(power_set_encoder.decode(predictions[0]))
    
    print(f"True indices: {true_indices}")
    print(f"Pred indices: {pred_indices}")
    
    # Create time segment
    t0, t1 = 0, 1
    
    # Add separate tracks for each active speaker (correct approach)
    for track_idx, idx in enumerate(true_indices):
        speaker_name = f"speaker_{speaker_id_list[idx]}"
        reference[Segment(t0, t1), track_idx] = speaker_name
        print(f"Added to reference: {speaker_name} on [{t0}, {t1}] track {track_idx}")
    
    for track_idx, idx in enumerate(pred_indices):
        speaker_name = f"speaker_{speaker_id_list[idx]}"
        hypothesis[Segment(t0, t1), track_idx] = speaker_name
        print(f"Added to hypothesis: {speaker_name} on [{t0}, {t1}] track {track_idx}")
    
    print(f"\nReference annotation:")
    for segment, track, label in reference.itertracks(yield_label=True):
        print(f"  {segment}: {label}")
    
    print(f"\nHypothesis annotation:")
    for segment, track, label in hypothesis.itertracks(yield_label=True):
        print(f"  {segment}: {label}")
    
    # Verify that we have separate segments for each speaker
    ref_speakers = set(label for segment, track, label in reference.itertracks(yield_label=True))
    hyp_speakers = set(label for segment, track, label in hypothesis.itertracks(yield_label=True))
    
    print(f"\nReference speakers: {ref_speakers}")
    print(f"Hypothesis speakers: {hyp_speakers}")
    
    expected_speakers = {"speaker_0", "speaker_1"}
    if ref_speakers == expected_speakers and hyp_speakers == expected_speakers:
        print("✓ Test PASSED: Annotations correctly contain separate segments for each speaker")
    else:
        print(f"✗ Test FAILED: Expected {expected_speakers}, got ref={ref_speakers}, hyp={hyp_speakers}")

def test_old_vs_new_approach():
    """Demonstrate the difference between old (incorrect) and new (correct) approaches."""
    print("\n" + "="*50)
    print("Demonstrating old vs new annotation approaches...")
    
    # Create annotations using both approaches
    old_reference = Annotation()
    new_reference = Annotation()
    
    # Test case: speakers 0 and 1 overlapping
    true_indices = {0, 1}
    speaker_id_list = [0, 1, 2]
    t0, t1 = 0, 1
    
    # OLD APPROACH (incorrect) - using frozenset
    print("OLD APPROACH (incorrect):")
    ref_speakers = frozenset(f"speaker_{speaker_id_list[idx]}" for idx in true_indices)
    old_reference[Segment(t0, t1)] = ref_speakers
    print(f"  Added: {ref_speakers} on [{t0}, {t1}]")
    
    # NEW APPROACH (correct) - using tracks
    print("NEW APPROACH (correct):")
    for track_idx, idx in enumerate(true_indices):
        speaker_name = f"speaker_{speaker_id_list[idx]}"
        new_reference[Segment(t0, t1), track_idx] = speaker_name
        print(f"  Added: {speaker_name} on [{t0}, {t1}] track {track_idx}")
    
    print(f"\nOld approach result:")
    for segment, track, label in old_reference.itertracks(yield_label=True):
        print(f"  {segment}: {label} (type: {type(label)})")
    
    print(f"\nNew approach result:")
    for segment, track, label in new_reference.itertracks(yield_label=True):
        print(f"  {segment}: {label} (type: {type(label)})")
    
    print(f"\nOld approach speakers: {set(label for segment, track, label in old_reference.itertracks(yield_label=True))}")
    print(f"New approach speakers: {set(label for segment, track, label in new_reference.itertracks(yield_label=True))}")

if __name__ == "__main__":
    test_overlapping_speakers()
    test_annotation_structure()
    test_old_vs_new_approach()
    print("\n" + "="*50)
    print("All tests completed!")
