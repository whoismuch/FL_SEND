#!/usr/bin/env python3
"""
Test script to demonstrate the fix for inconsistent speaker_id_list length.
This test shows the problem that ChatGPT identified and how it's been fixed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import calculate_der
from FL_SEND_PSE_AMI_improved import PowerSetEncoder
from pyannote.core import Segment, Annotation

def test_speaker_id_list_consistency():
    """Test that speaker_id_list is now consistent across frames."""
    print("Testing speaker_id_list consistency fix...")
    
    # Create a power set encoder for 4 speakers
    power_set_encoder = PowerSetEncoder(max_speakers=4)
    
    # Test case with varying number of active speakers per frame
    # Frame 0: only speaker 0 (encoded as 1 = 0001)
    # Frame 1: speakers 0 and 1 (encoded as 3 = 0011) 
    # Frame 2: speakers 1, 2, 3 (encoded as 14 = 1110)
    # Frame 3: only speaker 3 (encoded as 8 = 1000)
    
    predictions = [1, 3, 14, 8]  # Power set encoded predictions
    labels = [1, 3, 14, 8]       # Perfect predictions for testing
    
    print(f"Predictions: {predictions}")
    print(f"Labels: {labels}")
    print(f"Power set encoder max_speakers: {power_set_encoder.max_speakers}")
    
    # Test with speaker_id_list=None (should use max_speakers)
    print("\n--- Testing with speaker_id_list=None ---")
    der1 = calculate_der(predictions, labels, power_set_encoder, speaker_id_list=None, debug=True)
    print(f"DER with speaker_id_list=None: {der1}")
    
    # Test with explicit speaker_id_list
    print("\n--- Testing with explicit speaker_id_list=[0, 1, 2, 3] ---")
    speaker_id_list = [0, 1, 2, 3]
    der2 = calculate_der(predictions, labels, power_set_encoder, speaker_id_list=speaker_id_list, debug=True)
    print(f"DER with explicit speaker_id_list: {der2}")
    
    # Both should give the same result
    if der1 == der2 == 0.0:
        print("✓ Test PASSED: Consistent speaker_id_list handling")
    else:
        print(f"✗ Test FAILED: Inconsistent results - der1={der1}, der2={der2}")

def test_edge_cases():
    """Test edge cases that could cause problems."""
    print("\n" + "="*50)
    print("Testing edge cases...")
    
    power_set_encoder = PowerSetEncoder(max_speakers=3)
    
    # Test case: some frames have no active speakers (should be skipped)
    predictions = [0, 1, 0, 3]  # 0 means no speakers active
    labels = [0, 1, 0, 3]
    
    print(f"Predictions with zeros: {predictions}")
    print(f"Labels with zeros: {labels}")
    
    der = calculate_der(predictions, labels, power_set_encoder, debug=True)
    print(f"DER with zero frames: {der}")
    
    if der == 0.0:
        print("✓ Test PASSED: Zero frames handled correctly")
    else:
        print(f"✗ Test FAILED: Zero frames not handled correctly")

def demonstrate_problem_fix():
    """Demonstrate what the problem was and how it's fixed."""
    print("\n" + "="*50)
    print("Demonstrating the problem and fix...")
    
    power_set_encoder = PowerSetEncoder(max_speakers=4)
    
    # Simulate the old problematic behavior
    print("OLD PROBLEMATIC APPROACH (what ChatGPT identified):")
    print("- speaker_id_list was created from actual speaker IDs in data")
    print("- This could be ['speaker_1', 'speaker_5', 'speaker_12']")
    print("- But power set encoding uses indices 0, 1, 2, 3...")
    print("- This created a mismatch between indices and speaker names")
    
    print("\nNEW FIXED APPROACH:")
    print("- speaker_id_list=None triggers automatic creation")
    print("- Uses power_set_encoder.max_speakers as upper bound")
    print("- Speaker names are created as f'speaker_{idx}' where idx is the power set index")
    print("- This ensures consistency between power set indices and speaker names")
    
    # Test the fix
    predictions = [1, 3, 5, 15]  # Various combinations
    labels = [1, 3, 5, 15]
    
    print(f"\nTesting with predictions: {predictions}")
    print("Decoded indices:")
    for i, pred in enumerate(predictions):
        indices = power_set_encoder.decode(pred)
        print(f"  Frame {i}: {pred} -> {indices}")
    
    der = calculate_der(predictions, labels, power_set_encoder, debug=True)
    print(f"DER: {der}")
    
    if der == 0.0:
        print("✓ Fix works correctly!")
    else:
        print("✗ Fix needs more work")

if __name__ == "__main__":
    test_speaker_id_list_consistency()
    test_edge_cases()
    demonstrate_problem_fix()
    print("\n" + "="*50)
    print("All consistency tests completed!")
