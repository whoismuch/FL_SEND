#!/usr/bin/env python3
"""
Скрипт для тестирования экспорта результатов диаризации в RTTM формат.

Этот скрипт создает тестовые данные и демонстрирует, как работает экспорт
результатов диаризации в различные форматы.
"""

import os
import sys
import numpy as np
from typing import Dict, List
import logging

# Add current directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diarization_export import export_diarization_results, DiarizationExporter
from FL_SEND_PSE_AMI_improved import PowerSetEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_predictions() -> Dict[str, List[int]]:
    """
    Создает тестовые предсказания для демонстрации экспорта.
    
    Returns:
        Словарь с предсказаниями для разных записей
    """
    # Create PowerSetEncoder for 4 speakers
    power_set_encoder = PowerSetEncoder(max_speakers=4)
    
    # Test data: simulation of diarization for several recordings
    test_predictions = {
        'EN2002a': [
            # First 100 frames: only speaker 0
            *[power_set_encoder.encode([0]) for _ in range(100)],
            # Next 50 frames: speakers 0 and 1 (overlapping speech)
            *[power_set_encoder.encode([0, 1]) for _ in range(50)],
            # Next 80 frames: only speaker 1
            *[power_set_encoder.encode([1]) for _ in range(80)],
            # Next 30 frames: speakers 1 and 2
            *[power_set_encoder.encode([1, 2]) for _ in range(30)],
            # Last 40 frames: only speaker 2
            *[power_set_encoder.encode([2]) for _ in range(40)]
        ],
        
        'ES2004a': [
            # More complex diarization with three speakers
            *[power_set_encoder.encode([0]) for _ in range(60)],
            *[power_set_encoder.encode([0, 1]) for _ in range(40)],
            *[power_set_encoder.encode([1]) for _ in range(70)],
            *[power_set_encoder.encode([1, 2]) for _ in range(25)],
            *[power_set_encoder.encode([2]) for _ in range(50)],
            *[power_set_encoder.encode([0, 2]) for _ in range(35)],
            *[power_set_encoder.encode([0]) for _ in range(20)]
        ],
        
        'IS1009a': [
            # Simple diarization with two speakers
            *[power_set_encoder.encode([0]) for _ in range(120)],
            *[power_set_encoder.encode([1]) for _ in range(100)],
            *[power_set_encoder.encode([0, 1]) for _ in range(30)],
            *[power_set_encoder.encode([0]) for _ in range(50)]
        ]
    }
    
    return test_predictions


def create_test_ground_truth() -> Dict[str, List[int]]:
    """
    Create test ground truth data for metrics calculation.
    
    Returns:
        Dictionary with ground truth predictions for each recording
    """
    # Create PowerSetEncoder for 4 speakers
    power_set_encoder = PowerSetEncoder(max_speakers=4)
    
    # Test ground truth data: slightly different from predictions to show errors
    test_ground_truth = {
        'EN2002a': [
            # First 100 frames: only speaker 0 (same as prediction)
            *[power_set_encoder.encode([0]) for _ in range(100)],
            # Next 50 frames: only speaker 0 (different from prediction which has 0+1)
            *[power_set_encoder.encode([0]) for _ in range(50)],
            # Next 80 frames: only speaker 1 (same as prediction)
            *[power_set_encoder.encode([1]) for _ in range(80)],
            # Next 30 frames: only speaker 1 (different from prediction which has 1+2)
            *[power_set_encoder.encode([1]) for _ in range(30)],
            # Last 40 frames: only speaker 2 (same as prediction)
            *[power_set_encoder.encode([2]) for _ in range(40)]
        ],
        
        'ES2004a': [
            # Similar pattern with some differences
            *[power_set_encoder.encode([0]) for _ in range(60)],
            *[power_set_encoder.encode([0]) for _ in range(40)],  # Different from prediction
            *[power_set_encoder.encode([1]) for _ in range(70)],
            *[power_set_encoder.encode([1]) for _ in range(25)],  # Different from prediction
            *[power_set_encoder.encode([2]) for _ in range(50)],
            *[power_set_encoder.encode([0]) for _ in range(35)],  # Different from prediction
            *[power_set_encoder.encode([0]) for _ in range(20)]
        ],
        
        'IS1009a': [
            # Simple diarization with some differences
            *[power_set_encoder.encode([0]) for _ in range(120)],
            *[power_set_encoder.encode([1]) for _ in range(100)],
            *[power_set_encoder.encode([0]) for _ in range(30)],  # Different from prediction
            *[power_set_encoder.encode([0]) for _ in range(50)]
        ]
    }
    
    return test_ground_truth
def create_test_speaker_id_lists() -> Dict[str, List]:
    """
    Create test speaker ID lists for each recording.
    
    Returns:
        Dictionary with speaker IDs for each recording
    """
    return {
        'EN2002a': ['speaker_A', 'speaker_B', 'speaker_C', 'speaker_D'],
        'ES2004a': ['speaker_X', 'speaker_Y', 'speaker_Z', 'speaker_W'],
        'IS1009a': ['speaker_1', 'speaker_2', 'speaker_3', 'speaker_4']
    }


def test_export_functionality():
    """Тестирует функциональность экспорта результатов диаризации."""
    print("=== TESTING DIARIZATION RESULTS EXPORT ===\n")
    
    # Create test data
    test_predictions = create_test_predictions()
    test_ground_truth = create_test_ground_truth()
    test_speaker_id_lists = create_test_speaker_id_lists()
    power_set_encoder = PowerSetEncoder(max_speakers=4)
    
    print(f"Created {len(test_predictions)} test recordings:")
    for rec_id, predictions in test_predictions.items():
        duration = len(predictions) * 0.01  # frame_shift = 0.01
        print(f"  {rec_id}: {len(predictions)} frames ({duration:.2f} sec)")
    
    # Create directory for test export with experiment-like structure
    test_output_dir = "out_artifacts/diarization_export/test_exp_3size_1epochs_1rounds_2clients_2025-09-14-20-00"
    os.makedirs(test_output_dir, exist_ok=True)
    
    print(f"\nExporting results to directory: {test_output_dir}")
    
    try:
        # Export results
        export_results = export_diarization_results(
            predictions_by_recording=test_predictions,
            power_set_encoder=power_set_encoder,
            output_dir=test_output_dir,
            speaker_id_lists=test_speaker_id_lists,
            ground_truth_by_recording=test_ground_truth,
            formats=['rttm', 'ctm', 'summary', 'metrics']
        )
        
        print("\n✅ Export completed successfully!")
        print("\nCreated files:")
        
        for format_name, files in export_results.items():
            print(f"\n{format_name.upper()} files:")
            for file_path in files:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"  ✅ {file_path} ({file_size} bytes)")
                else:
                    print(f"  ❌ {file_path} (file not found)")
        
        # Show content of one RTTM file for verification
        if 'rttm' in export_results and export_results['rttm']:
            rttm_file = export_results['rttm'][0]
            print(f"\nExample content of RTTM file ({os.path.basename(rttm_file)}):")
            print("-" * 80)
            try:
                with open(rttm_file, 'r') as f:
                    lines = f.readlines()[:10]  # Show first 10 lines
                    for line in lines:
                        print(line.strip())
                    if len(lines) == 10:
                        print("... (showing first 10 lines)")
            except Exception as e:
                print(f"Error reading file: {e}")
            print("-" * 80)
        
        # Show content of metrics report
        if 'metrics' in export_results and export_results['metrics']:
            metrics_file = export_results['metrics'][0]
            print(f"\nContent of metrics report ({os.path.basename(metrics_file)}):")
            print("-" * 80)
            try:
                with open(metrics_file, 'r') as f:
                    content = f.read()
                    print(content)
            except Exception as e:
                print(f"Error reading file: {e}")
            print("-" * 80)
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        import traceback
        traceback.print_exc()


def test_individual_exporter():
    """Тестирует отдельные компоненты экспортера."""
    print("\n=== TESTING INDIVIDUAL COMPONENTS ===\n")
    
    power_set_encoder = PowerSetEncoder(max_speakers=4)
    exporter = DiarizationExporter(power_set_encoder)
    
    # Test conversion of predictions to segments
    test_predictions = [1, 1, 1, 2, 2, 2, 3, 3, 3]  # speaker_0, speaker_1, speaker_0+1
    segments = exporter.predictions_to_segments(test_predictions, "test_meeting")
    
    print(f"Converting predictions to segments:")
    print(f"  Input predictions: {test_predictions}")
    if segments is not None:
        print(f"  Number of segments: {len(segments)}")
        print(f"  Segments:")
        for i, segment in enumerate(segments):
            print(f"    {i+1}. {segment['speaker_id']}: {segment['start_time']:.3f}-{segment['end_time']:.3f} sec")
    else:
        print("  No segments generated")
    
    print("\n✅ Component testing completed!")


def main():
    """Основная функция для запуска тестов."""
    print("Starting diarization results export testing...\n")
    
    try:
        # Test individual components
        test_individual_exporter()
        
        # Test full export functionality
        test_export_functionality()
        
        print("\n🎉 All tests completed successfully!")
        print("\nNow you can use these files to compare with other diarization models.")
        print("RTTM files can be used with diarization evaluation tools such as:")
        print("- pyannote.metrics")
        print("- dscore")
        print("- pyannote-audio")
        
    except Exception as e:
        print(f"❌ Error during test execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
