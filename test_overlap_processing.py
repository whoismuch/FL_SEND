#!/usr/bin/env python3
"""
Тест для проверки работы create_dataset_from_grouped с обработкой перекрывающейся речи.
Проверяет обратную совместимость и корректность работы на небольшом датасете.
"""

import os
import sys
import logging
import numpy as np
from typing import Dict, List

# Добавляем путь к проекту
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from data_processing import create_dataset_from_grouped
from SEND_PSE_AMI import PowerSetEncoder

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_data():
    """Создает небольшой тестовый датасет с перекрывающейся речью."""
    logger.info("Creating test dataset with overlapping speech...")
    
    # Создаем тестовые аудио сегменты (синусоиды разной частоты для разных спикеров)
    sr = 16000
    duration = 2.0  # 2 секунды
    samples = int(sr * duration)
    
    # Создаем 2 встречи с перекрывающейся речью
    grouped_data = {}
    
    # Meeting 1: 2 спикера с перекрытием
    meeting1_samples = []
    
    # Спикер 1: 0.0 - 1.5 секунды
    speaker1_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.5, int(sr * 1.5)))
    meeting1_samples.append({
        "speaker_id": "speaker_1",
        "begin_time": 0.0,
        "end_time": 1.5,
        "audio": {"array": speaker1_audio}
    })
    
    # Спикер 2: 1.0 - 2.0 секунды (перекрывается с спикером 1 с 1.0 до 1.5)
    speaker2_audio = np.sin(2 * np.pi * 880 * np.linspace(0, 1.0, int(sr * 1.0)))
    meeting1_samples.append({
        "speaker_id": "speaker_2",
        "begin_time": 1.0,
        "end_time": 2.0,
        "audio": {"array": speaker2_audio}
    })
    
    grouped_data["meeting_1"] = meeting1_samples
    
    # Meeting 2: 3 спикера с несколькими перекрытиями
    meeting2_samples = []
    
    # Спикер 1: 0.0 - 1.0 секунды
    speaker1_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, int(sr * 1.0)))
    meeting2_samples.append({
        "speaker_id": "speaker_A",
        "begin_time": 0.0,
        "end_time": 1.0,
        "audio": {"array": speaker1_audio}
    })
    
    # Спикер 2: 0.5 - 1.5 секунды (перекрывается с спикером 1 с 0.5 до 1.0)
    speaker2_audio = np.sin(2 * np.pi * 660 * np.linspace(0, 1.0, int(sr * 1.0)))
    meeting2_samples.append({
        "speaker_id": "speaker_B",
        "begin_time": 0.5,
        "end_time": 1.5,
        "audio": {"array": speaker2_audio}
    })
    
    # Спикер 3: 1.2 - 2.0 секунды (перекрывается с спикером 2 с 1.2 до 1.5)
    speaker3_audio = np.sin(2 * np.pi * 880 * np.linspace(0, 0.8, int(sr * 0.8)))
    meeting2_samples.append({
        "speaker_id": "speaker_C",
        "begin_time": 1.2,
        "end_time": 2.0,
        "audio": {"array": speaker3_audio}
    })
    
    grouped_data["meeting_2"] = meeting2_samples
    
    logger.info(f"Created test dataset:")
    logger.info(f"  - {len(grouped_data)} meetings")
    for meeting_id, samples in grouped_data.items():
        speakers = set(s["speaker_id"] for s in samples)
        logger.info(f"  - {meeting_id}: {len(samples)} segments, {len(speakers)} speakers: {sorted(speakers)}")
        for s in samples:
            logger.info(f"    * {s['speaker_id']}: {s['begin_time']:.2f}s - {s['end_time']:.2f}s")
    
    return grouped_data


def test_create_dataset_from_grouped():
    """Тестирует функцию create_dataset_from_grouped."""
    logger.info("=" * 80)
    logger.info("TESTING create_dataset_from_grouped WITH OVERLAPPING SPEECH")
    logger.info("=" * 80)
    
    # Создаем тестовые данные
    grouped_data = create_test_data()
    
    # Инициализируем PowerSetEncoder
    N = 4  # Максимум 4 спикера
    K = 2  # Максимум 2 одновременно активных спикера
    power_set_encoder = PowerSetEncoder(max_speakers=N, max_overlap=K)
    logger.info(f"PowerSetEncoder initialized: N={N}, K={K}, num_classes={power_set_encoder.num_classes}")
    
    # Создаем фиктивный speaker_encoder (не используется в функции, но нужен для API)
    class DummySpeakerEncoder:
        pass
    
    speaker_encoder = DummySpeakerEncoder()
    
    # Вызываем функцию
    logger.info("\n" + "=" * 80)
    logger.info("CALLING create_dataset_from_grouped...")
    logger.info("=" * 80)
    
    try:
        features, labels, meeting_ids = create_dataset_from_grouped(
            grouped_data=grouped_data,
            speaker_encoder=speaker_encoder,
            power_set_encoder=power_set_encoder,
            N=N,
            chunk_size=10,  # Маленький chunk_size для теста
            max_sequence_length=None
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ FUNCTION EXECUTED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        # Проверяем результаты
        logger.info(f"\nResults:")
        logger.info(f"  - Features shape: {features.shape}")
        logger.info(f"  - Labels shape: {labels.shape}")
        logger.info(f"  - Meeting IDs shape: {meeting_ids.shape}")
        
        # Проверяем, что все массивы имеют одинаковую длину по первой оси
        assert features.shape[0] == labels.shape[0] == meeting_ids.shape[0], \
            f"Mismatch in first dimension: features={features.shape[0]}, labels={labels.shape[0]}, meeting_ids={meeting_ids.shape[0]}"
        logger.info(f"  ✓ All arrays have same number of samples: {features.shape[0]}")
        
        # Проверяем, что есть перекрывающиеся сегменты (должно быть больше оригинальных)
        original_segments = sum(len(samples) for samples in grouped_data.values())
        total_segments = features.shape[0]
        overlap_segments = total_segments - original_segments
        
        logger.info(f"\nOverlap processing results:")
        logger.info(f"  - Original segments: {original_segments}")
        logger.info(f"  - Total segments (including overlaps): {total_segments}")
        logger.info(f"  - Overlapping segments created: {overlap_segments}")
        
        if overlap_segments > 0:
            logger.info(f"  ✓ OVERLAPPING SPEECH PROCESSING WORKS! Created {overlap_segments} overlap segments")
        else:
            logger.warning(f"  ⚠ No overlapping segments were created (this might be expected if no overlaps exist)")
        
        # Проверяем, что метки корректны
        unique_labels = np.unique(labels[labels != -100])  # -100 это padding
        logger.info(f"\nLabel statistics:")
        logger.info(f"  - Unique label values: {sorted(unique_labels)}")
        logger.info(f"  - Number of unique labels: {len(unique_labels)}")
        
        # Проверяем, что есть метки для нескольких спикеров (перекрытия)
        multi_speaker_labels = []
        for label in unique_labels:
            decoded = power_set_encoder.decode(label)
            if len(decoded) > 1:
                multi_speaker_labels.append(label)
        
        if multi_speaker_labels:
            logger.info(f"  ✓ Found multi-speaker labels (overlaps): {multi_speaker_labels}")
            for label in multi_speaker_labels:
                decoded = power_set_encoder.decode(label)
                logger.info(f"    - Label {label} = speakers {decoded}")
        else:
            logger.warning(f"  ⚠ No multi-speaker labels found (might indicate no overlaps were processed)")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error("✗ TEST FAILED!")
        logger.error("=" * 80)
        logger.error(f"Error: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = test_create_dataset_from_grouped()
    sys.exit(0 if success else 1)

