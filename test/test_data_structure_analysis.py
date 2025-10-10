#!/usr/bin/env python3
"""
Подробный тест для понимания структуры входных данных в FL-SEND-PSE проекте.

Этот тест показывает:
1. Структуру AMI датасета
2. Как аудио преобразуется в mel-спектрограммы
3. Как создаются сэмплы, батчи и метки
4. Как работает Power Set Encoding
5. Как данные попадают в модель
"""

import os
import sys
import logging
import numpy as np
import torch
import librosa
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import (
    extract_features, 
    group_by_meeting, 
    power_set_encoding,
    compute_speaker_embeddings,
    create_dataset_from_grouped,
    OverlappingSpeechDataset,
    prepare_data_loaders
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataStructureAnalyzer:
    """Анализатор структуры данных для FL-SEND-PSE."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")
        
        # Инициализация speaker encoder
        logger.info("Инициализация speaker encoder...")
        self.speaker_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": self.device}
        ).to(self.device)
        logger.info("Speaker encoder инициализирован")
    
    def analyze_ami_dataset_structure(self, test_size: int = 2):
        """Анализ структуры AMI датасета."""
        print("\n" + "="*80)
        print("1. АНАЛИЗ СТРУКТУРЫ AMI ДАТАСЕТА")
        print("="*80)
        
        # Загрузка датасета
        logger.info("Загрузка AMI датасета...")
        dataset = load_dataset("edinburghcstr/ami", "ihm")
        
        # Анализ структуры всех splits
        train_split = dataset["train"].select(range(test_size))
        val_split = dataset["validation"].select(range(test_size))
        test_split = dataset["test"].select(range(test_size))
        
        logger.info(f"Загружено {len(train_split)} сэмплов из train split")
        logger.info(f"Загружено {len(val_split)} сэмплов из validation split")
        logger.info(f"Загружено {len(test_split)} сэмплов из test split")
        
        # Анализ первого сэмпла из train split
        first_sample = train_split[0]
        print(f"\n📊 СТРУКТУРА ПЕРВОГО СЭМПЛА (TRAIN):")
        print(f"Ключи сэмпла: {list(first_sample.keys())}")
        
        for key, value in first_sample.items():
            if key == "audio":
                audio_info = value
                print(f"\n🎵 АУДИО ИНФОРМАЦИЯ:")
                print(f"  - Тип: {type(audio_info)}")
                print(f"  - Ключи: {list(audio_info.keys())}")
                print(f"  - Sampling rate: {audio_info['sampling_rate']} Hz")
                print(f"  - Длина аудио массива: {len(audio_info['array'])} сэмплов")
                print(f"  - Длительность: {len(audio_info['array']) / audio_info['sampling_rate']:.2f} секунд")
                print(f"  - Диапазон значений: [{audio_info['array'].min():.4f}, {audio_info['array'].max():.4f}]")
            else:
                print(f"  - {key}: {type(value)} = {value}")
        
        # Группировка по встречам для всех splits
        print(f"\n🏢 ГРУППИРОВКА ПО ВСТРЕЧАМ:")
        
        print(f"\n📦 TRAIN SPLIT:")
        grouped_train = group_by_meeting(train_split)
        print(f"Количество встреч: {len(grouped_train)}")
        
        for meeting_id, samples in grouped_train.items():
            print(f"\n  Встреча {meeting_id}:")
            print(f"    - Количество сэмплов: {len(samples)}")
            
            # Анализ спикеров в встрече
            speakers = set(sample["speaker_id"] for sample in samples)
            print(f"    - Уникальные спикеры: {sorted(speakers)}")
            
            # Анализ временных меток
            begin_times = [sample["begin_time"] for sample in samples]
            end_times = [sample["end_time"] for sample in samples]
            print(f"    - Временной диапазон: {min(begin_times):.2f}s - {max(end_times):.2f}s")
            print(f"    - Общая длительность: {max(end_times) - min(begin_times):.2f}s")
            
            # Показываем первые несколько сэмплов
            print(f"    - Первые 3 сэмпла:")
            for i, sample in enumerate(samples[:3]):
                duration = sample["end_time"] - sample["begin_time"]
                print(f"      {i+1}. Спикер {sample['speaker_id']}, время: {sample['begin_time']:.2f}-{sample['end_time']:.2f}s ({duration:.2f}s)")
        
        print(f"\n📦 VALIDATION SPLIT:")
        grouped_validation = group_by_meeting(val_split)
        print(f"Количество встреч: {len(grouped_validation)}")
        
        for meeting_id, samples in grouped_validation.items():
            print(f"\n  Встреча {meeting_id}:")
            print(f"    - Количество сэмплов: {len(samples)}")
            
            # Анализ спикеров в встрече
            speakers = set(sample["speaker_id"] for sample in samples)
            print(f"    - Уникальные спикеры: {sorted(speakers)}")
            
            # Анализ временных меток
            begin_times = [sample["begin_time"] for sample in samples]
            end_times = [sample["end_time"] for sample in samples]
            print(f"    - Временной диапазон: {min(begin_times):.2f}s - {max(end_times):.2f}s")
            print(f"    - Общая длительность: {max(end_times) - min(begin_times):.2f}s")
            
            # Показываем первые несколько сэмплов
            print(f"    - Первые 3 сэмпла:")
            for i, sample in enumerate(samples[:3]):
                duration = sample["end_time"] - sample["begin_time"]
                print(f"      {i+1}. Спикер {sample['speaker_id']}, время: {sample['begin_time']:.2f}-{sample['end_time']:.2f}s ({duration:.2f}s)")
        
        print(f"\n📦 TEST SPLIT:")
        grouped_test = group_by_meeting(test_split)
        print(f"Количество встреч: {len(grouped_test)}")
        
        for meeting_id, samples in grouped_test.items():
            print(f"\n  Встреча {meeting_id}:")
            print(f"    - Количество сэмплов: {len(samples)}")
            
            # Анализ спикеров в встрече
            speakers = set(sample["speaker_id"] for sample in samples)
            print(f"    - Уникальные спикеры: {sorted(speakers)}")
            
            # Анализ временных меток
            begin_times = [sample["begin_time"] for sample in samples]
            end_times = [sample["end_time"] for sample in samples]
            print(f"    - Временной диапазон: {min(begin_times):.2f}s - {max(end_times):.2f}s")
            print(f"    - Общая длительность: {max(end_times) - min(begin_times):.2f}s")
            
            # Показываем первые несколько сэмплов
            print(f"    - Первые 3 сэмпла:")
            for i, sample in enumerate(samples[:3]):
                duration = sample["end_time"] - sample["begin_time"]
                print(f"      {i+1}. Спикер {sample['speaker_id']}, время: {sample['begin_time']:.2f}-{sample['end_time']:.2f}s ({duration:.2f}s)")
        
        # Сводная статистика
        print(f"\n📊 СВОДНАЯ СТАТИСТИКА:")
        print(f"  - Train: {len(grouped_train)} встреч, {sum(len(samples) for samples in grouped_train.values())} сэмплов")
        print(f"  - Validation: {len(grouped_validation)} встреч, {sum(len(samples) for samples in grouped_validation.values())} сэмплов")
        print(f"  - Test: {len(grouped_test)} встреч, {sum(len(samples) for samples in grouped_test.values())} сэмплов")
        
        return grouped_train, grouped_validation, grouped_test
    
    def analyze_group_by_meeting_detailed(self, test_size: int = 2):
        """Подробный анализ функции group_by_meeting с пошаговым объяснением."""
        print("\n" + "="*80)
        print("🔍 ПОДРОБНЫЙ АНАЛИЗ group_by_meeting")
        print("="*80)
        
        print(f"\n📋 ЦЕЛЬ ФУНКЦИИ:")
        print(f"  - Сгруппировать сэмплы датасета по meeting_id")
        print(f"  - Создать словарь: meeting_id → список_сэмплов")
        print(f"  - Объединить все аудио сегменты одной встречи в одну группу")
        print(f"  - Подготовить данные для дальнейшей обработки по встречам")
        
        # Загрузка датасета
        print(f"\n📥 ЗАГРУЗКА ДАННЫХ:")
        dataset = load_dataset("edinburghcstr/ami", "ihm")
        train_split = dataset["train"].select(range(test_size))
        print(f"  - Загружено {len(train_split)} сэмплов из train split")
        
        # Анализ входных данных
        print(f"\n🔍 АНАЛИЗ ВХОДНЫХ ДАННЫХ:")
        print(f"  - Тип: {type(train_split)}")
        print(f"  - Количество сэмплов: {len(train_split)}")
        
        # Показываем структуру каждого сэмпла
        print(f"\n📊 СТРУКТУРА СЭМПЛОВ:")
        for i, sample in enumerate(train_split):
            print(f"  Сэмпл {i}:")
            print(f"    - meeting_id: {sample['meeting_id']}")
            print(f"    - speaker_id: {sample['speaker_id']}")
            print(f"    - begin_time: {sample['begin_time']:.2f}s")
            print(f"    - end_time: {sample['end_time']:.2f}s")
            print(f"    - Длительность: {sample['end_time'] - sample['begin_time']:.2f}s")
            print(f"    - Аудио: {len(sample['audio']['array'])} сэмплов")
        
        # Пошаговое выполнение group_by_meeting
        print(f"\n🔄 ПОШАГОВОЕ ВЫПОЛНЕНИЕ group_by_meeting:")
        
        # Инициализация
        print(f"\n  🔧 ИНИЦИАЛИЗАЦИЯ:")
        grouped = {}
        print(f"    - grouped = {{}} (пустой словарь)")
        print(f"    - Тип: {type(grouped)}")
        
        # Обработка каждого сэмпла
        print(f"\n  📍 ОБРАБОТКА КАЖДОГО СЭМПЛА:")
        for i, sample in enumerate(train_split):
            print(f"\n    Сэмпл {i}:")
            meeting_id = sample["meeting_id"]
            print(f"      - meeting_id: {meeting_id}")
            print(f"      - speaker_id: {sample['speaker_id']}")
            print(f"      - Время: {sample['begin_time']:.2f}s - {sample['end_time']:.2f}s")
            
            # Проверка существования ключа
            if meeting_id in grouped:
                print(f"      - ✅ Ключ {meeting_id} уже существует в grouped")
                print(f"      - Текущее количество сэмплов: {len(grouped[meeting_id])}")
            else:
                print(f"      - 🆕 Ключ {meeting_id} не существует, создаем новый список")
                grouped[meeting_id] = []
                print(f"      - Создан пустой список для {meeting_id}")
            
            # Добавление сэмпла
            grouped[meeting_id].append(sample)
            print(f"      - ➕ Добавлен сэмпл в группу {meeting_id}")
            print(f"      - Новое количество сэмплов: {len(grouped[meeting_id])}")
        
        # Анализ результата
        print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТА:")
        print(f"  - Количество встреч: {len(grouped)}")
        print(f"  - Ключи (meeting_ids): {list(grouped.keys())}")
        
        for meeting_id, samples in grouped.items():
            print(f"\n  Встреча {meeting_id}:")
            print(f"    - Количество сэмплов: {len(samples)}")
            
            # Анализ спикеров
            speakers = set(sample["speaker_id"] for sample in samples)
            print(f"    - Уникальные спикеры: {sorted(speakers)}")
            print(f"    - Количество спикеров: {len(speakers)}")
            
            # Анализ временных меток
            begin_times = [sample["begin_time"] for sample in samples]
            end_times = [sample["end_time"] for sample in samples]
            print(f"    - Временной диапазон: {min(begin_times):.2f}s - {max(end_times):.2f}s")
            print(f"    - Общая длительность: {max(end_times) - min(begin_times):.2f}s")
            
            # Детальный анализ сэмплов
            print(f"    - Детальный анализ сэмплов:")
            for j, sample in enumerate(samples):
                duration = sample["end_time"] - sample["begin_time"]
                print(f"      {j+1}. {sample['speaker_id']}: {sample['begin_time']:.2f}s-{sample['end_time']:.2f}s ({duration:.2f}s)")
        
        # Сравнение с оригинальной функцией
        print(f"\n🔍 СРАВНЕНИЕ С ОРИГИНАЛЬНОЙ ФУНКЦИЕЙ:")
        original_grouped = group_by_meeting(train_split)
        print(f"  - Результат нашей реализации: {len(grouped)} встреч")
        print(f"  - Результат оригинальной функции: {len(original_grouped)} встреч")
        
        # Сравнение структур (без сравнения numpy массивов)
        keys_match = set(grouped.keys()) == set(original_grouped.keys())
        lengths_match = all(len(grouped[k]) == len(original_grouped[k]) for k in grouped.keys())
        print(f"  - Ключи совпадают: {keys_match}")
        print(f"  - Количества сэмплов совпадают: {lengths_match}")
        
        # Детальное сравнение
        if keys_match and lengths_match:
            print(f"  - ✅ Структуры совпадают")
        else:
            print(f"  - ❌ Структуры различаются")
        
        # Анализ структуры данных
        print(f"\n📋 АНАЛИЗ СТРУКТУРЫ ДАННЫХ:")
        print(f"  - Тип результата: {type(grouped)}")
        print(f"  - Тип ключей: {type(list(grouped.keys())[0]) if grouped else 'N/A'}")
        print(f"  - Тип значений: {type(list(grouped.values())[0]) if grouped else 'N/A'}")
        
        if grouped:
            first_key = list(grouped.keys())[0]
            first_value = grouped[first_key]
            print(f"  - Первый ключ: {first_key} (тип: {type(first_key)})")
            print(f"  - Первое значение: {len(first_value)} сэмплов (тип: {type(first_value)})")
            print(f"  - Тип сэмпла: {type(first_value[0]) if first_value else 'N/A'}")
        
        # Объяснение использования
        print(f"\n💡 ОБЪЯСНЕНИЕ ИСПОЛЬЗОВАНИЯ:")
        print(f"  🎯 Зачем нужна группировка по встречам:")
        print(f"    1. Обработка по встречам:")
        print(f"       - Каждая встреча обрабатывается отдельно")
        print(f"       - Создается маппинг спикеров на слоты для каждой встречи")
        print(f"       - Избегается смешивание спикеров между встречами")
        print(f"    2. Создание speaker embeddings:")
        print(f"       - Для каждой встречи создаются embeddings всех спикеров")
        print(f"       - Embeddings специфичны для каждой встречи")
        print(f"    3. Подготовка к Power Set Encoding:")
        print(f"       - Каждая встреча имеет свой набор спикеров")
        print(f"       - PSE кодирует активность спикеров в рамках одной встречи")
        print(f"    4. Валидация и метрики:")
        print(f"       - DER вычисляется по встречам")
        print(f"       - Результаты группируются по meeting_id")
        
        print(f"\n  🔄 Как используется результат:")
        print(f"    1. create_dataset_from_grouped:")
        print(f"       - Принимает grouped_data как вход")
        print(f"       - Обрабатывает каждую встречу отдельно")
        print(f"       - Создает маппинг спикеров на слоты")
        print(f"    2. compute_speaker_embeddings:")
        print(f"       - Извлекает embeddings для всех спикеров")
        print(f"       - Группирует аудио по speaker_id")
        print(f"    3. Подготовка к обучению:")
        print(f"       - Создает OverlappingSpeechDataset")
        print(f"       - Группирует данные для батчей")
        
        print(f"\n  📊 Структура данных после группировки:")
        print(f"    grouped_data = {{")
        for meeting_id, samples in grouped.items():
            speakers = set(sample["speaker_id"] for sample in samples)
            print(f"      '{meeting_id}': [")
            print(f"        # {len(samples)} сэмплов, спикеры: {sorted(speakers)}")
            print(f"        # Временной диапазон: {min([s['begin_time'] for s in samples]):.1f}s - {max([s['end_time'] for s in samples]):.1f}s")
            print(f"        # ... {len(samples)} сэмплов ...")
            print(f"      ],")
        print(f"    }}")
        
        return grouped
    
    def analyze_feature_extraction(self, grouped_data: Dict):
        """Анализ извлечения признаков из аудио."""
        print("\n" + "="*80)
        print("2. АНАЛИЗ ИЗВЛЕЧЕНИЯ ПРИЗНАКОВ (MEL-СПЕКТРОГРАММЫ)")
        print("="*80)
        
        # Берем первый сэмпл из первой встречи
        first_meeting_id = list(grouped_data.keys())[0]
        first_sample = grouped_data[first_meeting_id][0]
        
        print(f"\n🎵 ИСХОДНОЕ АУДИО:")
        audio_array = first_sample["audio"]["array"]
        sr = first_sample["audio"]["sampling_rate"]
        print(f"  - Длина массива: {len(audio_array)} сэмплов")
        print(f"  - Sampling rate: {sr} Hz")
        print(f"  - Длительность: {len(audio_array) / sr:.2f} секунд")
        print(f"  - Диапазон значений: [{audio_array.min():.4f}, {audio_array.max():.4f}]")
        
        # Извлечение mel-спектрограммы
        print(f"\n🔬 ИЗВЛЕЧЕНИЕ MEL-СПЕКТРОГРАММЫ:")
        mel_features = extract_features(audio_array, sr=sr, n_mels=80)
        
        print(f"  - Форма mel-спектрограммы: {mel_features.shape}")
        print(f"    * {mel_features.shape[0]} кадров (временные шаги)")
        print(f"    * {mel_features.shape[1]} mel-каналов (частотные характеристики)")
        print(f"  - Временное разрешение: {mel_features.shape[0] * 0.01:.2f} секунд (10ms на кадр)")
        print(f"  - Частотное разрешение: 80 mel-каналов")
        print(f"  - Диапазон значений: [{mel_features.min():.2f}, {mel_features.max():.2f}]")
        
        # Детальный анализ первых кадров
        print(f"\n📋 ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРВЫХ 5 КАДРОВ:")
        for i in range(min(5, mel_features.shape[0])):
            frame = mel_features[i]
            print(f"  Кадр {i+1}:")
            print(f"    - Форма: {frame.shape}")
            print(f"    - Среднее: {frame.mean():.4f}")
            print(f"    - Стандартное отклонение: {frame.std():.4f}")
            print(f"    - Первые 5 значений: {frame[:5]}")
            print(f"    - Последние 5 значений: {frame[-5:]}")
        
        # Визуализация mel-спектрограммы
        self._visualize_mel_spectrogram(mel_features, f"mel_spectrogram_{first_meeting_id}")
        
        return mel_features
    
    def analyze_power_set_encoding(self):
        """Анализ Power Set Encoding."""
        print("\n" + "="*80)
        print("3. АНАЛИЗ POWER SET ENCODING")
        print("="*80)
        
        max_speakers = 4
        print(f"\n🔢 POWER SET ENCODING ДЛЯ {max_speakers} СПИКЕРОВ:")
        print(f"Максимальное количество классов: 2^{max_speakers} = {2**max_speakers}")
        
        # Примеры кодирования
        print(f"\n📝 ПРИМЕРЫ КОДИРОВАНИЯ:")
        
        # Одиночные спикеры
        print(f"  Одиночные спикеры:")
        for speaker in range(max_speakers):
            encoded = power_set_encoding(speaker)
            binary = format(encoded, f'0{max_speakers}b')
            print(f"    Спикер {speaker}: {encoded} (binary: {binary})")
        
        # Множественные спикеры (перекрывающаяся речь)
        print(f"\n  Перекрывающаяся речь:")
        examples = [
            [0, 1],    # Спикеры 0 и 1
            [1, 2],    # Спикеры 1 и 2
            [0, 2],    # Спикеры 0 и 2
            [0, 1, 2], # Спикеры 0, 1 и 2
            [0, 1, 2, 3] # Все спикеры
        ]
        
        for speakers in examples:
            encoded = power_set_encoding(speakers)
            binary = format(encoded, f'0{max_speakers}b')
            print(f"    Спикеры {speakers}: {encoded} (binary: {binary})")
        
        # Декодирование
        print(f"\n🔄 ДЕКОДИРОВАНИЕ:")
        test_values = [1, 3, 5, 15]
        for value in test_values:
            binary = format(value, f'0{max_speakers}b')
            active_speakers = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
            print(f"    {value} (binary: {binary}) -> активные спикеры: {active_speakers}")
    
    def analyze_speaker_embeddings(self, grouped_data: Dict):
        """Анализ speaker embeddings."""
        print("\n" + "="*80)
        print("4. АНАЛИЗ SPEAKER EMBEDDINGS")
        print("="*80)
        
        # Вычисление speaker embeddings
        print(f"\n🧠 ВЫЧИСЛЕНИЕ SPEAKER EMBEDDINGS:")
        speaker_to_embedding = compute_speaker_embeddings(grouped_data, self.speaker_encoder)
        
        print(f"Количество уникальных спикеров: {len(speaker_to_embedding)}")
        
        for speaker_id, embedding in speaker_to_embedding.items():
            print(f"\n  Спикер {speaker_id}:")
            print(f"    - Форма embedding: {embedding.shape}")
            print(f"    - Тип: {type(embedding)}")
            print(f"    - Диапазон значений: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"    - Среднее: {embedding.mean():.4f}")
            print(f"    - Стандартное отклонение: {embedding.std():.4f}")
            print(f"    - Первые 10 значений: {embedding[:10]}")
        
        return speaker_to_embedding
    
    def analyze_dataset_creation(self, grouped_data: Dict, speaker_to_embedding: Dict):
        """Анализ создания датасета."""
        print("\n" + "="*80)
        print("5. АНАЛИЗ СОЗДАНИЯ ДАТАСЕТА")
        print("="*80)
        
        N = 4  # Максимальное количество спикеров на запись
        
        print(f"\n📦 СОЗДАНИЕ ДАТАСЕТА С N={N} СЛОТАМИ НА ЗАПИСЬ:")
        
        # Создание датасета
        features, labels, meeting_ids = create_dataset_from_grouped(
            grouped_data, self.speaker_encoder, N=N
        )
        
        print(f"\n📊 СТАТИСТИКИ ДАТАСЕТА:")
        print(f"  - Количество сэмплов: {features.shape[0]}")
        print(f"  - Форма features: {features.shape}")
        print(f"  - Форма labels: {labels.shape}")
        print(f"  - Форма meeting_ids: {meeting_ids.shape}")
        
        # Анализ padding
        print(f"\n🔧 АНАЛИЗ PADDING:")
        print(f"  - Максимальная длина последовательности: {features.shape[1]} кадров")
        print(f"  - Количество mel-каналов: {features.shape[2]}")
        
        # Анализ меток
        print(f"\n🏷️ АНАЛИЗ МЕТОК:")
        unique_labels = np.unique(labels)
        print(f"  - Уникальные значения меток: {unique_labels}")
        print(f"  - Количество уникальных меток: {len(unique_labels)}")
        
        # Распределение меток
        label_counts = {}
        for label in unique_labels:
            if label != -100:  # Исключаем padding
                count = np.sum(labels == label)
                label_counts[label] = count
        
        print(f"  - Распределение меток (без padding):")
        for label, count in sorted(label_counts.items()):
            binary = format(label, f'0{N}b')
            active_speakers = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
            print(f"    {label} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
        
        # Анализ первого сэмпла
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРВОГО СЭМПЛА:")
        first_feature = features[0]
        first_label = labels[0]
        first_meeting_id = meeting_ids[0]
        
        print(f"  - Форма feature: {first_feature.shape}")
        print(f"  - Форма label: {first_label.shape}")
        print(f"  - Форма meeting_id: {first_meeting_id.shape}")
        
        # Находим non-padding кадры
        non_padding_mask = first_label != -100
        non_padding_count = np.sum(non_padding_mask)
        print(f"  - Non-padding кадры: {non_padding_count} из {len(first_label)}")
        
        if non_padding_count > 0:
            non_padding_labels = first_label[non_padding_mask]
            unique_non_padding = np.unique(non_padding_labels)
            print(f"  - Уникальные метки (без padding): {unique_non_padding}")
            
            for label in unique_non_padding:
                count = np.sum(non_padding_labels == label)
                binary = format(label, f'0{N}b')
                active_speakers = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
                print(f"    {label} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
        
        return features, labels, meeting_ids
    
    def analyze_overlapping_speech_dataset(self, features: np.ndarray, labels: np.ndarray, 
                                         meeting_ids: np.ndarray, speaker_to_embedding: Dict, grouped_data: Dict):
        """Анализ OverlappingSpeechDataset."""
        print("\n" + "="*80)
        print("6. АНАЛИЗ OVERLAPPINGSPEECHDATASET")
        print("="*80)
        
        # Создание списка speaker_ids (один на сэмпл)
        speaker_ids = []
        for meeting_id, samples in grouped_data.items():
            for sample in samples:
                speaker_ids.append(sample["speaker_id"])
        
        # Создание датасета
        dataset = OverlappingSpeechDataset(
            features=features,
            labels=labels,
            meeting_ids=meeting_ids,
            speaker_ids=speaker_ids,
            speaker_to_embedding=speaker_to_embedding,
            max_speakers=4
        )
        
        print(f"\n📦 СТРУКТУРА ДАТАСЕТА:")
        print(f"  - Размер датасета: {len(dataset)} сэмплов")
        print(f"  - Speaker ID list: {dataset.speaker_id_list}")
        print(f"  - Форма embeddings: {dataset.all_embeddings.shape}")
        
        # Анализ первого элемента
        print(f"\n🔍 АНАЛИЗ ПЕРВОГО ЭЛЕМЕНТА ДАТАСЕТА:")
        feature, all_embeddings, label, meeting_id = dataset[0]
        
        print(f"  - Форма feature: {feature.shape}")
        print(f"  - Форма all_embeddings: {all_embeddings.shape}")
        print(f"  - Форма label: {label.shape}")
        print(f"  - Форма meeting_id: {meeting_id.shape}")
        
        # Анализ embeddings
        print(f"\n🧠 АНАЛИЗ SPEAKER EMBEDDINGS:")
        for i, embedding in enumerate(all_embeddings):
            print(f"  Слот {i} (спикер {dataset.speaker_id_list[i]}):")
            print(f"    - Форма: {embedding.shape}")
            print(f"    - Диапазон: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"    - Среднее: {embedding.mean():.4f}")
        
        return dataset
    
    def analyze_batch_creation(self, dataset: OverlappingSpeechDataset):
        """Анализ создания батчей."""
        print("\n" + "="*80)
        print("7. АНАЛИЗ СОЗДАНИЯ БАТЧЕЙ")
        print("="*80)
        
        from torch.utils.data import DataLoader
        
        # Функция collate_fn
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
                    meeting_id = np.pad(meeting_id, (0, pad_len), mode='constant', constant_values=None)
                
                features.append(feature)
                speaker_embeddings.append(all_embeddings)
                labels.append(label)
                meeting_ids.append(meeting_id)
            
            features = torch.tensor(np.array(features), dtype=torch.float32)
            speaker_embeddings = torch.stack(speaker_embeddings).float()
            labels = torch.tensor(np.array(labels), dtype=torch.long)
            
            return features, speaker_embeddings, labels, meeting_ids
        
        # Создание DataLoader
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        
        print(f"\n📦 СТРУКТУРА DATALOADER:")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Количество батчей: {len(dataloader)}")
        
        # Анализ первого батча
        print(f"\n🔍 АНАЛИЗ ПЕРВОГО БАТЧА:")
        batch = next(iter(dataloader))
        features, speaker_embeddings, labels, meeting_ids = batch
        
        print(f"  - Форма features: {features.shape}")
        print(f"    * Batch size: {features.shape[0]}")
        print(f"    * Sequence length: {features.shape[1]} кадров")
        print(f"    * Feature dimension: {features.shape[2]} mel-каналов")
        
        print(f"  - Форма speaker_embeddings: {speaker_embeddings.shape}")
        print(f"    * Batch size: {speaker_embeddings.shape[0]}")
        print(f"    * Number of speakers: {speaker_embeddings.shape[1]}")
        print(f"    * Embedding dimension: {speaker_embeddings.shape[2]}")
        
        print(f"  - Форма labels: {labels.shape}")
        print(f"    * Batch size: {labels.shape[0]}")
        print(f"    * Sequence length: {labels.shape[1]} кадров")
        
        print(f"  - Форма meeting_ids: {len(meeting_ids)} элементов")
        for i, meeting_id_array in enumerate(meeting_ids):
            print(f"    * Сэмпл {i}: {meeting_id_array.shape}")
        
        # Анализ меток в батче
        print(f"\n🏷️ АНАЛИЗ МЕТОК В БАТЧЕ:")
        for i in range(features.shape[0]):
            sample_labels = labels[i]
            unique_labels = torch.unique(sample_labels)
            print(f"  Сэмпл {i}:")
            print(f"    - Уникальные метки: {unique_labels.tolist()}")
            
            for label in unique_labels:
                if label != -100:  # Исключаем padding
                    count = torch.sum(sample_labels == label).item()
                    binary = format(label.item(), '04b')
                    active_speakers = [j for j, bit in enumerate(binary[::-1]) if bit == '1']
                    print(f"      {label.item()} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
        
        return dataloader
    
    def analyze_model_input(self, dataloader):
        """Анализ входных данных для модели."""
        print("\n" + "="*80)
        print("8. АНАЛИЗ ВХОДНЫХ ДАННЫХ ДЛЯ МОДЕЛИ")
        print("="*80)
        
        # Получаем первый батч
        batch = next(iter(dataloader))
        features, speaker_embeddings, labels, meeting_ids = batch
        
        print(f"\n🎯 ВХОДНЫЕ ДАННЫЕ ДЛЯ МОДЕЛИ SEND:")
        print(f"  - features: {features.shape}")
        print(f"    * Это mel-спектрограммы для каждого сэмпла в батче")
        print(f"    * Каждый кадр представляет 10ms аудио")
        print(f"    * 80 mel-каналов кодируют частотную информацию")
        
        print(f"  - speaker_embeddings: {speaker_embeddings.shape}")
        print(f"    * Это предобученные embeddings для каждого спикера")
        print(f"    * Используются для контекстно-независимого скоринга")
        print(f"    * 192-мерные векторы из ECAPA-TDNN")
        
        print(f"  - labels: {labels.shape}")
        print(f"    * Power Set Encoded метки для каждого кадра")
        print(f"    * Показывают, какие спикеры активны в каждый момент времени")
        
        print(f"  - meeting_ids: {len(meeting_ids)} элементов")
        print(f"    * Идентификаторы встреч для каждого кадра")
        print(f"    * Используются для группировки при вычислении DER")
        
        # Детальный анализ первого сэмпла
        print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРВОГО СЭМПЛА:")
        first_features = features[0]
        first_embeddings = speaker_embeddings[0]
        first_labels = labels[0]
        
        print(f"  - features[0]: {first_features.shape}")
        print(f"    * {first_features.shape[0]} кадров × {first_features.shape[1]} mel-каналов")
        print(f"    * Первый кадр: {first_features[0][:5]}...")
        print(f"    * Последний кадр: {first_features[-1][:5]}...")
        
        print(f"  - speaker_embeddings[0]: {first_embeddings.shape}")
        print(f"    * {first_embeddings.shape[0]} спикеров × {first_embeddings.shape[1]} размерность")
        for i, embedding in enumerate(first_embeddings):
            print(f"    * Спикер {i}: {embedding[:5]}...")
        
        print(f"  - labels[0]: {first_labels.shape}")
        non_padding_mask = first_labels != -100
        non_padding_labels = first_labels[non_padding_mask]
        if len(non_padding_labels) > 0:
            unique_labels = torch.unique(non_padding_labels)
            print(f"    * Уникальные метки (без padding): {unique_labels.tolist()}")
            for label in unique_labels:
                count = torch.sum(non_padding_labels == label).item()
                binary = format(label.item(), '04b')
                active_speakers = [j for j, bit in enumerate(binary[::-1]) if bit == '1']
                print(f"      {label.item()} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
    
    def _visualize_mel_spectrogram(self, mel_features: np.ndarray, title: str):
        """Визуализация mel-спектрограммы."""
        try:
            plt.figure(figsize=(12, 6))
            plt.imshow(mel_features.T, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Log Mel Power')
            plt.title(f'Mel Spectrogram: {title}')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Channels')
            plt.tight_layout()
            
            # Сохранение
            os.makedirs('test_outputs', exist_ok=True)
            plt.savefig(f'test_outputs/{title}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  - Mel-спектрограмма сохранена: test_outputs/{title}.png")
        except Exception as e:
            print(f"  - Ошибка при сохранении визуализации: {e}")
    
    def run_complete_analysis(self, test_size: int = 2):
        """Запуск полного анализа структуры данных."""
        print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА СТРУКТУРЫ ДАННЫХ FL-SEND-PSE")
        print("="*80)
        
        try:
            # 1. Анализ структуры AMI датасета
            grouped_train, grouped_validation, grouped_test = self.analyze_ami_dataset_structure(test_size)
            
            # 1.1. Подробный анализ group_by_meeting
            print("\n" + "="*80)
            print("🔍 ЗАПУСК ПОДРОБНОГО АНАЛИЗА group_by_meeting")
            print("="*80)
            detailed_grouped_data = self.analyze_group_by_meeting_detailed(test_size)
            
            # 2. Анализ извлечения признаков
            mel_features = self.analyze_feature_extraction(grouped_train)
            
            # 3. Анализ Power Set Encoding
            self.analyze_power_set_encoding()
            
            # 4. Анализ speaker embeddings
            speaker_to_embedding = self.analyze_speaker_embeddings(grouped_train)
            
            # 4.1. Подробный анализ compute_speaker_embeddings
            print("\n" + "="*80)
            print("🔍 ЗАПУСК ПОДРОБНОГО АНАЛИЗА compute_speaker_embeddings")
            print("="*80)
            detailed_speaker_to_embedding = self.analyze_compute_speaker_embeddings_detailed(grouped_train)
            
            # 5. Анализ создания датасета
            features, labels, meeting_ids = self.analyze_dataset_creation(grouped_train, speaker_to_embedding)
            
            # 5.1. Подробный анализ create_dataset_from_grouped
            print("\n" + "="*80)
            print("🔍 ЗАПУСК ПОДРОБНОГО АНАЛИЗА create_dataset_from_grouped")
            print("="*80)
            detailed_features, detailed_labels, detailed_meeting_ids = self.analyze_create_dataset_from_grouped_detailed(grouped_train, speaker_to_embedding)
            
            # 6. Анализ OverlappingSpeechDataset
            dataset = self.analyze_overlapping_speech_dataset(features, labels, meeting_ids, speaker_to_embedding, grouped_train)
            
            # 7. Анализ создания батчей
            dataloader = self.analyze_batch_creation(dataset)
            
            # 8. Анализ входных данных для модели
            self.analyze_model_input(dataloader)
            
            # 9. Подробный анализ prepare_data_loaders
            print("\n" + "="*80)
            print("🔍 ЗАПУСК ПОДРОБНОГО АНАЛИЗА prepare_data_loaders")
            print("="*80)
            detailed_train_loader, detailed_val_loader, detailed_test_loader = self.analyze_prepare_data_loaders_detailed(grouped_train, grouped_validation, grouped_test)
            
            print("\n" + "="*80)
            print("✅ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("="*80)
            
            # Создание итогового отчета
            self._create_summary_report()
            
        except Exception as e:
            logger.error(f"Ошибка при анализе: {e}")
            raise
    
    def analyze_create_dataset_from_grouped_detailed(self, grouped_data: Dict, speaker_to_embedding: Dict):
        """Подробный анализ функции create_dataset_from_grouped с логированием каждого шага."""
        print("\n" + "="*80)
        print("🔍 ПОДРОБНЫЙ АНАЛИЗ create_dataset_from_grouped")
        print("="*80)
        
        N = 4  # Максимальное количество спикеров на запись
        
        print(f"\n📋 ВХОДНЫЕ ДАННЫЕ:")
        print(f"  - grouped_data: {len(grouped_data)} встреч")
        print(f"  - speaker_to_embedding: {len(speaker_to_embedding)} спикеров")
        print(f"  - N (максимум спикеров на запись): {N}")
        
        # Показываем структуру grouped_data
        print(f"\n🏢 СТРУКТУРА GROUPED_DATA:")
        for meeting_id, samples in grouped_data.items():
            print(f"  Встреча {meeting_id}:")
            print(f"    - Количество сэмплов: {len(samples)}")
            speakers = set(sample["speaker_id"] for sample in samples)
            print(f"    - Уникальные спикеры: {sorted(speakers)}")
            print(f"    - Первые 3 сэмпла:")
            for i, sample in enumerate(samples[:3]):
                duration = sample["end_time"] - sample["begin_time"]
                print(f"      {i+1}. {sample['speaker_id']}: {duration:.2f}с")
        
        # Инициализация переменных
        print(f"\n🔧 ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ:")
        features = []
        labels = []
        meeting_ids = []
        raw_features = []
        raw_labels = []
        raw_meeting_ids = []
        speaker_ids = []
        
        print(f"  - features: {len(features)} (пустой список)")
        print(f"  - labels: {len(labels)} (пустой список)")
        print(f"  - meeting_ids: {len(meeting_ids)} (пустой список)")
        print(f"  - raw_features: {len(raw_features)} (пустой список)")
        print(f"  - raw_labels: {len(raw_labels)} (пустой список)")
        print(f"  - raw_meeting_ids: {len(raw_meeting_ids)} (пустой список)")
        print(f"  - speaker_ids: {len(speaker_ids)} (пустой список)")
        
        # Первый проход: извлечение признаков и создание маппинга спикеров
        print(f"\n🔄 ПЕРВЫЙ ПРОХОД: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ И МАППИНГ СПИКЕРОВ")
        
        for meeting_id, samples in grouped_data.items():
            print(f"\n  📍 ОБРАБОТКА ВСТРЕЧИ: {meeting_id}")
            print(f"    - Количество сэмплов: {len(samples)}")
            
            # Создание маппинга спикеров на слоты
            meeting_speakers = list(set(sample["speaker_id"] for sample in samples))
            print(f"    - Уникальные спикеры: {meeting_speakers}")
            
            speaker_to_slot = {}
            for i, speaker_id in enumerate(meeting_speakers[:N]):  # Ограничиваем N слотами
                speaker_to_slot[speaker_id] = i
                print(f"      Спикер {speaker_id} → слот {i}")
            
            print(f"    - Итоговый маппинг: {speaker_to_slot}")
            
            # Обработка каждого сэмпла в встрече
            for sample_idx, sample in enumerate(samples):
                print(f"\n    🎵 СЭМПЛ {sample_idx + 1}:")
                print(f"      - Спикер: {sample['speaker_id']}")
                print(f"      - Время: {sample['begin_time']:.2f}с - {sample['end_time']:.2f}с")
                print(f"      - Длительность: {sample['end_time'] - sample['begin_time']:.2f}с")
                
                # Извлечение признаков
                print(f"      🔬 ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ:")
                audio_array = sample["audio"]["array"]
                print(f"        - Длина аудио: {len(audio_array)} сэмплов")
                print(f"        - Sampling rate: {sample['audio']['sampling_rate']} Hz")
                
                feature = extract_features(audio_array)
                print(f"        - Результат: {feature.shape} (кадры × mel-каналы)")
                print(f"        - Диапазон значений: [{feature.min():.2f}, {feature.max():.2f}]")
                
                raw_features.append(feature)
                print(f"        - Добавлено в raw_features (теперь {len(raw_features)} элементов)")
                
                # Маппинг спикера на слот и кодирование
                speaker_id = sample["speaker_id"]
                print(f"      🏷️ СОЗДАНИЕ МЕТКИ:")
                print(f"        - Спикер ID: {speaker_id}")
                
                if speaker_id in speaker_to_slot:
                    slot_idx = speaker_to_slot[speaker_id]
                    label = power_set_encoding(slot_idx)
                    print(f"        - Слот: {slot_idx}")
                    print(f"        - Power Set Encoding: {label} (binary: {format(label, '04b')})")
                else:
                    print(f"        - ⚠️ Спикер не в топ-{N}, назначаем слот 0")
                    label = power_set_encoding(0)
                    print(f"        - Power Set Encoding: {label} (binary: {format(label, '04b')})")
                
                # Создание frame-wise меток
                frame_labels = np.full(feature.shape[0], label, dtype=np.int64)
                print(f"        - Frame-wise метки: {frame_labels.shape} (все значения = {label})")
                raw_labels.append(frame_labels)
                
                # Создание frame-wise meeting_ids
                frame_meeting_ids = np.full(feature.shape[0], meeting_id, dtype=object)
                print(f"        - Frame-wise meeting_ids: {frame_meeting_ids.shape}")
                raw_meeting_ids.append(frame_meeting_ids)
                
                speaker_ids.append(speaker_id)
                print(f"        - Добавлено в speaker_ids: {speaker_id}")
        
        # Поиск максимальной длины последовательности
        print(f"\n📏 ПОИСК МАКСИМАЛЬНОЙ ДЛИНЫ ПОСЛЕДОВАТЕЛЬНОСТИ:")
        sequence_lengths = [f.shape[0] for f in raw_features]
        print(f"  - Длины последовательностей: {sequence_lengths}")
        max_len = max(sequence_lengths)
        min_len = min(sequence_lengths)
        mean_len = np.mean(sequence_lengths)
        print(f"  - Минимальная длина: {min_len}")
        print(f"  - Максимальная длина: {max_len}")
        print(f"  - Средняя длина: {mean_len:.1f}")
        print(f"  - Выбранная max_len для padding: {max_len}")
        
        # Второй проход: padding всех признаков до максимальной длины
        print(f"\n🔧 ВТОРОЙ ПРОХОД: PADDING ДО МАКСИМАЛЬНОЙ ДЛИНЫ")
        
        for i, (feature, label, meeting_id_array) in enumerate(zip(raw_features, raw_labels, raw_meeting_ids)):
            print(f"\n  📦 ОБРАБОТКА СЭМПЛА {i + 1}:")
            print(f"    - Исходная длина: {feature.shape[0]} кадров")
            print(f"    - Целевая длина: {max_len} кадров")
            
            if feature.shape[0] < max_len:
                pad_len = max_len - feature.shape[0]
                print(f"    - Необходимо добавить: {pad_len} кадров")
                
                # Padding признаков
                feature_padded = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                print(f"    - Признаки после padding: {feature_padded.shape}")
                print(f"    - Диапазон значений: [{feature_padded.min():.2f}, {feature_padded.max():.2f}]")
                
                # Padding меток
                label_padded = np.pad(label, (0, pad_len), mode='constant', constant_values=-100)
                print(f"    - Метки после padding: {label_padded.shape}")
                print(f"    - Уникальные значения: {np.unique(label_padded)}")
                
                # Padding meeting_ids
                meeting_id_padded = np.pad(meeting_id_array, (0, pad_len), mode='constant', constant_values=None)
                print(f"    - Meeting IDs после padding: {meeting_id_padded.shape}")
                print(f"    - None значений: {np.sum(meeting_id_padded == None)}")
            else:
                print(f"    - Padding не требуется")
                feature_padded = feature
                label_padded = label
                meeting_id_padded = meeting_id_array
            
            features.append(feature_padded)
            labels.append(label_padded)
            meeting_ids.append(meeting_id_padded)
            print(f"    - Добавлено в финальные списки")
        
        # Конвертация в numpy массивы
        print(f"\n🔄 КОНВЕРТАЦИЯ В NUMPY МАССИВЫ:")
        features_array = np.array(features)
        labels_array = np.array(labels)
        meeting_ids_array = np.array(meeting_ids)
        
        print(f"  - features: {features_array.shape} (сэмплы × кадры × mel-каналы)")
        print(f"  - labels: {labels_array.shape} (сэмплы × кадры)")
        print(f"  - meeting_ids: {meeting_ids_array.shape} (сэмплы × кадры)")
        
        # Статистика датасета
        print(f"\n📊 СТАТИСТИКА ДАТАСЕТА:")
        print(f"  - Размер датасета: {features_array.shape[0]} сэмплов")
        print(f"  - Форма признаков: {features_array.shape}")
        print(f"  - Форма меток: {labels_array.shape}")
        print(f"  - Форма meeting_ids: {meeting_ids_array.shape}")
        
        # Анализ меток
        unique_labels = np.unique(labels_array)
        print(f"  - Уникальные значения меток: {unique_labels}")
        
        # Распределение меток (без padding)
        non_padding_mask = labels_array != -100
        non_padding_labels = labels_array[non_padding_mask]
        if len(non_padding_labels) > 0:
            unique_non_padding = np.unique(non_padding_labels)
            print(f"  - Уникальные метки (без padding): {unique_non_padding}")
            
            print(f"  - Распределение меток:")
            for label in unique_non_padding:
                count = np.sum(non_padding_labels == label)
                binary = format(label, f'0{N}b')
                active_speakers = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
                print(f"    {label} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
        
        # Анализ padding
        padding_count = np.sum(labels_array == -100)
        total_frames = labels_array.size
        padding_percentage = (padding_count / total_frames) * 100
        print(f"  - Padding кадры: {padding_count} из {total_frames} ({padding_percentage:.1f}%)")
        
        # Анализ первого сэмпла
        print(f"\n🔍 АНАЛИЗ ПЕРВОГО СЭМПЛА:")
        first_feature = features_array[0]
        first_label = labels_array[0]
        first_meeting_id = meeting_ids_array[0]
        
        print(f"  - Признаки: {first_feature.shape}")
        print(f"    * Первый кадр: {first_feature[0][:5]}...")
        print(f"    * Последний кадр: {first_feature[-1][:5]}...")
        
        print(f"  - Метки: {first_label.shape}")
        non_padding_mask = first_label != -100
        non_padding_count = np.sum(non_padding_mask)
        print(f"    * Non-padding кадры: {non_padding_count} из {len(first_label)}")
        
        if non_padding_count > 0:
            non_padding_labels = first_label[non_padding_mask]
            unique_labels = np.unique(non_padding_labels)
            print(f"    * Уникальные метки: {unique_labels}")
            
            for label in unique_labels:
                count = np.sum(non_padding_labels == label)
                binary = format(label, f'0{N}b')
                active_speakers = [i for i, bit in enumerate(binary[::-1]) if bit == '1']
                print(f"      {label} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
        
        print(f"  - Meeting IDs: {first_meeting_id.shape}")
        unique_meeting_ids = np.unique(first_meeting_id[first_meeting_id != None])
        print(f"    * Уникальные meeting IDs: {unique_meeting_ids}")
        
        return features_array, labels_array, meeting_ids_array

    def analyze_compute_speaker_embeddings_detailed(self, grouped_data: Dict):
        """Подробный анализ функции compute_speaker_embeddings с пошаговым объяснением."""
        print("\n" + "="*80)
        print("🧠 ПОДРОБНЫЙ АНАЛИЗ compute_speaker_embeddings")
        print("="*80)
        
        print(f"\n📋 ЦЕЛЬ ФУНКЦИИ:")
        print(f"  - Извлечь speaker embeddings для каждого уникального спикера")
        print(f"  - Использовать предобученную модель ECAPA-TDNN")
        print(f"  - Создать маппинг speaker_id → embedding_vector")
        print(f"  - Embeddings будут использоваться для контекстно-независимого скоринга")
        
        print(f"\n🔧 ВХОДНЫЕ ДАННЫЕ:")
        print(f"  - grouped_data: {len(grouped_data)} встреч")
        print(f"  - speaker_encoder: предобученная модель ECAPA-TDNN")
        
        # Показываем структуру grouped_data
        print(f"\n🏢 СТРУКТУРА GROUPED_DATA:")
        all_speakers = set()
        for meeting_id, samples in grouped_data.items():
            meeting_speakers = set(sample["speaker_id"] for sample in samples)
            all_speakers.update(meeting_speakers)
            print(f"  Встреча {meeting_id}: {len(samples)} сэмплов, спикеры: {sorted(meeting_speakers)}")
        
        print(f"\n  Всего уникальных спикеров: {len(all_speakers)}")
        print(f"  Спикеры: {sorted(all_speakers)}")
        
        # Шаг 1: Сбор аудио для каждого спикера
        print(f"\n🔄 ШАГ 1: СБОР АУДИО ДЛЯ КАЖДОГО СПИКЕРА")
        print(f"  Цель: Создать словарь speaker_id → список_аудио_сегментов")
        
        speaker_to_audio = {}
        for meeting_id, samples in grouped_data.items():
            print(f"\n  📍 Обработка встречи {meeting_id}:")
            for sample in samples:
                sid = sample["speaker_id"]
                audio_array = sample["audio"]["array"]
                
                if sid not in speaker_to_audio:
                    speaker_to_audio[sid] = []
                    print(f"    🆕 Новый спикер {sid}: создан список")
                
                speaker_to_audio[sid].append(audio_array)
                print(f"    🎵 Добавлено аудио для {sid}: {len(audio_array)} сэмплов "
                      f"({len(audio_array)/16000:.2f}с)")
        
        print(f"\n  📊 РЕЗУЛЬТАТ СБОРА:")
        for sid, audio_list in speaker_to_audio.items():
            total_samples = sum(len(audio) for audio in audio_list)
            total_duration = total_samples / 16000
            print(f"    {sid}: {len(audio_list)} сегментов, {total_samples} сэмплов, {total_duration:.2f}с")
        
        # Шаг 2: Извлечение embeddings
        print(f"\n🧠 ШАГ 2: ИЗВЛЕЧЕНИЕ SPEAKER EMBEDDINGS")
        print(f"  Цель: Для каждого спикера взять первый аудио сегмент и извлечь embedding")
        print(f"  Модель: ECAPA-TDNN (192-мерные векторы)")
        
        speaker_to_embedding = {}
        
        for sid, audio_list in speaker_to_audio.items():
            print(f"\n  🎯 Обработка спикера {sid}:")
            
            # Берем первый аудио сегмент
            audio = audio_list[0]
            print(f"    📥 Входное аудио:")
            print(f"      - Длина: {len(audio)} сэмплов")
            print(f"      - Длительность: {len(audio)/16000:.2f} секунд")
            print(f"      - Диапазон значений: [{audio.min():.4f}, {audio.max():.4f}]")
            print(f"      - Среднее: {audio.mean():.4f}")
            print(f"      - Стандартное отклонение: {audio.std():.4f}")
            
            # Конвертация в тензор
            print(f"    🔄 Конвертация в тензор:")
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, time]
            print(f"      - Форма тензора: {audio_tensor.shape}")
            print(f"      - Тип: {audio_tensor.dtype}")
            print(f"      - Устройство: {audio_tensor.device}")
            
            # Извлечение embedding
            print(f"    🧠 Извлечение embedding через ECAPA-TDNN:")
            with torch.no_grad():
                print(f"      - Вход в модель: {audio_tensor.shape}")
                emb = self.speaker_encoder.encode_batch(audio_tensor)
                print(f"      - Выход модели: {emb.shape}")
                emb = emb.squeeze().cpu()
                print(f"      - После squeeze и cpu: {emb.shape}")
            
            speaker_to_embedding[sid] = emb
            print(f"    ✅ Embedding сохранен для {sid}")
            
            # Анализ embedding
            print(f"    📊 Анализ embedding:")
            print(f"      - Форма: {emb.shape}")
            print(f"      - Тип: {emb.dtype}")
            print(f"      - Диапазон значений: [{emb.min():.4f}, {emb.max():.4f}]")
            print(f"      - Среднее: {emb.mean():.4f}")
            print(f"      - Стандартное отклонение: {emb.std():.4f}")
            print(f"      - Первые 10 значений: {emb[:10].tolist()}")
            print(f"      - Последние 10 значений: {emb[-10:].tolist()}")
        
        # Шаг 3: Анализ результатов
        print(f"\n📊 ШАГ 3: АНАЛИЗ РЕЗУЛЬТАТОВ")
        print(f"  Количество embeddings: {len(speaker_to_embedding)}")
        
        # Проверка размерности
        embedding_dims = [emb.shape[0] for emb in speaker_to_embedding.values()]
        print(f"  Размерности embeddings: {embedding_dims}")
        print(f"  Все embeddings имеют одинаковую размерность: {len(set(embedding_dims)) == 1}")
        
        # Статистика по embeddings
        all_embeddings = torch.stack(list(speaker_to_embedding.values()))
        print(f"\n  📈 СТАТИСТИКА ПО ВСЕМ EMBEDDINGS:")
        print(f"    - Форма объединенного тензора: {all_embeddings.shape}")
        print(f"    - Среднее по всем embeddings: {all_embeddings.mean():.4f}")
        print(f"    - Стандартное отклонение: {all_embeddings.std():.4f}")
        print(f"    - Минимальное значение: {all_embeddings.min():.4f}")
        print(f"    - Максимальное значение: {all_embeddings.max():.4f}")
        
        # Анализ различий между спикерами
        print(f"\n  🔍 АНАЛИЗ РАЗЛИЧИЙ МЕЖДУ СПИКЕРАМИ:")
        if len(speaker_to_embedding) >= 2:
            speaker_ids = list(speaker_to_embedding.keys())
            for i in range(len(speaker_ids)):
                for j in range(i + 1, len(speaker_ids)):
                    sid1, sid2 = speaker_ids[i], speaker_ids[j]
                    emb1, emb2 = speaker_to_embedding[sid1], speaker_to_embedding[sid2]
                    
                    # Евклидово расстояние
                    euclidean_dist = torch.norm(emb1 - emb2).item()
                    
                    # Косинусное сходство
                    cosine_sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                    
                    print(f"    {sid1} vs {sid2}:")
                    print(f"      - Евклидово расстояние: {euclidean_dist:.4f}")
                    print(f"      - Косинусное сходство: {cosine_sim:.4f}")
        
        # Шаг 4: Объяснение использования
        print(f"\n💡 ШАГ 4: ОБЪЯСНЕНИЕ ИСПОЛЬЗОВАНИЯ")
        print(f"  🎯 Зачем нужны speaker embeddings:")
        print(f"    1. Контекстно-независимый скоринг:")
        print(f"       - Модель может сравнивать новые аудио с известными спикерами")
        print(f"       - Не нужно переобучать модель для новых спикеров")
        print(f"    2. Инициализация контекстно-зависимого скоринга:")
        print(f"       - Embeddings используются как начальная точка")
        print(f"       - Модель может адаптировать их под конкретную встречу")
        print(f"    3. Обработка перекрывающейся речи:")
        print(f"       - Каждый спикер имеет уникальный embedding")
        print(f"       - Модель может различать спикеров в перекрывающихся сегментах")
        
        print(f"\n  🔄 Как embeddings попадают в модель:")
        print(f"    1. Создается OverlappingSpeechDataset")
        print(f"    2. Для каждого сэмпла загружаются embeddings всех спикеров встречи")
        print(f"    3. Модель получает features + speaker_embeddings")
        print(f"    4. Модель использует embeddings для предсказания активных спикеров")
        
        print(f"\n  📊 Структура данных в модели:")
        print(f"    - features: [batch_size, sequence_length, 80] - mel-спектрограммы")
        print(f"    - speaker_embeddings: [batch_size, num_speakers, 192] - embeddings спикеров")
        print(f"    - labels: [batch_size, sequence_length] - Power Set Encoded метки")
        
        return speaker_to_embedding

    def analyze_prepare_data_loaders_detailed(self, grouped_train: Dict, grouped_validation: Dict, grouped_test: Dict):
        """Detailed analysis of prepare_data_loaders function with step-by-step explanation."""
        print("\n" + "="*80)
        print("🔧 DETAILED ANALYSIS OF prepare_data_loaders")
        print("="*80)
        
        print(f"\n📋 FUNCTION PURPOSE:")
        print(f"  - Create DataLoaders for training, validation, and testing")
        print(f"  - Prepare data in format suitable for SEND model")
        print(f"  - Create OverlappingSpeechDataset for each split")
        print(f"  - Configure collate_fn for proper batching")
        
        print(f"\n🔧 INPUT DATA:")
        print(f"  - grouped_train: {len(grouped_train)} meetings")
        print(f"  - grouped_validation: {len(grouped_validation)} meetings")
        print(f"  - grouped_test: {len(grouped_test)} meetings")
        print(f"  - speaker_encoder: pre-trained ECAPA-TDNN model")
        print(f"  - batch_size: 4 (default)")
        print(f"  - N: 4 (maximum speakers per recording)")
        
        # Показываем структуру входных данных
        print(f"\n🏢 СТРУКТУРА ВХОДНЫХ ДАННЫХ:")
        for split_name, grouped_data in [("Train", grouped_train), ("Validation", grouped_validation), ("Test", grouped_test)]:
            print(f"  {split_name} split:")
            for meeting_id, samples in grouped_data.items():
                speakers = set(sample["speaker_id"] for sample in samples)
                print(f"    {meeting_id}: {len(samples)} сэмплов, спикеры: {sorted(speakers)}")
        
        # Шаг 1: Создание датасетов через create_dataset_from_grouped
        print(f"\n🔄 ШАГ 1: СОЗДАНИЕ ДАТАСЕТОВ ЧЕРЕЗ create_dataset_from_grouped")
        print(f"  Цель: Преобразовать grouped_data в features, labels, meeting_ids")
        
        print(f"\n  📦 СОЗДАНИЕ TRAIN ДАТАСЕТА:")
        train_features, train_labels, train_meeting_ids = create_dataset_from_grouped(grouped_train, self.speaker_encoder, N=4)
        print(f"    - train_features: {train_features.shape} (сэмплы × кадры × mel-каналы)")
        print(f"    - train_labels: {train_labels.shape} (сэмплы × кадры)")
        print(f"    - train_meeting_ids: {train_meeting_ids.shape} (сэмплы × кадры)")
        
        print(f"\n  📦 СОЗДАНИЕ VALIDATION ДАТАСЕТА:")
        val_features, val_labels, val_meeting_ids = create_dataset_from_grouped(grouped_validation, self.speaker_encoder, N=4)
        print(f"    - val_features: {val_features.shape} (сэмплы × кадры × mel-каналы)")
        print(f"    - val_labels: {val_labels.shape} (сэмплы × кадры)")
        print(f"    - val_meeting_ids: {val_meeting_ids.shape} (сэмплы × кадры)")
        
        print(f"\n  📦 СОЗДАНИЕ TEST ДАТАСЕТА:")
        test_features, test_labels, test_meeting_ids = create_dataset_from_grouped(grouped_test, self.speaker_encoder, N=4)
        print(f"    - test_features: {test_features.shape} (сэмплы × кадры × mel-каналы)")
        print(f"    - test_labels: {test_labels.shape} (сэмплы × кадры)")
        print(f"    - test_meeting_ids: {test_meeting_ids.shape} (сэмплы × кадры)")
        
        # Шаг 2: Создание маппинга speaker_id → index
        print(f"\n🔄 ШАГ 2: СОЗДАНИЕ МАППИНГА SPEAKER_ID → INDEX")
        print(f"  Цель: Создать стабильный маппинг спикеров на индексы для speaker_ids списка")
        
        speaker_to_idx = {}
        print(f"  📍 ОБРАБОТКА TRAIN SPLIT:")
        for meeting_id, samples in grouped_train.items():
            print(f"    Встреча {meeting_id}:")
            for sample in samples:
                speaker_id = sample["speaker_id"]
                if speaker_id not in speaker_to_idx:
                    speaker_to_idx[speaker_id] = len(speaker_to_idx)
                    print(f"      🆕 Новый спикер {speaker_id} → индекс {speaker_to_idx[speaker_id]}")
                else:
                    print(f"      ✅ Спикер {speaker_id} уже существует → индекс {speaker_to_idx[speaker_id]}")
        
        print(f"  📊 ИТОГОВЫЙ МАППИНГ:")
        for speaker_id, idx in speaker_to_idx.items():
            print(f"    {speaker_id} → {idx}")
        
        # Шаг 3: Создание списков speaker_ids
        print(f"\n🔄 ШАГ 3: СОЗДАНИЕ СПИСКОВ SPEAKER_IDS")
        print(f"  Цель: Создать список speaker_ids для каждого сэмпла в каждом split'е")
        
        print(f"\n  📋 СОЗДАНИЕ TRAIN SPEAKER_IDS:")
        train_speaker_ids = []
        for meeting_id, samples in grouped_train.items():
            print(f"    Встреча {meeting_id}:")
            for i, sample in enumerate(samples):
                speaker_id = sample["speaker_id"]
                train_speaker_ids.append(speaker_id)
                print(f"      Сэмпл {i+1}: {speaker_id}")
        print(f"    Итого train_speaker_ids: {len(train_speaker_ids)} элементов")
        print(f"    Первые 5: {train_speaker_ids[:5]}")
        
        print(f"\n  📋 СОЗДАНИЕ VALIDATION SPEAKER_IDS:")
        val_speaker_ids = []
        for meeting_id, samples in grouped_validation.items():
            print(f"    Встреча {meeting_id}:")
            for i, sample in enumerate(samples):
                speaker_id = sample["speaker_id"]
                val_speaker_ids.append(speaker_id)
                print(f"      Сэмпл {i+1}: {speaker_id}")
        print(f"    Итого val_speaker_ids: {len(val_speaker_ids)} элементов")
        print(f"    Первые 5: {val_speaker_ids[:5]}")
        
        print(f"\n  📋 СОЗДАНИЕ TEST SPEAKER_IDS:")
        test_speaker_ids = []
        for meeting_id, samples in grouped_test.items():
            print(f"    Встреча {meeting_id}:")
            for i, sample in enumerate(samples):
                speaker_id = sample["speaker_id"]
                test_speaker_ids.append(speaker_id)
                print(f"      Сэмпл {i+1}: {speaker_id}")
        print(f"    Итого test_speaker_ids: {len(test_speaker_ids)} элементов")
        print(f"    Первые 5: {test_speaker_ids[:5]}")
        
        # Шаг 4: Создание OverlappingSpeechDataset
        print(f"\n🔄 ШАГ 4: СОЗДАНИЕ OVERLAPPINGSPEECHDATASET")
        print(f"  Цель: Создать PyTorch Dataset для каждого split'а")
        
        print(f"\n  📦 СОЗДАНИЕ TRAIN DATASET:")
        print(f"    - features: {train_features.shape}")
        print(f"    - labels: {train_labels.shape}")
        print(f"    - meeting_ids: {train_meeting_ids.shape}")
        print(f"    - speaker_ids: {len(train_speaker_ids)} элементов")
        print(f"    - speaker_to_embedding: будет вычислен через compute_speaker_embeddings")
        print(f"    - max_speakers: 4")
        
        # Вычисляем speaker embeddings для train
        print(f"    🧠 ВЫЧИСЛЕНИЕ SPEAKER EMBEDDINGS ДЛЯ TRAIN:")
        train_speaker_to_embedding = compute_speaker_embeddings(grouped_train, self.speaker_encoder)
        print(f"      - Количество embeddings: {len(train_speaker_to_embedding)}")
        for speaker_id, embedding in train_speaker_to_embedding.items():
            print(f"      - {speaker_id}: {embedding.shape}")
        
        train_dataset = OverlappingSpeechDataset(
            features=train_features,
            labels=train_labels,
            meeting_ids=train_meeting_ids,
            speaker_ids=train_speaker_ids,
            speaker_to_embedding=train_speaker_to_embedding,
            max_speakers=4
        )
        print(f"    ✅ Train dataset создан: {len(train_dataset)} сэмплов")
        print(f"    - speaker_id_list: {train_dataset.speaker_id_list}")
        print(f"    - all_embeddings shape: {train_dataset.all_embeddings.shape}")
        
        print(f"\n  📦 СОЗДАНИЕ VALIDATION DATASET:")
        val_speaker_to_embedding = compute_speaker_embeddings(grouped_validation, self.speaker_encoder)
        val_dataset = OverlappingSpeechDataset(
            features=val_features,
            labels=val_labels,
            meeting_ids=val_meeting_ids,
            speaker_ids=val_speaker_ids,
            speaker_to_embedding=val_speaker_to_embedding,
            max_speakers=4
        )
        print(f"    ✅ Validation dataset создан: {len(val_dataset)} сэмплов")
        
        print(f"\n  📦 СОЗДАНИЕ TEST DATASET:")
        test_speaker_to_embedding = compute_speaker_embeddings(grouped_test, self.speaker_encoder)
        test_dataset = OverlappingSpeechDataset(
            features=test_features,
            labels=test_labels,
            meeting_ids=test_meeting_ids,
            speaker_ids=test_speaker_ids,
            speaker_to_embedding=test_speaker_to_embedding,
            max_speakers=4
        )
        print(f"    ✅ Test dataset создан: {len(test_dataset)} сэмплов")
        
        # Шаг 5: Создание collate_fn
        print(f"\n🔄 ШАГ 5: СОЗДАНИЕ COLLATE_FN")
        print(f"  Цель: Функция для объединения сэмплов в батчи")
        
        def collate_fn(batch):
            print(f"    📦 COLLATE_FN: Обработка батча из {len(batch)} сэмплов")
            
            # Находим максимальную длину последовательности в батче
            max_len = max(x[0].shape[0] for x in batch)
            print(f"      - Максимальная длина последовательности: {max_len}")
            
            features = []
            speaker_embeddings = []
            labels = []
            meeting_ids = []
            
            for i, (feature, all_embeddings, label, meeting_id) in enumerate(batch):
                print(f"      📍 Сэмпл {i+1}:")
                print(f"        - Исходная длина: {feature.shape[0]} кадров")
                print(f"        - Speaker embeddings: {all_embeddings.shape}")
                print(f"        - Labels: {label.shape}")
                print(f"        - Meeting IDs: {meeting_id.shape}")
                
                if feature.shape[0] < max_len:
                    pad_len = max_len - feature.shape[0]
                    print(f"        - Необходимо добавить: {pad_len} кадров padding")
                    
                    # Padding признаков
                    feature = np.pad(feature, ((0, pad_len), (0, 0)), mode='constant')
                    print(f"        - Признаки после padding: {feature.shape}")
                    
                    # Padding meeting_ids
                    meeting_id = np.pad(meeting_id, (0, pad_len), mode='constant', constant_values=None)
                    print(f"        - Meeting IDs после padding: {meeting_id.shape}")
                else:
                    print(f"        - Padding не требуется")
                
                features.append(feature)
                speaker_embeddings.append(all_embeddings)
                labels.append(label)
                meeting_ids.append(meeting_id)
            
            # Конвертация в тензоры
            print(f"      🔄 КОНВЕРТАЦИЯ В ТЕНЗОРЫ:")
            features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
            speaker_embeddings_tensor = torch.stack(speaker_embeddings).float()
            labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
            
            print(f"        - features: {features_tensor.shape}")
            print(f"        - speaker_embeddings: {speaker_embeddings_tensor.shape}")
            print(f"        - labels: {labels_tensor.shape}")
            print(f"        - meeting_ids: {len(meeting_ids)} списков")
            
            return features_tensor, speaker_embeddings_tensor, labels_tensor, meeting_ids
        
        # Шаг 6: Создание DataLoader'ов
        print(f"\n🔄 ШАГ 6: СОЗДАНИЕ DATALOADER'ОВ")
        print(f"  Цель: Создать DataLoader'ы для каждого split'а")
        
        from torch.utils.data import DataLoader
        
        batch_size = 2  # Используем меньший batch_size для демонстрации
        print(f"  📦 BATCH_SIZE: {batch_size}")
        
        print(f"\n  📦 СОЗДАНИЕ TRAIN DATALOADER:")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        print(f"    - Количество батчей: {len(train_loader)}")
        print(f"    - Shuffle: True")
        
        print(f"\n  📦 СОЗДАНИЕ VALIDATION DATALOADER:")
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"    - Количество батчей: {len(val_loader)}")
        print(f"    - Shuffle: False")
        
        print(f"\n  📦 СОЗДАНИЕ TEST DATALOADER:")
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        print(f"    - Количество батчей: {len(test_loader)}")
        print(f"    - Shuffle: False")
        
        # Шаг 7: Анализ первого батча
        print(f"\n🔄 ШАГ 7: АНАЛИЗ ПЕРВОГО БАТЧА")
        print(f"  Цель: Показать структуру данных, которые получает модель")
        
        print(f"\n  📦 АНАЛИЗ ПЕРВОГО TRAIN БАТЧА:")
        train_batch = next(iter(train_loader))
        features, speaker_embeddings, labels, meeting_ids = train_batch
        
        print(f"    - features: {features.shape}")
        print(f"      * Batch size: {features.shape[0]}")
        print(f"      * Sequence length: {features.shape[1]} кадров")
        print(f"      * Feature dimension: {features.shape[2]} mel-каналов")
        print(f"      * Диапазон значений: [{features.min():.4f}, {features.max():.4f}]")
        
        print(f"    - speaker_embeddings: {speaker_embeddings.shape}")
        print(f"      * Batch size: {speaker_embeddings.shape[0]}")
        print(f"      * Number of speakers: {speaker_embeddings.shape[1]}")
        print(f"      * Embedding dimension: {speaker_embeddings.shape[2]}")
        print(f"      * Диапазон значений: [{speaker_embeddings.min():.4f}, {speaker_embeddings.max():.4f}]")
        
        print(f"    - labels: {labels.shape}")
        print(f"      * Batch size: {labels.shape[0]}")
        print(f"      * Sequence length: {labels.shape[1]} кадров")
        print(f"      * Уникальные значения: {torch.unique(labels).tolist()}")
        
        print(f"    - meeting_ids: {len(meeting_ids)} элементов")
        for i, meeting_id_array in enumerate(meeting_ids):
            unique_meeting_ids = np.unique(meeting_id_array[meeting_id_array != None])
            print(f"      * Сэмпл {i}: {meeting_id_array.shape}, meeting_ids: {unique_meeting_ids}")
        
        # Детальный анализ меток в батче
        print(f"\n  🏷️ ДЕТАЛЬНЫЙ АНАЛИЗ МЕТОК В БАТЧЕ:")
        for i in range(features.shape[0]):
            sample_labels = labels[i]
            non_padding_mask = sample_labels != -100
            non_padding_labels = sample_labels[non_padding_mask]
            
            print(f"    Сэмпл {i}:")
            print(f"      - Общее количество кадров: {len(sample_labels)}")
            print(f"      - Non-padding кадры: {len(non_padding_labels)}")
            print(f"      - Padding кадры: {len(sample_labels) - len(non_padding_labels)}")
            
            if len(non_padding_labels) > 0:
                unique_labels = torch.unique(non_padding_labels)
                print(f"      - Уникальные метки (без padding): {unique_labels.tolist()}")
                
                for label in unique_labels:
                    count = torch.sum(non_padding_labels == label).item()
                    binary = format(label.item(), '04b')
                    active_speakers = [j for j, bit in enumerate(binary[::-1]) if bit == '1']
                    print(f"        {label.item()} (binary: {binary}, спикеры: {active_speakers}): {count} кадров")
        
        # Шаг 8: Сравнение с оригинальной функцией
        print(f"\n🔄 ШАГ 8: СРАВНЕНИЕ С ОРИГИНАЛЬНОЙ ФУНКЦИЕЙ")
        print(f"  Цель: Убедиться, что наша реализация соответствует оригиналу")
        
        print(f"\n  📊 ВЫЗОВ ОРИГИНАЛЬНОЙ ФУНКЦИИ:")
        original_train_loader, original_val_loader, original_test_loader = prepare_data_loaders(
            grouped_train, grouped_validation, grouped_test, self.speaker_encoder, batch_size=batch_size, N=4
        )
        
        print(f"    - Оригинальный train_loader: {len(original_train_loader)} батчей")
        print(f"    - Оригинальный val_loader: {len(original_val_loader)} батчей")
        print(f"    - Оригинальный test_loader: {len(original_test_loader)} батчей")
        
        # Сравнение структур
        print(f"\n  🔍 СРАВНЕНИЕ СТРУКТУР:")
        original_batch = next(iter(original_train_loader))
        orig_features, orig_speaker_embeddings, orig_labels, orig_meeting_ids = original_batch
        
        print(f"    - features shapes совпадают: {features.shape == orig_features.shape}")
        print(f"    - speaker_embeddings shapes совпадают: {speaker_embeddings.shape == orig_speaker_embeddings.shape}")
        print(f"    - labels shapes совпадают: {labels.shape == orig_labels.shape}")
        print(f"    - meeting_ids lengths совпадают: {len(meeting_ids) == len(orig_meeting_ids)}")
        
        # Шаг 9: Объяснение использования
        print(f"\n💡 ШАГ 9: ОБЪЯСНЕНИЕ ИСПОЛЬЗОВАНИЯ")
        print(f"  🎯 Зачем нужна функция prepare_data_loaders:")
        print(f"    1. Унификация данных:")
        print(f"       - Преобразует grouped_data в стандартный формат PyTorch")
        print(f"       - Создает одинаковую структуру для train/val/test")
        print(f"    2. Подготовка к обучению:")
        print(f"       - Создает DataLoader'ы с правильным батчингом")
        print(f"       - Настраивает collate_fn для padding")
        print(f"    3. Интеграция с моделью:")
        print(f"       - Данные готовы для подачи в модель SEND")
        print(f"       - Поддерживает как контекстно-независимый, так и контекстно-зависимый скоринг")
        
        print(f"\n  🔄 Как данные попадают в модель:")
        print(f"    1. DataLoader создает батчи")
        print(f"    2. collate_fn объединяет сэмплы и делает padding")
        print(f"    3. Модель получает:")
        print(f"       - features: [batch_size, sequence_length, 80] - mel-спектрограммы")
        print(f"       - speaker_embeddings: [batch_size, num_speakers, 192] - embeddings спикеров")
        print(f"       - labels: [batch_size, sequence_length] - Power Set Encoded метки")
        print(f"       - meeting_ids: [batch_size] списков - идентификаторы встреч")
        
        print(f"\n  📊 Структура данных в модели:")
        print(f"    - Каждый батч содержит сэмплы из разных встреч")
        print(f"    - Все сэмплы в батче имеют одинаковую длину (после padding)")
        print(f"    - Speaker embeddings специфичны для каждой встречи")
        print(f"    - Labels кодируют активных спикеров для каждого кадра")
        
        return train_loader, val_loader, test_loader

    def _create_summary_report(self):
        """Создание итогового отчета."""
        print("\n" + "="*80)
        print("📋 ИТОГОВЫЙ ОТЧЕТ: СТРУКТУРА ВХОДНЫХ ДАННЫХ")
        print("="*80)
        
        report = """
🎯 КРАТКОЕ РЕЗЮМЕ СТРУКТУРЫ ДАННЫХ:

1. ИСХОДНЫЕ ДАННЫЕ (AMI Dataset):
   - Встречи (meetings) содержат аудио сегменты
   - Каждый сегмент имеет: audio, speaker_id, begin_time, end_time
   - Аудио: массив сэмплов с частотой дискретизации 16kHz

2. ПРЕДОБРАБОТКА:
   - Аудио → mel-спектрограммы (80 каналов, 10ms кадры)
   - Спикеры → embeddings (192-мерные векторы из ECAPA-TDNN)
   - Метки → Power Set Encoding (бинарное кодирование активных спикеров)

3. СТРУКТУРА СЭМПЛА:
   - feature: [sequence_length, 80] - mel-спектрограмма
   - speaker_embeddings: [num_speakers, 192] - embeddings всех спикеров
   - label: [sequence_length] - Power Set Encoded метки для каждого кадра
   - meeting_id: [sequence_length] - идентификатор встречи для каждого кадра

4. СТРУКТУРА БАТЧА:
   - features: [batch_size, sequence_length, 80]
   - speaker_embeddings: [batch_size, num_speakers, 192]
   - labels: [batch_size, sequence_length]
   - meeting_ids: [batch_size] списков по sequence_length элементов

5. ВХОД В МОДЕЛЬ:
   - Модель получает features и speaker_embeddings
   - Предсказывает Power Set Encoded метки для каждого кадра
   - Использует контекстно-независимый и контекстно-зависимый скоринг

6. POWER SET ENCODING:
   - Каждый бит в числе представляет активность спикера
   - Например: 5 (binary: 0101) = спикеры 0 и 2 активны
   - Позволяет кодировать перекрывающуюся речь

7. ВАЛИДАЦИЯ:
   - Метки декодируются обратно в активных спикеров
   - Создаются Annotation объекты для pyannote
   - Вычисляется Diarization Error Rate (DER)
        """
        
        print(report)
        
        # Сохранение отчета
        os.makedirs('test_outputs', exist_ok=True)
        with open('test_outputs/data_structure_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("📄 Отчет сохранен: test_outputs/data_structure_report.txt")


def main():
    """Главная функция для запуска анализа."""
    analyzer = DataStructureAnalyzer()
    analyzer.run_complete_analysis(test_size=2)


if __name__ == "__main__":
    main()
