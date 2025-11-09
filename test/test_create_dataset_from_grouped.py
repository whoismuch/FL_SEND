#!/usr/bin/env python3
"""
Подробный тест для функции create_dataset_from_grouped с детальным выводом промежуточной информации.

Этот тест показывает:
1. Как функция обрабатывает grouped_data
2. Как создается маппинг спикеров на слоты
3. Как извлекаются mel-спектрограммы
4. Как работает Power Set Encoding
5. Как происходит padding
6. Как создаются финальные массивы
"""

import os
import sys
import logging
import numpy as np
import torch
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier
from typing import Dict, List, Tuple
import gc

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing import (
    extract_features,
    group_by_meeting,
    create_dataset_from_grouped,
    power_set_encoding
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_create_dataset_from_grouped_detailed(test_size: int = 2, N: int = 4, chunk_size: int = 500):
    """
    Подробный тест функции create_dataset_from_grouped с пошаговым выводом.
    
    Args:
        test_size: Количество встреч для тестирования
        N: Максимальное количество спикеров на запись
        chunk_size: Размер чанка для обработки
    """
    print("\n" + "="*100)
    print("🔍 ПОДРОБНЫЙ ТЕСТ ФУНКЦИИ create_dataset_from_grouped")
    print("="*100)
    
    # ========================================================================
    # ШАГ 1: ПОДГОТОВКА ДАННЫХ
    # ========================================================================
    print("\n" + "="*100)
    print("📥 ШАГ 1: ПОДГОТОВКА ВХОДНЫХ ДАННЫХ")
    print("="*100)
    
    print(f"\n🔧 ПАРАМЕТРЫ ТЕСТА:")
    print(f"  - test_size: {test_size} (количество встреч)")
    print(f"  - N: {N} (максимум спикеров на запись)")
    print(f"  - chunk_size: {chunk_size} (размер чанка для обработки)")
    
    # Загрузка датасета
    print(f"\n📦 ЗАГРУЗКА AMI ДАТАСЕТА:")
    dataset = load_dataset("edinburghcstr/ami", "ihm")
    train_split = dataset["train"].select(range(test_size * 10))  # Берем больше для разнообразия
    
    print(f"  - Загружено {len(train_split)} сэмплов из train split")
    
    # Группировка по встречам
    print(f"\n🏢 ГРУППИРОВКА ПО ВСТРЕЧАМ:")
    grouped_data = group_by_meeting(train_split)
    print(f"  - Количество встреч: {len(grouped_data)}")
    
    # Ограничиваем количество встреч для теста
    if len(grouped_data) > test_size:
        meeting_ids = list(grouped_data.keys())[:test_size]
        grouped_data = {mid: grouped_data[mid] for mid in meeting_ids}
        print(f"  - Ограничено до {test_size} встреч для теста")
    
    # Показываем структуру grouped_data
    print(f"\n📊 СТРУКТУРА GROUPED_DATA:")
    total_samples = 0
    for meeting_id, samples in grouped_data.items():
        speakers = set(sample["speaker_id"] for sample in samples)
        total_samples += len(samples)
        print(f"  Встреча {meeting_id}:")
        print(f"    - Количество сэмплов: {len(samples)}")
        print(f"    - Уникальные спикеры: {sorted(speakers)}")
        print(f"    - Количество спикеров: {len(speakers)}")
        
        # Показываем первые 3 сэмпла
        print(f"    - Первые 3 сэмпла:")
        for i, sample in enumerate(samples[:3]):
            duration = sample["end_time"] - sample["begin_time"]
            audio_len = len(sample["audio"]["array"])
            print(f"      {i+1}. Спикер {sample['speaker_id']}: "
                  f"{sample['begin_time']:.2f}s-{sample['end_time']:.2f}s "
                  f"({duration:.2f}s, {audio_len} сэмплов)")
    
    print(f"\n  ИТОГО: {len(grouped_data)} встреч, {total_samples} сэмплов")
    
    # Инициализация speaker encoder (нужен для функции, но не используется в тесте)
    print(f"\n🧠 ИНИЦИАЛИЗАЦИЯ SPEAKER ENCODER:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  - Устройство: {device}")
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa",
        run_opts={"device": device}
    ).to(device)
    print(f"  - Speaker encoder инициализирован")
    
    # Инициализация PowerSetEncoder (упрощенная версия для демонстрации)
    print(f"\n🔢 ИНИЦИАЛИЗАЦИЯ POWER SET ENCODER:")
    print(f"  - Максимум спикеров: {N}")
    print(f"  - Максимум классов: 2^{N} = {2**N}")
    
    # Создаем простой PowerSetEncoder для демонстрации
    class SimplePowerSetEncoder:
        def __init__(self, max_speakers=4):
            self.max_speakers = max_speakers
            self.num_classes = 2 ** max_speakers
        
        def encode(self, speaker_labels):
            """Простое кодирование: битовая маска"""
            if isinstance(speaker_labels, list):
                return sum(2 ** i for i in speaker_labels)
            else:
                return 2 ** speaker_labels
        
        def decode(self, encoded_value):
            """Простое декодирование: извлечение битов"""
            return [i for i in range(self.max_speakers) if (encoded_value >> i) & 1]
    
    power_set_encoder = SimplePowerSetEncoder(max_speakers=N)
    print(f"  - Power Set Encoder создан")
    
    # ========================================================================
    # ШАГ 2: РУЧНАЯ РЕАЛИЗАЦИЯ (для сравнения)
    # ========================================================================
    print("\n" + "="*100)
    print("🔄 ШАГ 2: РУЧНАЯ РЕАЛИЗАЦИЯ (для понимания процесса)")
    print("="*100)
    
    print(f"\n📋 ОБЪЯСНЕНИЕ АЛГОРИТМА:")
    print(f"  Функция create_dataset_from_grouped выполняет следующие шаги:")
    print(f"  1. Первый проход: извлечение признаков и поиск максимальной длины")
    print(f"  2. Создание маппинга спикеров на слоты для каждой встречи")
    print(f"  3. Второй проход: извлечение признаков, кодирование меток, padding")
    print(f"  4. Создание финальных numpy массивов")
    
    # Собираем информацию о всех сэмплах
    print(f"\n🔄 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБРАБОТКИ:")
    all_samples_info = []
    
    for meeting_id, samples in grouped_data.items():
        print(f"\n  📍 ВСТРЕЧА: {meeting_id}")
        print(f"    - Количество сэмплов: {len(samples)}")
        
        # Создание маппинга спикеров на слоты
        meeting_speakers = list(set(sample["speaker_id"] for sample in samples))
        print(f"    - Уникальные спикеры: {meeting_speakers}")
        
        speaker_to_slot = {}
        for i, speaker_id in enumerate(meeting_speakers[:N]):
            speaker_to_slot[speaker_id] = i
            print(f"      Спикер {speaker_id} → слот {i}")
        
        if len(meeting_speakers) > N:
            print(f"      ⚠️ Внимание: {len(meeting_speakers)} спикеров, но только первые {N} будут использованы")
        
        print(f"    - Итоговый маппинг: {speaker_to_slot}")
        
        # Сохраняем информацию о сэмплах
        for sample in samples:
            all_samples_info.append({
                'meeting_id': meeting_id,
                'sample': sample,
                'speaker_to_slot': speaker_to_slot
            })
    
    print(f"\n  ИТОГО: {len(all_samples_info)} сэмплов подготовлено для обработки")
    
    # ========================================================================
    # ШАГ 3: ПЕРВЫЙ ПРОХОД - ПОИСК МАКСИМАЛЬНОЙ ДЛИНЫ
    # ========================================================================
    print("\n" + "="*100)
    print("🔍 ШАГ 3: ПЕРВЫЙ ПРОХОД - ПОИСК МАКСИМАЛЬНОЙ ДЛИНЫ")
    print("="*100)
    
    print(f"\n📏 ЦЕЛЬ: Найти максимальную длину последовательности среди всех сэмплов")
    print(f"  Это нужно для предварительного выделения памяти и определения размера padding")
    
    max_len = 0
    feature_dim = None
    sequence_lengths = []
    
    print(f"\n🔄 ОБРАБОТКА СЭМПЛОВ (в чанках по {chunk_size}):")
    
    total_samples = len(all_samples_info)
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_info = all_samples_info[chunk_start:chunk_end]
        
        print(f"\n  📦 ЧАНК {chunk_start // chunk_size + 1} (сэмплы {chunk_start}-{chunk_end-1}):")
        
        for local_idx, item in enumerate(chunk_info):
            global_idx = chunk_start + local_idx
            
            # Извлечение признаков
            audio_array = item['sample']["audio"]["array"]
            feature = extract_features(audio_array)
            
            if feature_dim is None:
                feature_dim = feature.shape[1]
                print(f"    - Определена размерность признаков: {feature_dim} mel-каналов")
            
            seq_len = feature.shape[0]
            sequence_lengths.append(seq_len)
            max_len = max(max_len, seq_len)
            
            if global_idx < 5 or global_idx == total_samples - 1:  # Показываем первые 5 и последний
                print(f"    - Сэмпл {global_idx + 1}: длина = {seq_len} кадров "
                      f"({seq_len * 0.01:.2f} секунд)")
            
            del feature
        
        # Прогресс
        if chunk_start % (chunk_size * 10) == 0 or chunk_end == total_samples:
            print(f"    📊 Прогресс: {chunk_end}/{total_samples} сэмплов обработано, "
                  f"текущая max_len: {max_len}")
        
        gc.collect()
    
    # Статистика по длинам
    print(f"\n📊 СТАТИСТИКА ПО ДЛИНАМ ПОСЛЕДОВАТЕЛЬНОСТЕЙ:")
    print(f"  - Минимальная длина: {min(sequence_lengths)} кадров ({min(sequence_lengths) * 0.01:.2f} сек)")
    print(f"  - Максимальная длина: {max_len} кадров ({max_len * 0.01:.2f} сек)")
    print(f"  - Средняя длина: {np.mean(sequence_lengths):.1f} кадров ({np.mean(sequence_lengths) * 0.01:.2f} сек)")
    print(f"  - Медианная длина: {np.median(sequence_lengths):.1f} кадров ({np.median(sequence_lengths) * 0.01:.2f} сек)")
    print(f"  - Размерность признаков: {feature_dim} mel-каналов")
    
    # Оценка памяти
    print(f"\n💾 ОЦЕНКА ПАМЯТИ:")
    features_memory_gb = (total_samples * max_len * feature_dim * 4) / (1024**3)  # float32
    labels_memory_gb = (total_samples * max_len * 8) / (1024**3)  # int64
    meeting_ids_memory_gb = (total_samples * max_len * 8) / (1024**3)  # object pointer
    total_memory_gb = features_memory_gb + labels_memory_gb + meeting_ids_memory_gb
    
    print(f"  - Features: {features_memory_gb:.2f} GB")
    print(f"  - Labels: {labels_memory_gb:.2f} GB")
    print(f"  - Meeting IDs: {meeting_ids_memory_gb:.2f} GB")
    print(f"  - ИТОГО: {total_memory_gb:.2f} GB")
    
    # ========================================================================
    # ШАГ 4: ВТОРОЙ ПРОХОД - ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ И СОЗДАНИЕ МЕТОК
    # ========================================================================
    print("\n" + "="*100)
    print("🔄 ШАГ 4: ВТОРОЙ ПРОХОД - ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ И СОЗДАНИЕ МЕТОК")
    print("="*100)
    
    print(f"\n📋 ЦЕЛЬ: Извлечь признаки, создать метки с Power Set Encoding, выполнить padding")
    
    # Предварительное выделение памяти
    print(f"\n💾 ПРЕДВАРИТЕЛЬНОЕ ВЫДЕЛЕНИЕ ПАМЯТИ:")
    print(f"  - features: shape ({total_samples}, {max_len}, {feature_dim}), dtype=float32")
    print(f"  - labels: shape ({total_samples}, {max_len}), dtype=int64, fill_value=-100")
    print(f"  - meeting_ids: shape ({total_samples}, {max_len}), dtype=object, fill_value=None")
    
    try:
        features = np.zeros((total_samples, max_len, feature_dim), dtype=np.float32)
        labels = np.full((total_samples, max_len), -100, dtype=np.int64)
        meeting_ids = np.empty((total_samples, max_len), dtype=object)
        print(f"  ✅ Память успешно выделена")
    except MemoryError as e:
        print(f"  ❌ Ошибка выделения памяти: {e}")
        raise
    
    # Обработка сэмплов
    print(f"\n🔄 ОБРАБОТКА СЭМПЛОВ:")
    
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_info = all_samples_info[chunk_start:chunk_end]
        
        for local_idx, item in enumerate(chunk_info):
            global_idx = chunk_start + local_idx
            
            # Извлечение признаков
            audio_array = item['sample']["audio"]["array"]
            feature = extract_features(audio_array)
            speaker_to_slot = item['speaker_to_slot']
            meeting_id = item['meeting_id']
            sample = item['sample']
            
            # Обрезка, если длиннее max_len (не должно происходить)
            if feature.shape[0] > max_len:
                print(f"    ⚠️ Сэмпл {global_idx + 1}: обрезка с {feature.shape[0]} до {max_len}")
                feature = feature[:max_len, :]
            
            # Маппинг спикера на слот и кодирование
            speaker_id = sample["speaker_id"]
            if speaker_id in speaker_to_slot:
                slot_idx = speaker_to_slot[speaker_id]
                label = power_set_encoder.encode([slot_idx])
            else:
                print(f"    ⚠️ Сэмпл {global_idx + 1}: спикер {speaker_id} не в топ-{N}, назначаем слот 0")
                label = power_set_encoder.encode([0])
            
            # Сохранение данных
            seq_len = feature.shape[0]
            features[global_idx, :seq_len, :] = feature
            labels[global_idx, :seq_len] = label
            meeting_ids[global_idx, :seq_len] = meeting_id
            
            # Детальный вывод для первых 5 сэмплов
            if global_idx < 5:
                print(f"\n    📦 СЭМПЛ {global_idx + 1}:")
                print(f"      - Встреча: {meeting_id}")
                print(f"      - Спикер ID: {speaker_id}")
                print(f"      - Слот: {slot_idx if speaker_id in speaker_to_slot else 0}")
                print(f"      - Power Set Encoding: {label} (binary: {format(label, f'0{N}b')})")
                print(f"      - Декодирование: {power_set_encoder.decode(label)}")
                print(f"      - Длина последовательности: {seq_len} кадров")
                print(f"      - Форма признаков: {feature.shape}")
                print(f"      - Диапазон признаков: [{feature.min():.2f}, {feature.max():.2f}]")
                print(f"      - Padding: {max_len - seq_len} кадров (значение -100 для labels)")
            
            del feature
            
            # Прогресс
            if (global_idx + 1) % 100 == 0 or global_idx == total_samples - 1:
                progress = 100 * (global_idx + 1) / total_samples
                print(f"    📊 Прогресс: {global_idx + 1}/{total_samples} ({progress:.1f}%)")
        
        gc.collect()
    
    print(f"\n✅ Обработка завершена")
    
    # ========================================================================
    # ШАГ 5: АНАЛИЗ РЕЗУЛЬТАТОВ
    # ========================================================================
    print("\n" + "="*100)
    print("📊 ШАГ 5: АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*100)
    
    print(f"\n📦 ФИНАЛЬНЫЕ МАССИВЫ:")
    print(f"  - features: {features.shape} (сэмплы × кадры × mel-каналы)")
    print(f"  - labels: {labels.shape} (сэмплы × кадры)")
    print(f"  - meeting_ids: {meeting_ids.shape} (сэмплы × кадры)")
    
    # Анализ padding
    print(f"\n🔧 АНАЛИЗ PADDING:")
    padding_mask = labels == -100
    padding_count = np.sum(padding_mask)
    total_frames = labels.size
    padding_percentage = 100 * padding_count / total_frames
    
    print(f"  - Всего кадров: {total_frames}")
    print(f"  - Padding кадров: {padding_count} ({padding_percentage:.1f}%)")
    print(f"  - Реальных кадров: {total_frames - padding_count} ({100 - padding_percentage:.1f}%)")
    
    # Анализ меток
    print(f"\n🏷️ АНАЛИЗ МЕТОК:")
    non_padding_labels = labels[~padding_mask]
    unique_labels = np.unique(non_padding_labels)
    
    print(f"  - Уникальные метки (без padding): {unique_labels}")
    print(f"  - Количество уникальных меток: {len(unique_labels)}")
    
    print(f"\n  Распределение меток:")
    for label in sorted(unique_labels):
        count = np.sum(non_padding_labels == label)
        percentage = 100 * count / len(non_padding_labels)
        binary = format(label, f'0{N}b')
        decoded = power_set_encoder.decode(label)
        print(f"    {label:3d} (binary: {binary}, спикеры: {decoded}): "
              f"{count:6d} кадров ({percentage:5.1f}%)")
    
    # Анализ по встречам
    print(f"\n🏢 АНАЛИЗ ПО ВСТРЕЧАМ:")
    for meeting_id in grouped_data.keys():
        meeting_mask = meeting_ids == meeting_id
        meeting_frames = np.sum(meeting_mask)
        meeting_samples = np.sum(meeting_mask.any(axis=1))
        
        print(f"  Встреча {meeting_id}:")
        print(f"    - Сэмплов: {meeting_samples}")
        print(f"    - Кадров: {meeting_frames}")
        
        # Уникальные метки в этой встрече
        meeting_labels = labels[meeting_mask]
        meeting_unique_labels = np.unique(meeting_labels[meeting_labels != -100])
        print(f"    - Уникальные метки: {meeting_unique_labels}")
    
    # Анализ первого сэмпла
    print(f"\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПЕРВОГО СЭМПЛА:")
    first_feature = features[0]
    first_label = labels[0]
    first_meeting_id = meeting_ids[0]
    
    non_padding_mask = first_label != -100
    non_padding_count = np.sum(non_padding_mask)
    
    print(f"  - Признаки: {first_feature.shape}")
    print(f"    * Реальных кадров: {non_padding_count}")
    print(f"    * Padding кадров: {len(first_label) - non_padding_count}")
    print(f"    * Первый кадр: {first_feature[0][:5]}...")
    print(f"    * Последний реальный кадр: {first_feature[non_padding_count-1][:5]}...")
    
    print(f"  - Метки: {first_label.shape}")
    non_padding_labels = first_label[non_padding_mask]
    unique_first_labels = np.unique(non_padding_labels)
    print(f"    * Уникальные метки: {unique_first_labels}")
    for label in unique_first_labels:
        count = np.sum(non_padding_labels == label)
        binary = format(label, f'0{N}b')
        decoded = power_set_encoder.decode(label)
        print(f"      {label} (binary: {binary}, спикеры: {decoded}): {count} кадров")
    
    print(f"  - Meeting IDs: {first_meeting_id.shape}")
    unique_meeting_ids = np.unique(first_meeting_id[first_meeting_id != None])
    print(f"    * Уникальные meeting IDs: {unique_meeting_ids}")
    
    # ========================================================================
    # ШАГ 6: СРАВНЕНИЕ С ОРИГИНАЛЬНОЙ ФУНКЦИЕЙ
    # ========================================================================
    print("\n" + "="*100)
    print("🔄 ШАГ 6: СРАВНЕНИЕ С ОРИГИНАЛЬНОЙ ФУНКЦИЕЙ")
    print("="*100)
    
    print(f"\n📋 ВЫЗОВ ОРИГИНАЛЬНОЙ ФУНКЦИИ:")
    print(f"  create_dataset_from_grouped(grouped_data, speaker_encoder, power_set_encoder, "
          f"N={N}, chunk_size={chunk_size})")
    
    # Нужно создать правильный PowerSetEncoder
    from SEND_PSE_AMI import PowerSetEncoder
    real_power_set_encoder = PowerSetEncoder(max_speakers=N, max_overlap=1)
    
    original_features, original_labels, original_meeting_ids = create_dataset_from_grouped(
        grouped_data, speaker_encoder, real_power_set_encoder, N=N, chunk_size=chunk_size
    )
    
    print(f"\n📊 РЕЗУЛЬТАТЫ ОРИГИНАЛЬНОЙ ФУНКЦИИ:")
    print(f"  - features: {original_features.shape}")
    print(f"  - labels: {original_labels.shape}")
    print(f"  - meeting_ids: {original_meeting_ids.shape}")
    
    # Сравнение
    print(f"\n🔍 СРАВНЕНИЕ:")
    shapes_match = (features.shape == original_features.shape and
                   labels.shape == original_labels.shape and
                   meeting_ids.shape == original_meeting_ids.shape)
    print(f"  - Формы совпадают: {shapes_match}")
    
    if shapes_match:
        # Сравнение значений (только для первых нескольких сэмплов)
        print(f"\n  Сравнение значений (первые 3 сэмпла):")
        for i in range(min(3, len(features))):
            feature_match = np.allclose(features[i], original_features[i], atol=1e-5)
            label_match = np.array_equal(labels[i], original_labels[i])
            meeting_id_match = np.array_equal(meeting_ids[i], original_meeting_ids[i])
            
            print(f"    Сэмпл {i+1}:")
            print(f"      - Features совпадают: {feature_match}")
            print(f"      - Labels совпадают: {label_match}")
            print(f"      - Meeting IDs совпадают: {meeting_id_match}")
    
    # ========================================================================
    # ШАГ 7: ИТОГОВОЕ РЕЗЮМЕ
    # ========================================================================
    print("\n" + "="*100)
    print("📋 ИТОГОВОЕ РЕЗЮМЕ")
    print("="*100)
    
    summary = f"""
🎯 ЧТО ДЕЛАЕТ ФУНКЦИЯ create_dataset_from_grouped:

1. ВХОДНЫЕ ДАННЫЕ:
   - grouped_data: словарь meeting_id → список сэмплов
   - speaker_encoder: модель для извлечения speaker embeddings (не используется напрямую)
   - power_set_encoder: кодировщик для Power Set Encoding
   - N: максимум спикеров на запись (по умолчанию 4)
   - chunk_size: размер чанка для обработки (по умолчанию 500)

2. ПРОЦЕСС ОБРАБОТКИ:
   a) Первый проход:
      - Извлекает mel-спектрограммы для всех сэмплов
      - Находит максимальную длину последовательности
      - Определяет размерность признаков
      - Оценивает требования к памяти
   
   b) Создание маппинга спикеров:
      - Для каждой встречи создает маппинг speaker_id → slot (0 до N-1)
      - Только первые N спикеров получают уникальные слоты
      - Остальные спикеры назначаются в слот 0
   
   c) Второй проход:
      - Извлекает mel-спектрограммы для всех сэмплов
      - Маппит спикера на слот
      - Кодирует метку через Power Set Encoding
      - Создает frame-wise метки и meeting_ids
      - Выполняет padding до максимальной длины
   
   d) Создание финальных массивов:
      - features: [num_samples, max_len, feature_dim] - mel-спектрограммы
      - labels: [num_samples, max_len] - Power Set Encoded метки (-100 для padding)
      - meeting_ids: [num_samples, max_len] - идентификаторы встреч (None для padding)

3. POWER SET ENCODING:
   - Каждый бит в числе представляет активность спикера в слоте
   - Например, для N=4:
     * 1 (0001) = только спикер в слоте 0 активен
     * 2 (0010) = только спикер в слоте 1 активен
     * 5 (0101) = спикеры в слотах 0 и 2 активны
   - Позволяет кодировать перекрывающуюся речь

4. PADDING:
   - Все последовательности дополняются до максимальной длины
   - Padding для features: нули
   - Padding для labels: -100 (специальное значение для игнорирования в loss)
   - Padding для meeting_ids: None

5. ОПТИМИЗАЦИЯ ПАМЯТИ:
   - Обработка в чанках для снижения пикового использования памяти
   - Предварительное выделение массивов для избежания фрагментации
   - Периодическая сборка мусора

6. РЕЗУЛЬТАТ:
   - Три numpy массива готовых для использования в PyTorch Dataset
   - Все сэмплы имеют одинаковую длину (после padding)
   - Метки кодируют активных спикеров для каждого кадра
   - Meeting IDs позволяют группировать кадры по встречам для вычисления DER
"""
    
    print(summary)
    
    return features, labels, meeting_ids


def main():
    """Главная функция для запуска теста."""
    print("🚀 ЗАПУСК ПОДРОБНОГО ТЕСТА create_dataset_from_grouped")
    
    try:
        features, labels, meeting_ids = test_create_dataset_from_grouped_detailed(
            test_size=2,  # Количество встреч для теста
            N=4,          # Максимум спикеров на запись
            chunk_size=500  # Размер чанка
        )
        
        print("\n" + "="*100)
        print("✅ ТЕСТ ЗАВЕРШЕН УСПЕШНО!")
        print("="*100)
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении теста: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

