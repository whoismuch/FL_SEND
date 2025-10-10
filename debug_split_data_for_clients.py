"""
Функция отладки для split_data_for_clients.

Эта функция демонстрирует работу split_data_for_clients с подробным выводом
промежуточных состояний данных на каждом этапе обработки.
"""

import numpy as np
from datetime import datetime

def debug_split_data_for_clients(grouped_data, num_clients, speaker_encoder, power_set_encoder):
    """
    Отладочная версия функции split_data_for_clients с подробным выводом промежуточных состояний.
    
    Args:
        grouped_data: Dictionary of meeting_id to samples
        num_clients: Number of clients to split data among
        speaker_encoder: Speaker encoder model
        min_overlap_ratio: Minimum ratio of overlapping samples to total samples
    """
    
    print("=" * 80)
    print("ОТЛАДКА ФУНКЦИИ split_data_for_clients")
    print("=" * 80)
    print(f"Время начала: {datetime.now()}")
    print(f"Количество встреч: {len(grouped_data)}")
    print(f"Количество клиентов: {num_clients}")
    print(f"PowerSetEncoder: max_speakers={power_set_encoder.max_speakers}, max_overlap={power_set_encoder.max_overlap}")
    print(f"Количество классов: {power_set_encoder.num_classes}")
    
    # ИНФОРМАЦИЯ О GROUPED_DATA
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ О GROUPED_DATA")
    print("=" * 60)
    print("Структура grouped_data:")
    print("  grouped_data = {")
    print("    'meeting_id': [")
    print("      {")
    print("        'audio': {'array': numpy_array},  # Аудио данные")
    print("        'speaker_id': 'speaker_name',      # ID спикера")
    print("        'begin_time': float,               # Время начала (секунды)")
    print("        'end_time': float                  # Время окончания (секунды)")
    print("      }, ...")
    print("    ], ...")
    print("  }")
    
    print(f"\nДетальная информация о grouped_data:")
    total_segments = 0
    for meeting_id, samples in grouped_data.items():
        print(f"\nВстреча '{meeting_id}':")
        print(f"  Количество сегментов: {len(samples)}")
        total_segments += len(samples)
        
        # Показываем первый элемент встречи
        if samples:
            first_sample = samples[0]
            print(f"  Первый сегмент:")
            print(f"    speaker_id: {first_sample['speaker_id']}")
            print(f"    begin_time: {first_sample['begin_time']}")
            print(f"    end_time: {first_sample['end_time']}")
            print(f"    duration: {first_sample['end_time'] - first_sample['begin_time']:.2f}s")
            if 'audio' in first_sample and 'array' in first_sample['audio']:
                audio_array = first_sample['audio']['array']
                print(f"    audio shape: {audio_array.shape}")
                print(f"    audio dtype: {audio_array.dtype}")
                print(f"    audio min/max: {audio_array.min():.3f}/{audio_array.max():.3f}")
    
    print(f"\nОбщая статистика grouped_data:")
    print(f"  Всего встреч: {len(grouped_data)}")
    print(f"  Всего сегментов: {total_segments}")
    print(f"  Среднее сегментов на встречу: {total_segments/len(grouped_data):.1f}")
    
    # Показываем первый элемент всего grouped_data
    if grouped_data:
        first_meeting_id = list(grouped_data.keys())[0]
        first_meeting_samples = grouped_data[first_meeting_id]
        if first_meeting_samples:
            first_element = first_meeting_samples[0]
            print(f"\nПервый элемент grouped_data:")
            print(f"  Встреча: {first_meeting_id}")
            print(f"  Сегмент: {first_element}")
            print(f"  Типы данных:")
            for key, value in first_element.items():
                if key == 'audio' and isinstance(value, dict) and 'array' in value:
                    print(f"    {key}: dict с 'array' -> numpy.ndarray")
                else:
                    print(f"    {key}: {type(value).__name__}")
    
    # ЭТАП 1: СОЗДАНИЕ МАППИНГА СПИКЕРОВ
    print("\n" + "=" * 60)
    print("ЭТАП 1: СОЗДАНИЕ МАППИНГА СПИКЕРОВ")
    print("=" * 60)
    print("Цель: Создать словарь speaker_to_idx для преобразования ID спикеров в индексы")
    
    speaker_to_idx = {}
    for meeting_id, samples in grouped_data.items():
        print(f"\nОбработка встречи {meeting_id}:")
        for sample in samples:
            speaker_id = sample["speaker_id"]
            if speaker_id not in speaker_to_idx:
                speaker_to_idx[speaker_id] = len(speaker_to_idx)
                print(f"  Добавлен новый спикер: {speaker_id} → индекс {speaker_to_idx[speaker_id]}")
            else:
                print(f"  Спикер уже существует: {speaker_id} → индекс {speaker_to_idx[speaker_id]}")
    
    print(f"\nИтоговый маппинг спикеров: {speaker_to_idx}")
    print(f"Всего уникальных спикеров: {len(speaker_to_idx)}")
    
    # ЭТАП 2: ОБРАБОТКА ВСТРЕЧ ДЛЯ ПОИСКА ПЕРЕКРЫВАЮЩИХСЯ СЕГМЕНТОВ
    print("\n" + "=" * 60)
    print("ЭТАП 2: ОБРАБОТКА ВСТРЕЧ ДЛЯ ПОИСКА ПЕРЕКРЫВАЮЩИХСЯ СЕГМЕНТОВ")
    print("=" * 60)
    print("Цель: Найти естественные перекрытия в аудио и создать комбинированные сегменты")
    
    all_samples = []
    total_original_segments = 0
    total_natural_overlaps = 0
    
    for meeting_id, samples in grouped_data.items():
        print(f"\nОбработка встречи {meeting_id}:")
        print(f"  Количество сегментов: {len(samples)}")
        
        # Сортировка сегментов по времени начала
        samples = sorted(samples, key=lambda x: x["begin_time"])
        print("  Сегменты после сортировки:")
        for i, sample in enumerate(samples):
            print(f"    {i}: {sample['begin_time']:.2f}-{sample['end_time']:.2f}s, спикер {sample['speaker_id']}")
        
        # Добавление оригинальных неперекрывающихся сегментов
        print("\n  Добавление оригинальных сегментов:")
        meeting_original_segments = 0
        for sample in samples:
            # Создание Power Set метки для одного спикера используя PowerSetEncoder
            speaker_idx = speaker_to_idx[sample["speaker_id"]]
            encoded_label = power_set_encoder.encode([speaker_idx])
            decoded_speakers = power_set_encoder.decode(encoded_label)
            
            print(f"    Сегмент {sample['speaker_id']} (индекс {speaker_idx}): → {encoded_label} → спикеры {decoded_speakers}")
            
            all_samples.append({
                "audio": sample["audio"],
                "speaker_id": encoded_label,
                "begin_time": sample["begin_time"],
                "end_time": sample["end_time"],
                "is_overlap": False,
                "original_speaker": sample["speaker_id"]
            })
            meeting_original_segments += 1
        
        # Поиск перекрывающихся сегментов
        print("\n  Поиск перекрывающихся сегментов:")
        meeting_natural_overlaps = 0
        for i in range(len(samples)):
            current = samples[i]
            for j in range(i + 1, len(samples)):
                next_seg = samples[j]
                # Проверка перекрытия
                if next_seg["begin_time"] < current["end_time"]:
                    overlap_begin = max(current["begin_time"], next_seg["begin_time"])
                    overlap_end = min(current["end_time"], next_seg["end_time"])
                    overlap_duration = overlap_end - overlap_begin
                    
                    print(f"    Перекрытие найдено: сегменты {i} и {j}")
                    print(f"      Время перекрытия: {overlap_begin:.2f}-{overlap_end:.2f}s")
                    print(f"      Длительность: {overlap_duration:.2f}s")
                    print(f"      Спикеры: {current['speaker_id']} и {next_seg['speaker_id']}")
                    
                    # Создание Power Set метки для двух спикеров используя PowerSetEncoder
                    current_speaker_idx = speaker_to_idx[current["speaker_id"]]
                    next_speaker_idx = speaker_to_idx[next_seg["speaker_id"]]
                    active_speakers = [current_speaker_idx, next_speaker_idx]
                    encoded_label = power_set_encoder.encode(active_speakers)
                    decoded_speakers = power_set_encoder.decode(encoded_label)
                    
                    print(f"      Power Set метка: спикеры {active_speakers} → {encoded_label} → декодированные {decoded_speakers}")
                    
                    # Создание комбинированного аудио (упрощенная версия)
                    combined_audio = {"array": np.random.randn(16000)}  # Заглушка
                    
                    all_samples.append({
                        "audio": combined_audio,
                        "speaker_id": encoded_label,
                        "begin_time": overlap_begin,
                        "end_time": overlap_end,
                        "is_overlap": True,
                        "original_speakers": [current["speaker_id"], next_seg["speaker_id"]]
                    })
                    meeting_natural_overlaps += 1
        
        total_original_segments += meeting_original_segments
        total_natural_overlaps += meeting_natural_overlaps
        
        print(f"\n  Статистика встречи {meeting_id}:")
        print(f"    Оригинальных сегментов: {meeting_original_segments}")
        print(f"    Естественных перекрытий: {meeting_natural_overlaps}")
    
    # ЭТАП 3: БАЛАНСИРОВКА ДАТАСЕТА
    print("\n" + "=" * 60)
    print("ЭТАП 3: БАЛАНСИРОВКА ДАТАСЕТА")
    print("=" * 60)
    print("Цель: Обеспечить баланс между перекрывающимися и неперекрывающимися сегментами")
    
    non_overlap_samples = [s for s in all_samples if not s["is_overlap"]]
    natural_overlap_samples = [s for s in all_samples if s["is_overlap"]]
    
    print(f"Неперекрывающихся сегментов: {len(non_overlap_samples)}")
    print(f"Естественных перекрытий: {len(natural_overlap_samples)}")
    
    # Используем все доступные образцы (без искусственной балансировки)
    balanced_samples = non_overlap_samples + natural_overlap_samples
    
    print(f"Всего образцов: {len(balanced_samples)}")
    
    # Перемешивание
    np.random.shuffle(balanced_samples)
    print("Данные перемешаны")
    
    # ЭТАП 5: РАЗДЕЛЕНИЕ ДАННЫХ МЕЖДУ КЛИЕНТАМИ
    print("\n" + "=" * 60)
    print("ЭТАП 5: РАЗДЕЛЕНИЕ ДАННЫХ МЕЖДУ КЛИЕНТАМИ")
    print("=" * 60)
    print("Цель: Равномерно распределить данные между клиентами для федеративного обучения")
    
    samples_per_client = len(balanced_samples) // num_clients
    print(f"Образцов на клиента: {samples_per_client}")
    
    client_samples = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_data = balanced_samples[start_idx:end_idx]
        client_samples.append(client_data)
        
        print(f"\nКлиент {i}:")
        print(f"  Индексы: {start_idx}-{end_idx-1}")
        print(f"  Количество образцов: {len(client_data)}")
        
        # Статистика по типам сегментов
        client_non_overlap = len([s for s in client_data if not s["is_overlap"]])
        client_natural_overlap = len([s for s in client_data if s["is_overlap"]])
        
        print(f"  Неперекрывающихся: {client_non_overlap}")
        print(f"  Естественных перекрытий: {client_natural_overlap}")
        
        # Показываем примеры Power Set меток
        print("  Примеры Power Set меток:")
        for j, sample in enumerate(client_data[:3]):  # Показываем первые 3
            print(f"    Образец {j}: {sample['speaker_id']} ({'перекрытие' if sample['is_overlap'] else 'одиночный'})")
    
    # ЭТАП 6: СОЗДАНИЕ ДАТАЛОАДЕРОВ (упрощенная версия)
    print("\n" + "=" * 60)
    print("ЭТАП 6: СОЗДАНИЕ ДАТАЛОАДЕРОВ")
    print("=" * 60)
    print("Цель: Подготовить данные для обучения в формате PyTorch")
    
    print("В реальной версии здесь создаются:")
    print("  1. extract_features(audio) - извлечение признаков")
    print("  2. frame-wise метки - np.full(feature.shape[0], speaker_id)")
    print("  3. Паддинг до максимальной длины")
    print("  4. Разделение на train/validation (80%/20%)")
    print("  5. OverlappingSpeechDataset")
    print("  6. DataLoader с collate_fn")
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Всего встреч: {len(grouped_data)}")
    print(f"Всего уникальных спикеров: {len(speaker_to_idx)}")
    print(f"Всего оригинальных сегментов: {total_original_segments}")
    print(f"Всего естественных перекрытий: {total_natural_overlaps}")
    print(f"Всего образцов: {len(balanced_samples)}")
    print(f"Количество клиентов: {num_clients}")
    print(f"Образцов на клиента: {samples_per_client}")
    
    print(f"\nВремя завершения: {datetime.now()}")
    print("=" * 80)
    
    return client_samples

def demo_debug_function():
    """Демонстрация работы отладочной функции с реальными данными из main функции."""
    
    print("ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ ИЗ ДАТАСЕТА")
    print("=" * 50)
    
    # Импортируем функции для загрузки реальных данных как в основном коде
    from datasets import load_dataset
    from data_processing import group_by_meeting
    import torch
    from speechbrain.pretrained import EncoderClassifier
    from FL_SEND_PSE_AMI_improved import PowerSetEncoder
    
    print("Загружаем реальные данные из AMI датасета...")
    
    # Загружаем AMI датасет как в основном коде
    dataset = load_dataset("edinburghcstr/ami", "ihm")
    print("Dataset loaded successfully")
    
    # Используем небольшой subset для тестирования
    test_size = 100  # Используем то же значение как в main
    print(f"Using test_size: {test_size}")
    
    # Group data by meeting ID for all splits
    print("Grouping data by meeting ID...")
    grouped_train = group_by_meeting(dataset["train"].select(range(test_size)))
    grouped_validation = group_by_meeting(dataset["validation"].select(range(test_size)))
    grouped_test = group_by_meeting(dataset["test"].select(range(test_size)))
    
    print("Реальные данные загружены:")
    print(f"  Train встреч: {len(grouped_train)}")
    print(f"  Validation встреч: {len(grouped_validation)}")
    print(f"  Test встреч: {len(grouped_test)}")
    
    # Используем только train данные для демонстрации
    grouped_data = grouped_train
    
    # Инициализируем реальный speaker_encoder как в основном коде
    print("\nИнициализация реального speaker_encoder...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")
    
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa",
        run_opts={"device": device}
    ).to(device)
    print("Speaker encoder инициализирован успешно")
    
    # Инициализируем PowerSetEncoder как в основном коде
    print("\nИнициализация PowerSetEncoder...")
    
    N = 5  # max_speakers как в main
    K = 3  # max_overlap как в main
    power_set_encoder = PowerSetEncoder(max_speakers=N, max_overlap=K)
    print(f"PowerSetEncoder инициализирован: N={N}, K={K}, классов={power_set_encoder.num_classes}")
    
    print(f"\nИспользуем train данные для демонстрации:")
    print(f"  Встреч: {len(grouped_data)}")
    print(f"  Всего сегментов: {sum(len(samples) for samples in grouped_data.values())}")
    
    # Запускаем отладочную функцию
    result = debug_split_data_for_clients(
        grouped_data=grouped_data,
        num_clients=2,
        speaker_encoder=speaker_encoder,
        power_set_encoder=power_set_encoder
    )
    
    print(f"\nРезультат: {len(result)} клиентов с данными")
    return result

if __name__ == "__main__":
    demo_debug_function()
 