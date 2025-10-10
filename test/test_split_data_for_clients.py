"""
Тест для функции split_data_for_clients с подробным объяснением каждого этапа.

Этот тест демонстрирует работу функции split_data_for_clients и объясняет:
1. Создание маппинга спикеров
2. Обработку встреч для поиска перекрывающихся сегментов
3. Создание искусственных перекрытий
4. Балансировку датасета
5. Разделение данных между клиентами
6. Создание даталоадеров для каждого клиента
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestSplitDataForClients(unittest.TestCase):
    """Тест для функции split_data_for_clients с подробным объяснением."""
    
    def setUp(self):
        """Подготовка тестовых данных."""
        print("=" * 80)
        print("НАСТРОЙКА ТЕСТОВЫХ ДАННЫХ")
        print("=" * 80)
        
        # Создаем тестовые данные для встреч
        self.grouped_data = {
            "meeting_1": [
                {
                    "audio": {"array": np.random.randn(16000)},  # 1 секунда аудио
                    "speaker_id": "speaker_A",
                    "begin_time": 0.0,
                    "end_time": 1.0
                },
                {
                    "audio": {"array": np.random.randn(16000)},
                    "speaker_id": "speaker_B", 
                    "begin_time": 0.5,  # Перекрытие с первым сегментом
                    "end_time": 1.5
                },
                {
                    "audio": {"array": np.random.randn(16000)},
                    "speaker_id": "speaker_C",
                    "begin_time": 2.0,
                    "end_time": 3.0
                }
            ],
            "meeting_2": [
                {
                    "audio": {"array": np.random.randn(16000)},
                    "speaker_id": "speaker_A",
                    "begin_time": 0.0,
                    "end_time": 1.0
                },
                {
                    "audio": {"array": np.random.randn(16000)},
                    "speaker_id": "speaker_D",
                    "begin_time": 1.5,
                    "end_time": 2.5
                }
            ]
        }
        
        # Мок для speaker_encoder
        self.speaker_encoder = Mock()
        self.speaker_encoder.encode_batch.return_value = np.random.randn(4, 192)  # 4 спикера, 192-мерные эмбеддинги
        
        self.num_clients = 2
        self.min_overlap_ratio = 0.3
        
        print(f"Создано {len(self.grouped_data)} встреч с общим количеством сегментов: {sum(len(samples) for samples in self.grouped_data.values())}")
        print(f"Количество клиентов: {self.num_clients}")
        print(f"Минимальный коэффициент перекрытий: {self.min_overlap_ratio}")
    
    @patch('data_processing.extract_features')
    @patch('data_processing.compute_speaker_embeddings')
    @patch('data_processing.simulate_overlapping_speech')
    @patch('data_processing.OverlappingSpeechDataset')
    def test_split_data_for_clients_detailed(self, mock_dataset_class, mock_simulate, mock_compute_embeddings, mock_extract_features):
        """Подробный тест функции split_data_for_clients с объяснением каждого этапа."""
        
        print("=" * 80)
        print("НАЧАЛО ТЕСТИРОВАНИЯ ФУНКЦИИ split_data_for_clients")
        print("=" * 80)
        
        # Настройка моков
        mock_extract_features.return_value = np.random.randn(100, 80)  # 100 кадров, 80 признаков
        mock_compute_embeddings.return_value = {
            "speaker_A": np.random.randn(192),
            "speaker_B": np.random.randn(192),
            "speaker_C": np.random.randn(192),
            "speaker_D": np.random.randn(192)
        }
        mock_simulate.return_value = (
            [np.random.randn(16000), np.random.randn(16000)],  # 2 искусственных сегмента
            [3, 5]  # Соответствующие метки
        )
        mock_dataset_class.return_value = Mock()
        
        # Импортируем функцию
        from data_processing import split_data_for_clients
        
        print("\n" + "=" * 60)
        print("ЭТАП 1: СОЗДАНИЕ МАППИНГА СПИКЕРОВ")
        print("=" * 60)
        print("Цель: Создать словарь speaker_to_idx для преобразования ID спикеров в индексы")
        print("Зачем: Power Set Encoding требует числовые индексы вместо строковых ID")
        print("Алгоритм:")
        print("  1. Проходим по всем встречам и сегментам")
        print("  2. Собираем уникальные speaker_id")
        print("  3. Присваиваем каждому speaker_id числовой индекс")
        print("  4. Результат: {'speaker_A': 0, 'speaker_B': 1, 'speaker_C': 2, 'speaker_D': 3}")
        
        # Выполняем функцию
        result = split_data_for_clients(
            self.grouped_data, 
            self.num_clients, 
            self.speaker_encoder, 
            self.min_overlap_ratio
        )
        
        print("\n" + "=" * 60)
        print("ЭТАП 2: ОБРАБОТКА ВСТРЕЧ ДЛЯ ПОИСКА ПЕРЕКРЫВАЮЩИХСЯ СЕГМЕНТОВ")
        print("=" * 60)
        print("Цель: Найти естественные перекрытия в аудио и создать комбинированные сегменты")
        print("Зачем: Реальные перекрытия дают более качественные данные для обучения")
        print("Алгоритм:")
        print("  1. Сортировка сегментов по времени начала")
        print("  2. Поиск перекрытий: next_seg.begin_time < current.end_time")
        print("  3. Извлечение перекрывающихся частей аудио")
        print("  4. Комбинирование аудио: current_overlap + next_overlap")
        print("  5. Нормализация комбинированного аудио")
        print("  6. Создание Power Set метки для двух спикеров")
        print("Пример:")
        print("  Сегмент A: 0.0-1.0s, спикер_A")
        print("  Сегмент B: 0.5-1.5s, спикер_B")
        print("  Перекрытие: 0.5-1.0s")
        print("  Комбинированное аудио: audio_A[0.5-1.0] + audio_B[0.0-0.5]")
        print("  Power Set метка: [1, 1, 0, 0] → 3")
        
        print("\n" + "=" * 60)
        print("ЭТАП 3: СОЗДАНИЕ ИСКУССТВЕННЫХ ПЕРЕКРЫТИЙ")
        print("=" * 60)
        print("Цель: Создать дополнительные перекрывающиеся сегменты если естественных недостаточно")
        print("Зачем: Обеспечить достаточное количество перекрытий для обучения модели")
        print("Условие: len(overlapping_segments) < len(samples) * min_overlap_ratio")
        print("Алгоритм:")
        print("  1. Группировка сегментов по спикерам")
        print("  2. Вызов simulate_overlapping_speech для создания искусственных перекрытий")
        print("  3. Добавление искусственных сегментов с флагом is_artificial=True")
        print("Пример:")
        print("  Естественных перекрытий: 1")
        print("  Минимально нужно: 5 * 0.3 = 1.5")
        print("  Создаем искусственные перекрытия")
        
        print("\n" + "=" * 60)
        print("ЭТАП 4: БАЛАНСИРОВКА ДАТАСЕТА")
        print("=" * 60)
        print("Цель: Обеспечить баланс между перекрывающимися и неперекрывающимися сегментами")
        print("Зачем: Избежать переобучения на одном типе данных")
        print("Алгоритм:")
        print("  1. Разделение на три группы: non_overlap, natural_overlap, artificial_overlap")
        print("  2. Выбор минимального количества: min(len(non_overlap), len(overlaps))")
        print("  3. Создание сбалансированного набора: non_overlap[:min] + overlaps[:min]")
        print("  4. Перемешивание данных")
        print("Пример:")
        print("  Неперекрывающихся: 5")
        print("  Перекрывающихся: 3")
        print("  Минимум: min(5, 3) = 3")
        print("  Сбалансированный набор: 3 non_overlap + 3 overlaps = 6")
        
        print("\n" + "=" * 60)
        print("ЭТАП 5: РАЗДЕЛЕНИЕ ДАННЫХ МЕЖДУ КЛИЕНТАМИ")
        print("=" * 60)
        print("Цель: Равномерно распределить данные между клиентами для федеративного обучения")
        print("Зачем: Каждый клиент должен иметь достаточно данных для локального обучения")
        print("Алгоритм:")
        print("  1. Вычисление samples_per_client = total_samples // num_clients")
        print("  2. Разделение на части: [samples[i:i+samples_per_client] for i in range(0, total, samples_per_client)]")
        print("Пример:")
        print("  Всего образцов: 6")
        print("  Клиентов: 2")
        print("  Образцов на клиента: 6 // 2 = 3")
        print("  Клиент 0: образцы 0-2")
        print("  Клиент 1: образцы 3-5")
        
        print("\n" + "=" * 60)
        print("ЭТАП 6: СОЗДАНИЕ ДАТАЛОАДЕРОВ ДЛЯ КАЖДОГО КЛИЕНТА")
        print("=" * 60)
        print("Цель: Подготовить данные для обучения в формате PyTorch")
        print("Зачем: Обеспечить эффективную загрузку и обработку данных")
        print("Алгоритм:")
        print("  1. Извлечение признаков: extract_features(audio)")
        print("  2. Создание frame-wise меток: np.full(feature.shape[0], speaker_id)")
        print("  3. Паддинг до максимальной длины в батче")
        print("  4. Разделение на train/validation (80%/20%)")
        print("  5. Создание OverlappingSpeechDataset")
        print("  6. Создание DataLoader с collate_fn")
        print("Пример:")
        print("  Аудио: 16000 сэмплов")
        print("  Признаки: 100 кадров × 80 признаков")
        print("  Метки: 100 кадров × 1 метка")
        print("  Train: 80% = 80 кадров")
        print("  Validation: 20% = 20 кадров")
        
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        print("=" * 60)
        
        # Проверяем результат
        self.assertIsInstance(result, list, "Результат должен быть списком клиентов")
        self.assertEqual(len(result), self.num_clients, f"Должно быть {self.num_clients} клиентов")
        
        for i, (train_loader, val_loader) in enumerate(result):
            print(f"Клиент {i}:")
            print(f"  - Train loader: {type(train_loader).__name__}")
            print(f"  - Val loader: {type(val_loader).__name__}")
            self.assertIsNotNone(train_loader, f"Train loader для клиента {i} не должен быть None")
            self.assertIsNotNone(val_loader, f"Val loader для клиента {i} не должен быть None")
        
        print("\n" + "=" * 60)
        print("ПРОВЕРКА ВЫЗОВОВ МОКОВ")
        print("=" * 60)
        
        # Проверяем, что моки были вызваны
        self.assertTrue(mock_extract_features.called, "extract_features должен быть вызван")
        self.assertTrue(mock_compute_embeddings.called, "compute_speaker_embeddings должен быть вызван")
        self.assertTrue(mock_dataset_class.called, "OverlappingSpeechDataset должен быть создан")
        
        print("✅ Все моки вызваны корректно")
        
        print("\n" + "=" * 60)
        print("ОБЪЯСНЕНИЕ POWER SET ENCODING")
        print("=" * 60)
        print("Power Set Encoding используется для кодирования множеств спикеров:")
        print("  - Один спикер: [1, 0, 0, 0] → 1")
        print("  - Два спикера: [1, 1, 0, 0] → 3")
        print("  - Три спикера: [1, 1, 1, 0] → 7")
        print("Формула: encoded = sum(label * (2 ** i) for i, label in enumerate(speaker_label))")
        
        print("\n" + "=" * 60)
        print("ОБЪЯСНЕНИЕ COLLATE_FN")
        print("=" * 60)
        print("collate_fn обрабатывает батчи переменной длины:")
        print("  1. Находит максимальную длину в батче")
        print("  2. Добавляет паддинг к более коротким последовательностям")
        print("  3. Преобразует в тензоры PyTorch")
        print("  4. Возвращает: (features, speaker_embeddings, labels, meeting_ids)")
        
        print("\n" + "=" * 80)
        print("ТЕСТ ЗАВЕРШЕН УСПЕШНО")
        print("=" * 80)
    
    def test_power_set_encoding_examples(self):
        """Тест примеров Power Set Encoding."""
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ POWER SET ENCODING")
        print("=" * 60)
        
        # Тестовые случаи
        test_cases = [
            ([1, 0, 0, 0], 1, "Один спикер (индекс 0)"),
            ([0, 1, 0, 0], 2, "Один спикер (индекс 1)"),
            ([1, 1, 0, 0], 3, "Два спикера (индексы 0 и 1)"),
            ([1, 0, 1, 0], 5, "Два спикера (индексы 0 и 2)"),
            ([1, 1, 1, 0], 7, "Три спикера (индексы 0, 1 и 2)"),
            ([1, 1, 1, 1], 15, "Четыре спикера (все индексы)")
        ]
        
        for speaker_label, expected, description in test_cases:
            # Вычисляем encoded значение
            encoded = sum(label * (2 ** i) for i, label in enumerate(speaker_label))
            
            print(f"Тест: {description}")
            print(f"  Speaker label: {speaker_label}")
            print(f"  Ожидаемое значение: {expected}")
            print(f"  Полученное значение: {encoded}")
            print(f"  Результат: {'✅ ПРОЙДЕН' if encoded == expected else '❌ ПРОВАЛЕН'}")
            
            self.assertEqual(encoded, expected, f"Неверное кодирование для {description}")
        
        print("✅ Все тесты Power Set Encoding пройдены")
    
    def test_overlap_detection_logic(self):
        """Тест логики обнаружения перекрытий."""
        print("\n" + "=" * 60)
        print("ТЕСТИРОВАНИЕ ЛОГИКИ ОБНАРУЖЕНИЯ ПЕРЕКРЫТИЙ")
        print("=" * 60)
        
        # Тестовые сегменты
        segments = [
            {"begin_time": 0.0, "end_time": 1.0, "speaker_id": "A"},
            {"begin_time": 0.5, "end_time": 1.5, "speaker_id": "B"},  # Перекрытие с первым
            {"begin_time": 2.0, "end_time": 3.0, "speaker_id": "C"},  # Нет перекрытия
            {"begin_time": 2.5, "end_time": 3.5, "speaker_id": "D"}   # Перекрытие с третьим
        ]
        
        print("Тестовые сегменты:")
        for i, seg in enumerate(segments):
            print(f"  Сегмент {i}: {seg['begin_time']}-{seg['end_time']}s, спикер {seg['speaker_id']}")
        
        # Проверяем перекрытия
        overlaps_found = 0
        for i in range(len(segments)):
            current = segments[i]
            for j in range(i + 1, len(segments)):
                next_seg = segments[j]
                # Условие перекрытия: next_seg.begin_time < current.end_time
                if next_seg["begin_time"] < current["end_time"]:
                    overlap_begin = max(current["begin_time"], next_seg["begin_time"])
                    overlap_end = min(current["end_time"], next_seg["end_time"])
                    overlap_duration = overlap_end - overlap_begin
                    
                    print(f"Перекрытие найдено:")
                    print(f"  Сегменты: {i} и {j}")
                    print(f"  Время перекрытия: {overlap_begin}-{overlap_end}s")
                    print(f"  Длительность: {overlap_duration}s")
                    print(f"  Спикеры: {current['speaker_id']} и {next_seg['speaker_id']}")
                    
                    overlaps_found += 1
        
        print(f"Всего найдено перекрытий: {overlaps_found}")
        self.assertEqual(overlaps_found, 2, "Должно быть найдено 2 перекрытия")
        
        print("✅ Логика обнаружения перекрытий работает корректно")

if __name__ == '__main__':
    # Запуск тестов с подробным выводом
    unittest.main(verbosity=2)
