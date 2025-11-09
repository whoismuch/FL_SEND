#!/usr/bin/env python3
"""
Скрипт для анализа распределения спикеров в датасете.

Показывает:
- Сколько встреч имеет N спикеров
- Максимальное количество спикеров
- Рекомендации по выбору N
"""

import sys
import os
from collections import Counter
from datasets import load_dataset

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing import group_by_meeting


def analyze_speaker_distribution(dataset_name="edinburghcstr/ami", split="train", max_meetings=None):
    """
    Анализирует распределение спикеров в датасете.
    
    Args:
        dataset_name: Имя датасета
        split: Раздел датасета (train, validation, test)
        max_meetings: Максимальное количество встреч для анализа (None = все)
    """
    print("="*80)
    print("📊 АНАЛИЗ РАСПРЕДЕЛЕНИЯ СПИКЕРОВ В ДАТАСЕТЕ")
    print("="*80)
    
    # Загрузка датасета
    print(f"\n📥 Загрузка датасета: {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, "ihm")
    split_data = dataset[split]
    
    if max_meetings:
        split_data = split_data.select(range(min(max_meetings, len(split_data))))
        print(f"   Ограничено до {len(split_data)} сэмплов")
    
    # Группировка по встречам
    print(f"\n🏢 Группировка по встречам...")
    grouped_data = group_by_meeting(split_data)
    print(f"   Найдено {len(grouped_data)} встреч")
    
    # Анализ количества спикеров в каждой встрече
    print(f"\n📊 АНАЛИЗ КОЛИЧЕСТВА СПИКЕРОВ:")
    speaker_counts = []
    speaker_count_distribution = Counter()
    
    for meeting_id, samples in grouped_data.items():
        speakers = set(sample["speaker_id"] for sample in samples)
        num_speakers = len(speakers)
        speaker_counts.append(num_speakers)
        speaker_count_distribution[num_speakers] += 1
    
    # Статистика
    print(f"\n   Статистика:")
    print(f"   - Минимум спикеров: {min(speaker_counts)}")
    print(f"   - Максимум спикеров: {max(speaker_counts)}")
    print(f"   - Среднее количество спикеров: {sum(speaker_counts) / len(speaker_counts):.2f}")
    print(f"   - Медианное количество спикеров: {sorted(speaker_counts)[len(speaker_counts) // 2]}")
    
    # Распределение
    print(f"\n   Распределение по количеству спикеров:")
    print(f"   {'Кол-во спикеров':<20} {'Кол-во встреч':<20} {'Процент':<20}")
    print(f"   {'-'*60}")
    
    total_meetings = len(grouped_data)
    for num_speakers in sorted(speaker_count_distribution.keys()):
        count = speaker_count_distribution[num_speakers]
        percentage = 100 * count / total_meetings
        print(f"   {num_speakers:<20} {count:<20} {percentage:>6.2f}%")
    
    # Визуализация
    print(f"\n   Визуализация (каждая '#' = 1%):")
    for num_speakers in sorted(speaker_count_distribution.keys()):
        count = speaker_count_distribution[num_speakers]
        percentage = 100 * count / total_meetings
        bar = '#' * int(percentage)
        print(f"   {num_speakers} спикеров: {bar} {percentage:.1f}%")
    
    # Рекомендации
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    max_speakers = max(speaker_counts)
    
    print(f"\n   1. Текущее ограничение N=4:")
    meetings_with_4_or_less = sum(count for n, count in speaker_count_distribution.items() if n <= 4)
    meetings_with_more_than_4 = total_meetings - meetings_with_4_or_less
    percentage_covered = 100 * meetings_with_4_or_less / total_meetings
    percentage_lost = 100 * meetings_with_more_than_4 / total_meetings
    
    print(f"      - Покрывает {meetings_with_4_or_less}/{total_meetings} встреч ({percentage_covered:.1f}%)")
    print(f"      - Теряет информацию о {meetings_with_more_than_4} встречах ({percentage_lost:.1f}%)")
    
    if meetings_with_more_than_4 > 0:
        print(f"      ⚠️  ВНИМАНИЕ: {meetings_with_more_than_4} встреч имеют >4 спикеров!")
        print(f"         Эти встречи будут обработаны с потерей информации.")
    
    print(f"\n   2. Рекомендуемое значение N:")
    print(f"      - Минимум: N = {max_speakers} (покрывает все встречи)")
    
    # Вычисление количества классов для разных N
    print(f"\n   3. Количество классов для разных N:")
    print(f"      {'N':<10} {'Классов (K=N)':<20} {'Классов (K=2)':<20} {'Классов (K=3)':<20}")
    print(f"      {'-'*70}")
    
    for n in range(1, min(max_speakers + 2, 9)):
        from math import comb
        classes_k_n = sum(comb(n, k) for k in range(n + 1))
        classes_k_2 = sum(comb(n, k) for k in range(3))  # k=0,1,2
        classes_k_3 = sum(comb(n, k) for k in range(4))  # k=0,1,2,3
        
        marker = " ← Рекомендуется" if n == max_speakers else ""
        print(f"      {n:<10} {classes_k_n:<20} {classes_k_2:<20} {classes_k_3:<20}{marker}")
    
    # Детальная информация о встречах с >4 спикерами
    if meetings_with_more_than_4 > 0:
        print(f"\n   4. Встречи с >4 спикерами:")
        for meeting_id, samples in grouped_data.items():
            speakers = set(sample["speaker_id"] for sample in samples)
            num_speakers = len(speakers)
            if num_speakers > 4:
                print(f"      - {meeting_id}: {num_speakers} спикеров {sorted(speakers)}")
    
    # Итоговая рекомендация
    print(f"\n✅ ИТОГОВАЯ РЕКОМЕНДАЦИЯ:")
    if max_speakers <= 4:
        print(f"   Текущее значение N=4 подходит для всех встреч.")
        print(f"   Нет необходимости увеличивать N.")
    else:
        print(f"   Рекомендуется увеличить N до {max_speakers}.")
        print(f"   Это потребует:")
        from math import comb
        new_num_classes = sum(comb(max_speakers, k) for k in range(max_speakers + 1))
        print(f"   - Пересоздать PowerSetEncoder с max_speakers={max_speakers}")
        print(f"   - Пересоздать модель с num_classes={new_num_classes} (вместо 16)")
        print(f"   - Переобучить модель")
        print(f"   - Это покроет 100% встреч без потери информации")
    
    return {
        'max_speakers': max_speakers,
        'distribution': dict(speaker_count_distribution),
        'total_meetings': total_meetings,
        'meetings_with_more_than_4': meetings_with_more_than_4
    }


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ распределения спикеров в датасете')
    parser.add_argument('--dataset', type=str, default='edinburghcstr/ami',
                        help='Имя датасета (по умолчанию: edinburghcstr/ami)')
    parser.add_argument('--split', type=str, default='train',
                        help='Раздел датасета (по умолчанию: train)')
    parser.add_argument('--max-meetings', type=int, default=None,
                        help='Максимальное количество встреч для анализа (по умолчанию: все)')
    
    args = parser.parse_args()
    
    analyze_speaker_distribution(
        dataset_name=args.dataset,
        split=args.split,
        max_meetings=args.max_meetings
    )


if __name__ == "__main__":
    main()

