#!/bin/bash
# Скрипт для проверки доступных партиций и GPU

echo "=== Available Partitions ==="
sinfo

echo ""
echo "=== GPU Partitions ==="
sinfo -p gpu

echo ""
echo "=== H200 GPU Nodes ==="
sinfo -N -p gpu-h200 2>/dev/null || echo "gpu-h200 partition not found, checking all GPU partitions..."
sinfo -p gpu

echo ""
echo "=== Cluster Info ==="
clusterinfo -v 2>/dev/null || echo "clusterinfo not available"

echo ""
echo "=== Available GPUs ==="
sinfo -o "%P %G" | grep -i gpu

