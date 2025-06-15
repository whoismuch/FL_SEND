# Setup Guide for FL-SEND-PSE

## 1. Create and activate conda environment

```bash
# Create new environment
conda create -n fl-send python=3.11
conda activate fl-send
```

## 2. Install dependencies through conda

```bash
# Install PyTorch and related packages
conda install pytorch torchaudio torchvision -c pytorch

# Install scientific computing and visualization packages
conda install numpy matplotlib seaborn tqdm scikit-learn

# Install audio processing packages
conda install -c conda-forge librosa soundfile

# Install diarization packages
conda install -c conda-forge pyannote.metrics pyannote.core
```

## 3. Install remaining packages through pip

```bash
# Install federated learning packages
pip install flwr[simulation] ray

# Install speech processing packages
pip install speechbrain datasets
```

## 4. Verify installation

```bash
# Start Python interpreter
python

# Try importing key packages
import torch
import torchaudio
import numpy as np
import librosa
import flwr
import pyannote.metrics
import speechbrain
import datasets

# Check PyTorch CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
```

## 5. Common Issues and Solutions

### CUDA Issues
If you encounter CUDA-related issues:
1. Make sure you have NVIDIA drivers installed
2. Install CUDA toolkit matching your PyTorch version
3. Try reinstalling PyTorch with specific CUDA version:
```bash
conda install pytorch torchaudio torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Audio Processing Issues
If you have problems with librosa or soundfile:
1. Make sure you have system audio libraries installed:
```bash
# For Ubuntu/Debian
sudo apt-get install libsndfile1

# For macOS
brew install libsndfile
```

### Memory Issues
If you encounter memory issues:
1. Reduce batch size in the code
2. Use gradient accumulation
3. Enable memory efficient attention if available

## 6. Running the Code

After setting up the environment, you can run the code:

```bash
# Make sure you're in the correct directory
cd /path/to/FL-SEND

# Run the main script
python FL_SEND_PSE_AMI_improved.py
``` 