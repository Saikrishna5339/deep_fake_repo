 HEAD
# deep_fake_repo

# Deepfake Detection using GenConViT

This project implements a deepfake detection system using GenConViT (Generative Convolutional Vision Transformer) architecture. The system can detect both deepfake images and videos.

## Project Structure
```
deepfake_detection/
├── data/                   # Dataset directory
├── models/                 # Model architecture
├── utils/                  # Utility functions
├── train.py               # Training script
├── test.py                # Testing script
└── requirements.txt       # Project dependencies
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
- For images: Use the UADFV dataset or Celeb-DF dataset
- For videos: Use the Deepfake Detection Challenge (DFDC) dataset

4. Train the model:
```bash
python train.py
```

5. Test the model:
```bash
python test.py
```

## Dataset Information

This project uses the following datasets:
1. UADFV Dataset: Contains real and deepfake videos/images
2. Celeb-DF Dataset: High-quality deepfake videos
3. DFDC Dataset: Large-scale deepfake detection challenge dataset

## Model Architecture

The GenConViT architecture combines:
- Convolutional layers for feature extraction
- Vision Transformer for sequence modeling
- Generative adversarial components for better feature learning

## Performance

The model achieves:
- Image detection accuracy: ~95%
- Video detection accuracy: ~92%
- Real-time processing capability 
 ac20d49 (Normalize line endings)
