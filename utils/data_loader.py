import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from PIL import Image
import numpy as np

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        # Define paths for real and fake data
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        # Get all image paths and labels
        self.image_paths = []
        self.labels = []
        
        # Load real images
        for img_name in os.listdir(self.real_dir):
            self.image_paths.append(os.path.join(self.real_dir, img_name))
            self.labels.append(0)  # 0 for real
            
        # Load fake images
        for img_name in os.listdir(self.fake_dir):
            self.image_paths.append(os.path.join(self.fake_dir, img_name))
            self.labels.append(1)  # 1 for fake
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if any
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])

def create_data_loaders(train_dir, val_dir, batch_size=32):
    # Create datasets
    train_dataset = DeepfakeDataset(
        train_dir,
        transform=get_transforms(is_train=True),
        is_train=True
    )
    
    val_dataset = DeepfakeDataset(
        val_dir,
        transform=get_transforms(is_train=False),
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        
        # Define paths for real and fake videos
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        # Get all video paths and labels
        self.video_paths = []
        self.labels = []
        
        # Load real videos
        for video_name in os.listdir(self.real_dir):
            self.video_paths.append(os.path.join(self.real_dir, video_name))
            self.labels.append(0)  # 0 for real
            
        # Load fake videos
        for video_name in os.listdir(self.fake_dir):
            self.video_paths.append(os.path.join(self.fake_dir, video_name))
            self.labels.append(1)  # 1 for fake
            
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
        cap.release()
        
        # Sample frames if video is too long
        if len(frames) > 32:  # Sample 32 frames
            indices = np.linspace(0, len(frames)-1, 32, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Apply transforms if any
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed = self.transform(image=frame)
                transformed_frames.append(transformed['image'])
            frames = transformed_frames
        
        # Stack frames
        frames = torch.stack(frames)
        
        return frames, label 