import os
import requests
import zipfile
import shutil
from tqdm import tqdm
import sys

def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        sys.exit(1)

def create_directory_structure():
    # Create main directories
    directories = [
        'data/train/real',
        'data/train/fake',
        'data/val/real',
        'data/val/fake',
        'data/test/real',
        'data/test/fake'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def download_sample_dataset():
    print("Creating sample dataset structure...")
    
    # Create directories
    create_directory_structure()
    
    # Create sample images for testing
    import cv2
    import numpy as np
    
    def create_sample_image(output_path, is_real=True):
        # Create a simple image
        img = np.ones((256, 256, 3), dtype=np.uint8) * 255
        if is_real:
            # Add some random noise for real images
            noise = np.random.normal(0, 25, (256, 256, 3)).astype(np.uint8)
            img = cv2.add(img, noise)
        else:
            # Add some artificial patterns for fake images
            for i in range(5):
                cv2.circle(img, 
                          (np.random.randint(50, 206), np.random.randint(50, 206)),
                          np.random.randint(10, 30),
                          (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                          -1)
        
        cv2.imwrite(output_path, img)
    
    # Create sample images for each directory
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            for i in range(10):  # Create 10 images per category
                output_path = f'data/{split}/{label}/sample_{i}.jpg'
                create_sample_image(output_path, label == 'real')
    
    print("Sample dataset creation completed!")

def main():
    print("Starting dataset preparation...")
    
    # Instead of downloading UADFV dataset, create a sample dataset
    # This is because the UADFV dataset URL is not accessible
    print("Note: Using sample dataset for testing purposes.")
    print("For real training, please download the UADFV dataset manually from:")
    print("https://github.com/danmohaha/DSP-FWA")
    print("or use the Celeb-DF dataset from:")
    print("https://github.com/yuezunli/celeb-deepfakeforensics")
    
    download_sample_dataset()

if __name__ == "__main__":
    main() 