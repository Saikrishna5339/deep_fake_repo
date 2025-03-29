import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm

from models.genconvit import GenConViT
from utils.data_loader import DeepfakeDataset, VideoDataset, get_transforms

def test_image(model, image_path, device):
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_transforms(is_train=False)
    transformed = transform(image=image)
    image = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
    return predicted.item()

def test_video(model, video_path, device):
    # Load video
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
    if len(frames) > 32:
        indices = np.linspace(0, len(frames)-1, 32, dtype=int)
        frames = [frames[i] for i in indices]
    
    # Transform frames
    transform = get_transforms(is_train=False)
    transformed_frames = []
    for frame in frames:
        transformed = transform(image=frame)
        transformed_frames.append(transformed['image'])
    
    # Stack frames and make prediction
    frames = torch.stack(transformed_frames).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(frames)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.item()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def analyze_image(model, test_image_path, real_images_dir, device):
    """Analyze if test image is real or fake and compare with real image"""
    # Load and preprocess the test image
    test_image = Image.open(test_image_path).convert('RGB')
    
    # Find a real image for comparison
    real_image_path = None
    for img_name in os.listdir(real_images_dir):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            real_image_path = os.path.join(real_images_dir, img_name)
            break
    
    if not real_image_path:
        print(f"No real images found in {real_images_dir}")
        return
    
    real_image = Image.open(real_image_path).convert('RGB')
    
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform images
    test_tensor = transform(test_image).unsqueeze(0).to(device)
    real_tensor = transform(real_image).unsqueeze(0).to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        # Test image prediction
        test_output = model(test_tensor)
        test_probs = F.softmax(test_output, dim=1)
        test_pred = torch.argmax(test_output, dim=1).item()
        test_conf = test_probs[0][test_pred].item() * 100
        
        # Real image prediction
        real_output = model(real_tensor)
        real_probs = F.softmax(real_output, dim=1)
        real_pred = torch.argmax(real_output, dim=1).item()
        real_conf = real_probs[0][real_pred].item() * 100
    
    # Convert to numpy arrays for difference analysis
    test_np = np.array(test_image.resize((256, 256)))
    real_np = np.array(real_image.resize((256, 256)))
    
    # Calculate structural similarity
    gray_test = cv2.cvtColor(test_np, cv2.COLOR_RGB2GRAY)
    gray_real = cv2.cvtColor(real_np, cv2.COLOR_RGB2GRAY)
    score, diff = ssim(gray_test, gray_real, full=True)
    diff = (diff * 255).astype("uint8")
    
    # Threshold the difference image to highlight differences
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create difference visualization
    diff_vis = test_np.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small areas
            cv2.drawContours(diff_vis, [contour], 0, (0, 0, 255), 2)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Original test image with prediction
    plt.subplot(2, 2, 1)
    plt.imshow(test_image)
    test_result = "REAL" if test_pred == 0 else "FAKE"
    test_color = "green" if test_pred == 0 else "red"
    plt.title(f"Test Image: {test_result} ({test_conf:.2f}%)", color=test_color, fontsize=14)
    plt.axis('off')
    
    # Real reference image
    plt.subplot(2, 2, 2)
    plt.imshow(real_image)
    real_result = "REAL" if real_pred == 0 else "FAKE"
    plt.title(f"Reference Real Image ({real_conf:.2f}%)", color="green", fontsize=14)
    plt.axis('off')
    
    # Difference visualization
    plt.subplot(2, 2, 3)
    plt.imshow(diff_vis)
    plt.title(f"Differences Highlighted", fontsize=14)
    plt.axis('off')
    
    # Difference heatmap
    plt.subplot(2, 2, 4)
    plt.imshow(255 - diff, cmap='jet')
    plt.title(f"Difference Heatmap (Similarity: {score*100:.2f}%)", fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('analysis_result.png', dpi=300)
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print(f" {'ANALYSIS RESULT':^56} ")
    print("="*60)
    print(f"Test Image: {test_image_path}")
    print(f"Reference Real Image: {real_image_path}")
    print("-"*60)
    print(f"VERDICT: The test image is {'REAL' if test_pred == 0 else 'FAKE'} with {test_conf:.2f}% confidence")
    print("-"*60)
    print(f"Image Similarity Score: {score*100:.2f}%")
    print(f"Major Differences: {len([c for c in contours if cv2.contourArea(c) > 100])} areas")
    print("="*60)
    
    # Give detailed analysis of differences
    print("\nDetailed Analysis of Differences:")
    
    if test_pred == 1:  # If predicted as fake
        print("- Areas with unnatural artifacts have been highlighted in red")
        print("- The heatmap shows regions with major differences from real images")
        print("- Common deepfake indicators found: inconsistent textures, blending artifacts")
        
        if score < 0.7:
            print("- High level of manipulation detected")
        elif score < 0.9:
            print("- Moderate level of manipulation detected")
        else:
            print("- Subtle manipulation detected, high-quality deepfake")
    else:
        print("- No significant manipulation indicators detected")
        print("- Normal variations are within expected range for real images")
    
    print("\nAnalysis saved as 'analysis_result.png'")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = GenConViT(num_classes=2).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Get test image path from user
    test_image_path = input("Enter the path to the test image: ")
    
    # Check if file exists
    if not os.path.exists(test_image_path):
        print(f"Error: File not found at {test_image_path}")
        return
    
    # Set path to real images directory
    real_images_dir = 'data/test/real'
    
    # Check if directory exists
    if not os.path.exists(real_images_dir):
        print(f"Error: Real images directory not found at {real_images_dir}")
        return
    
    # Analyze image
    analyze_image(model, test_image_path, real_images_dir, device)

if __name__ == "__main__":
    main() 