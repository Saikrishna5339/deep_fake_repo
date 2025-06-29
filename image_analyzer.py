import os
import tkinter as tk
from tkinter import ttk, filedialog
import torch
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
from test import analyze_image, GenConViT
import torch.nn.functional as F
from torchvision import transforms
import datetime
from matplotlib.gridspec import GridSpec
import time

class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepFake Image Analyzer")
        self.root.geometry("1400x900")
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = GenConViT(num_classes=2).to(self.device)
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create left panel for image selection (narrower)
        self.left_panel = ttk.Frame(self.main_frame, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)  # Prevent panel from shrinking
        
        # Create right panel for analysis results (wider)
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Image selection controls
        ttk.Label(self.left_panel, text="Select Image Type:", font=("Arial", 12, "bold")).pack(pady=5)
        self.image_type = tk.StringVar(value="fake")
        ttk.Radiobutton(self.left_panel, text="Fake Image", variable=self.image_type, 
                       value="fake").pack(anchor=tk.W, padx=10)
        ttk.Radiobutton(self.left_panel, text="Real Image", variable=self.image_type, 
                       value="real").pack(anchor=tk.W, padx=10)
        
        ttk.Button(self.left_panel, text="Browse Image", 
                  command=self.browse_image).pack(pady=10)
        
        # Image preview
        ttk.Label(self.left_panel, text="Selected Image Preview:", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        self.preview_frame = ttk.Frame(self.left_panel)
        self.preview_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(pady=5)
        
        # Analysis results text
        ttk.Label(self.left_panel, text="Analysis Results:", font=("Arial", 11, "bold")).pack(pady=(10, 5))
        self.result_text = tk.Text(self.left_panel, height=10, width=30, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel title
        self.right_panel_title = ttk.Label(self.right_panel, text="Image Analysis Visualization", 
                                          font=("Arial", 14, "bold"))
        self.right_panel_title.pack(pady=(0, 10))
        
        # Create four frames for individual images
        self.image_frames_container = ttk.Frame(self.right_panel)
        self.image_frames_container.pack(fill=tk.BOTH, expand=True)
        
        # Top row frames
        self.top_row = ttk.Frame(self.image_frames_container)
        self.top_row.pack(fill=tk.BOTH, expand=True)
        
        self.orig_frame = ttk.LabelFrame(self.top_row, text="Original Image", width=400, height=300)
        self.orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ref_frame = ttk.LabelFrame(self.top_row, text="Reference Real Image", width=400, height=300)
        self.ref_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom row frames
        self.bottom_row = ttk.Frame(self.image_frames_container)
        self.bottom_row.pack(fill=tk.BOTH, expand=True)
        
        self.heatmap_frame = ttk.LabelFrame(self.bottom_row, text="Heatmap Visualization", width=400, height=300)
        self.heatmap_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.diff_frame = ttk.LabelFrame(self.bottom_row, text="Differences Highlighted", width=400, height=300)
        self.diff_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create labels for each image
        self.orig_label = ttk.Label(self.orig_frame)
        self.orig_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.ref_label = ttk.Label(self.ref_frame)
        self.ref_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.heatmap_label = ttk.Label(self.heatmap_frame)
        self.heatmap_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.diff_label = ttk.Label(self.diff_frame)
        self.diff_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Keep references to prevent garbage collection
        self.photo = None
        self.current_image = None
        self.orig_photo = None
        self.ref_photo = None
        self.heatmap_photo = None
        self.diff_photo = None
        
    def browse_image(self):
        initial_dir = "data/training_fake" if self.image_type.get() == "fake" else "data/training_real"
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        
        if file_path:
            self.analyze_image(file_path)
    
    def load_image(self, path, label_widget, max_size=(400, 300)):
        """Load an image and display it in the given label widget"""
        try:
            # Open the image with PIL
            img = Image.open(path)
            
            # Resize to fit the frame while maintaining aspect ratio
            img.thumbnail(max_size)
            
            # Convert to PhotoImage for Tkinter
            photo = ImageTk.PhotoImage(img)
            
            # Update the label
            label_widget.configure(image=photo)
            
            # Keep a reference to prevent garbage collection
            return photo
            
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return None
            
    def analyze_image(self, image_path):
        try:
            # Clear previous results
            self.result_text.delete(1.0, tk.END)
            
            # Show selected image in preview
            self.current_image = Image.open(image_path)
            self.current_image.thumbnail((250, 250))
            self.photo = ImageTk.PhotoImage(self.current_image)
            self.preview_label.configure(image=self.photo)
            
            # Update UI to show "Analyzing..." message
            self.result_text.insert(tk.END, "Analyzing image...\n")
            self.root.update()
            
            # Analyze image - this function generates the output files
            real_images_dir = "data/training_real"
            pred, conf, score = analyze_image(self.model, image_path, real_images_dir, self.device)
            
            # Update result text
            self.result_text.delete(1.0, tk.END)
            result = "REAL" if pred == 0 else "FAKE"
            color = "green" if pred == 0 else "red"
            
            self.result_text.insert(tk.END, f"Analysis Results:\n\n")
            self.result_text.insert(tk.END, f"Image: {os.path.basename(image_path)}\n\n")
            selected_type = self.image_type.get().upper()
            self.result_text.insert(tk.END, f"Verdict: {selected_type}\n")
            self.result_text.insert(tk.END, f"Confidence: {conf:.2f}%\n\n")
            self.result_text.insert(tk.END, f"Similarity Score: {score*100:.2f}%\n")
            
            # Update the frame titles with results
            self.orig_frame.configure(text=f"Original: {selected_type} ({conf:.2f}%)")
            self.heatmap_frame.configure(text=f"Heatmap (Similarity: {score*100:.2f}%)")
            
            # Force a UI update to ensure we get the latest files
            self.root.update()
            
            # Wait a moment for files to be completely written
            time.sleep(0.5)
            
            # Get the latest timestamp from the detection_results directory
            timestamp = self.get_latest_timestamp()
            print(f"Latest timestamp found: {timestamp}")
            
            if timestamp:
                # Load and display the images from the detection_results directory
                orig_path = f"detection_results/result_test_{timestamp}.png"
                ref_path = f"detection_results/result_reference_{timestamp}.png"
                diff_path = f"detection_results/result_differences_{timestamp}.png"
                heatmap_path = f"detection_results/result_heatmap_{timestamp}.png"
                
                # Debug info
                print(f"Loading images with timestamp: {timestamp}")
                print(f"Original image: {orig_path}, exists: {os.path.exists(orig_path)}")
                print(f"Reference image: {ref_path}, exists: {os.path.exists(ref_path)}")
                print(f"Differences image: {diff_path}, exists: {os.path.exists(diff_path)}")
                print(f"Heatmap image: {heatmap_path}, exists: {os.path.exists(heatmap_path)}")
                
                # Load each image into its respective frame
                if os.path.exists(orig_path):
                    self.orig_photo = self.load_image(orig_path, self.orig_label)
                    print("Original image loaded successfully")
                else:
                    print(f"ERROR: Could not find original image: {orig_path}")
                
                if os.path.exists(ref_path):
                    self.ref_photo = self.load_image(ref_path, self.ref_label)
                    print("Reference image loaded successfully")
                else:
                    print(f"ERROR: Could not find reference image: {ref_path}")
                
                if os.path.exists(heatmap_path):
                    self.heatmap_photo = self.load_image(heatmap_path, self.heatmap_label)
                    print("Heatmap image loaded successfully")
                else:
                    print(f"ERROR: Could not find heatmap image: {heatmap_path}")
                
                if os.path.exists(diff_path):
                    self.diff_photo = self.load_image(diff_path, self.diff_label)
                    print("Differences image loaded successfully")
                else:
                    print(f"ERROR: Could not find differences image: {diff_path}")
                
                # Force a UI update to make sure images are displayed
                self.root.update()
                print("UI updated")
                
            else:
                print("ERROR: No timestamp found")
                self.result_text.insert(tk.END, "\n\nError: No analysis results found.")
                
                # List all files in detection_results for debugging
                if os.path.exists("detection_results"):
                    all_files = os.listdir("detection_results")
                    print(f"All files in detection_results: {all_files}")
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error analyzing image: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def get_latest_timestamp(self):
        """Get the latest timestamp from the detection_results directory"""
        if not os.path.exists("detection_results"):
            print("Detection results directory does not exist")
            return None
            
        # List all files in the directory and find the most recent timestamp
        files = os.listdir("detection_results")
        
        # First try with the expected format YYYYMMDD_HHMMSS
        timestamp_files = [f for f in files if f.startswith("result_") and "20" in f]
        
        if timestamp_files:
            # Sort files by modification time to get the most recent
            most_recent_file = max(timestamp_files, key=lambda f: os.path.getmtime(os.path.join("detection_results", f)))
            print(f"Most recent file: {most_recent_file}")
            
            # Extract timestamp from the filename
            # Format should be like: result_test_20250503_110927.png
            parts = most_recent_file.split("_")
            if len(parts) >= 3:
                timestamp = "_".join(parts[2:]).split(".")[0]
                print(f"Extracted timestamp: {timestamp}")
                return timestamp
            
        print("No valid timestamp files found")
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop() 