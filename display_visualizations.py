import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np

def display_all_visualizations():
    # Check if visualizations directory exists
    if not os.path.exists('visualizations'):
        print("Visualizations not found. Please run model_comparison_visualizations.py first.")
        return
    
    # List of visualization files to display
    visualization_files = [
        'accuracy_vs_epochs.png',
        'loss_vs_epochs.png',
        'confusion_matrices.png',
        'roc_curves.png',
        'precision_recall_curve.png',
        'model_comparison_bar.png',
        'parameter_count_vs_accuracy.png',
        'inference_time_fps.png'
    ]
    
    # Make sure all files exist
    for file in visualization_files:
        filepath = os.path.join('visualizations', file)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found.")
    
    # Display each visualization in full window
    for file in visualization_files:
        filepath = os.path.join('visualizations', file)
        if os.path.exists(filepath):
            # Create figure with title
            plt.figure(figsize=(12, 8))
            img = mpimg.imread(filepath)
            plt.imshow(img)
            plt.axis('off')
            plt.title(file.replace('.png', '').replace('_', ' ').title(), fontsize=16)
            plt.tight_layout()
            plt.show()

def display_summary_grid():
    """Display all visualizations in a single grid layout"""
    # Check if visualizations directory exists
    if not os.path.exists('visualizations'):
        print("Visualizations not found. Please run model_comparison_visualizations.py first.")
        return
    
    # Define the visualization files
    visualization_files = [
        'accuracy_vs_epochs.png',
        'loss_vs_epochs.png',
        'confusion_matrices.png',
        'roc_curves.png',
        'precision_recall_curve.png',
        'model_comparison_bar.png',
        'parameter_count_vs_accuracy.png',
        'inference_time_fps.png'
    ]
    
    # Check if all files exist
    missing_files = []
    for file in visualization_files:
        filepath = os.path.join('visualizations', file)
        if not os.path.exists(filepath):
            missing_files.append(file)
    
    if missing_files:
        print(f"Warning: Some visualization files are missing: {', '.join(missing_files)}")
    
    # Create a grid figure
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 2, figure=fig)
    
    # Load and display each visualization in the grid
    for i, file in enumerate(visualization_files):
        filepath = os.path.join('visualizations', file)
        if os.path.exists(filepath):
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(gs[row, col])
            img = mpimg.imread(filepath)
            ax.imshow(img)
            ax.set_title(file.replace('.png', '').replace('_', ' ').title(), fontsize=14)
            ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.show()

if __name__ == "__main__":
    print("Displaying model comparison visualizations...")
    
    # Ask user which display mode they prefer
    print("\nDisplay options:")
    print("1. Show each visualization in separate window (one after another)")
    print("2. Show all visualizations in a single grid")
    
    choice = input("\nEnter your choice (1 or 2): ")
    
    if choice == "1":
        display_all_visualizations()
    else:
        display_summary_grid() 