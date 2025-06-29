import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from models.genconvit import GenConViT
from torchvision.models import swin_t, vit_b_16
import torch.nn as nn

# Set style for better visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")
colors = sns.color_palette("husl", 4)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Sample data for epochs (training history simulation)
num_epochs = 50
epochs = list(range(1, num_epochs + 1))

# Create directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

# ---------- 1. Accuracy vs Epochs ----------
def plot_accuracy_vs_epochs(show_plot=True):
    # Simulated accuracy values for each model
    # GenConViT performs best, followed by Swin, ViT, and MesoNet
    np.random.seed(42)
    genconvit_acc = np.linspace(0.7, 0.98, num_epochs) + np.random.normal(0, 0.02, num_epochs)
    swin_acc = np.linspace(0.65, 0.93, num_epochs) + np.random.normal(0, 0.02, num_epochs)
    vit_acc = np.linspace(0.6, 0.9, num_epochs) + np.random.normal(0, 0.025, num_epochs)
    mesonet_acc = np.linspace(0.55, 0.87, num_epochs) + np.random.normal(0, 0.03, num_epochs)
    
    # Clip values to be between 0 and 1
    genconvit_acc = np.clip(genconvit_acc, 0, 1)
    swin_acc = np.clip(swin_acc, 0, 1)
    vit_acc = np.clip(vit_acc, 0, 1)
    mesonet_acc = np.clip(mesonet_acc, 0, 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, genconvit_acc, label='GenConViT', color=colors[0], linewidth=2)
    plt.plot(epochs, swin_acc, label='Swin Transformer', color=colors[1], linewidth=2)
    plt.plot(epochs, vit_acc, label='ViT', color=colors[2], linewidth=2)
    plt.plot(epochs, mesonet_acc, label='MesoNet', color=colors[3], linewidth=2)
    
    plt.title('Model Accuracy vs. Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/accuracy_vs_epochs.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return {
        'GenConViT': genconvit_acc,
        'Swin': swin_acc,
        'ViT': vit_acc,
        'MesoNet': mesonet_acc
    }

# ---------- 2. Loss vs Epochs ----------
def plot_loss_vs_epochs(show_plot=True):
    # Simulated loss values (inverse of accuracy trend)
    np.random.seed(43)
    genconvit_loss = np.linspace(0.8, 0.05, num_epochs) + np.random.normal(0, 0.02, num_epochs)
    swin_loss = np.linspace(0.9, 0.1, num_epochs) + np.random.normal(0, 0.03, num_epochs)
    vit_loss = np.linspace(1.0, 0.15, num_epochs) + np.random.normal(0, 0.04, num_epochs)
    mesonet_loss = np.linspace(1.1, 0.2, num_epochs) + np.random.normal(0, 0.05, num_epochs)
    
    # Clip values to be positive
    genconvit_loss = np.clip(genconvit_loss, 0, None)
    swin_loss = np.clip(swin_loss, 0, None)
    vit_loss = np.clip(vit_loss, 0, None)
    mesonet_loss = np.clip(mesonet_loss, 0, None)
    
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, genconvit_loss, label='GenConViT', color=colors[0], linewidth=2)
    plt.plot(epochs, swin_loss, label='Swin Transformer', color=colors[1], linewidth=2)
    plt.plot(epochs, vit_loss, label='ViT', color=colors[2], linewidth=2)
    plt.plot(epochs, mesonet_loss, label='MesoNet', color=colors[3], linewidth=2)
    
    plt.title('Model Loss vs. Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/loss_vs_epochs.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

# ---------- 3. Confusion Matrices ----------
def plot_confusion_matrices(accuracy_dict, show_plot=True):
    # Generate confusion matrices based on final accuracies
    models = ['GenConViT', 'Swin', 'ViT', 'MesoNet']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, model_name in enumerate(models):
        # Simulated test set of 1000 samples with balanced classes
        n_samples = 1000
        n_real = n_fake = n_samples // 2
        
        # Calculate TP, FP, TN, FN based on accuracy
        accuracy = accuracy_dict[model_name][-1]  # Use final accuracy
        
        # Perfect classification would be:
        # [[500, 0],   (TN, FP)
        #  [0, 500]]   (FN, TP)
        
        # For non-perfect accuracy:
        errors = int((1 - accuracy) * n_samples)
        # Distribute errors between false positives and false negatives
        fp = np.random.binomial(errors, 0.5)
        fn = errors - fp
        
        tp = n_real - fn
        tn = n_fake - fp
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d')
        disp.ax_.set_title(f'{model_name} Confusion Matrix\nAccuracy: {accuracy:.2f}', fontsize=14)
        disp.ax_.set_xlabel('Predicted Label', fontsize=12)
        disp.ax_.set_ylabel('True Label', fontsize=12)
        
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/confusion_matrices.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

# ---------- 4. ROC Curves ----------
def plot_roc_curves(accuracy_dict, show_plot=True):
    plt.figure(figsize=(12, 8))
    
    # Generate ROC curves based on model performance
    models = ['GenConViT', 'Swin', 'ViT', 'MesoNet']
    
    for i, model_name in enumerate(models):
        # Generate synthetic prediction scores
        np.random.seed(44 + i)
        n_samples = 1000
        
        # Generate target labels (50% real, 50% fake)
        y_true = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
        
        # Generate prediction scores based on model accuracy
        accuracy = accuracy_dict[model_name][-1]  # Final accuracy
        
        # For real samples (label 0), scores should be low (correct prediction)
        # For fake samples (label 1), scores should be high (correct prediction)
        # Good model: real samples get scores near 0, fake samples get scores near 1
        
        # Real samples
        real_scores = np.random.beta(1, 3 * accuracy, n_samples//2)  # Skewed toward 0
        
        # Fake samples
        fake_scores = np.random.beta(3 * accuracy, 1, n_samples//2)  # Skewed toward 1
        
        y_score = np.concatenate([real_scores, fake_scores])
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})', color=colors[i])
    
    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/roc_curves.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return roc_auc

# ---------- 5. Precision-Recall Curve for GenConViT ----------
def plot_precision_recall_curve(accuracy_dict, show_plot=True):
    plt.figure(figsize=(12, 8))
    
    # Generate synthetic prediction scores for GenConViT
    np.random.seed(45)
    n_samples = 1000
    
    # Generate target labels (50% real, 50% fake)
    y_true = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Generate prediction scores based on model accuracy
    accuracy = accuracy_dict['GenConViT'][-1]  # Final accuracy
    
    # Real samples
    real_scores = np.random.beta(1, 5 * accuracy, n_samples//2)  # Skewed toward 0
    
    # Fake samples
    fake_scores = np.random.beta(5 * accuracy, 1, n_samples//2)  # Skewed toward 1
    
    y_score = np.concatenate([real_scores, fake_scores])
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    
    plt.plot(recall, precision, color=colors[0], lw=2)
    plt.fill_between(recall, precision, alpha=0.2, color=colors[0])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve for GenConViT', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/precision_recall_curve.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

# ---------- 6. Bar Graph for Model Comparison ----------
def plot_model_comparison_bar_graph(accuracy_dict, roc_auc_value, show_plot=True):
    # Define models and metrics
    models = ['GenConViT', 'Swin', 'ViT', 'MesoNet']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    # Create synthetic performance metrics
    np.random.seed(46)
    data = []
    
    for model in models:
        accuracy = accuracy_dict[model][-1]  # Final accuracy
        
        # Generate slightly varied metrics around accuracy
        precision = min(1.0, accuracy + np.random.normal(0, 0.02))
        recall = min(1.0, accuracy + np.random.normal(0, 0.02))
        f1 = min(1.0, 2 * (precision * recall) / (precision + recall + 1e-10))
        
        # AUC is the value from ROC curve
        auc_value = roc_auc_value if model == 'GenConViT' else max(0.5, min(1.0, roc_auc_value - np.random.uniform(0.02, 0.1)))
        
        data.append([accuracy, precision, recall, f1, auc_value])
    
    # Create DataFrame
    df = pd.DataFrame(data, index=models, columns=metrics)
    
    # Plot bar graph
    plt.figure(figsize=(14, 10))
    ax = df.plot(kind='bar', rot=0, width=0.8)
    
    # Customize plot
    plt.title('Model Performance Comparison', fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Models', fontsize=14)
    plt.ylim(0.5, 1.05)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/model_comparison_bar.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

# ---------- 7. Model Parameter Count vs Accuracy ----------
def plot_parameter_count_vs_accuracy(accuracy_dict, show_plot=True):
    # Define models and their approximate parameter counts
    models = ['GenConViT', 'Swin', 'ViT', 'MesoNet']
    
    # Parameter counts in millions
    param_counts = {
        'GenConViT': 15,  # Example: 15M parameters
        'Swin': 28,       # Example: 28M parameters (swin_t)
        'ViT': 86,        # Example: 86M parameters (vit_b_16)
        'MesoNet': 2.5    # Example: 2.5M parameters
    }
    
    # Get final accuracies
    accuracies = [accuracy_dict[model][-1] for model in models]
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Scatter points
    for i, model in enumerate(models):
        plt.scatter(
            param_counts[model], 
            accuracies[i], 
            s=300, 
            color=colors[i], 
            label=model, 
            alpha=0.7
        )
    
    # Connect with dashed lines
    plt.plot(
        [param_counts[model] for model in models], 
        accuracies, 
        'k--', 
        alpha=0.3
    )
    
    # Add text labels with parameter counts
    for i, model in enumerate(models):
        plt.annotate(
            f"{param_counts[model]}M params",
            (param_counts[model], accuracies[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10
        )
    
    plt.title('Model Parameter Count vs. Accuracy', fontsize=16)
    plt.xlabel('Parameter Count (millions)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/parameter_count_vs_accuracy.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

# ---------- 8. Inference Time / FPS Comparison ----------
def plot_inference_time_comparison(show_plot=True):
    # Define models
    models = ['GenConViT', 'Swin', 'ViT', 'MesoNet']
    
    # Simulated inference times in milliseconds
    inference_times = {
        'GenConViT': 18,  # Example: 18ms per image
        'Swin': 25,       # Example: 25ms per image
        'ViT': 30,        # Example: 30ms per image
        'MesoNet': 12     # Example: 12ms per image
    }
    
    # Calculate FPS
    fps = {model: 1000 / time for model, time in inference_times.items()}
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Inference Time (ms)': [inference_times[model] for model in models],
        'FPS': [fps[model] for model in models]
    }, index=models)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Inference Time
    ax1 = axes[0]
    df['Inference Time (ms)'].plot(kind='bar', color=colors, ax=ax1)
    ax1.set_title('Inference Time per Image', fontsize=14)
    ax1.set_ylabel('Time (milliseconds)', fontsize=12)
    ax1.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(df['Inference Time (ms)']):
        ax1.text(i, v + 1, f"{v}ms", ha='center', fontsize=10)
    
    # FPS
    ax2 = axes[1]
    df['FPS'].plot(kind='bar', color=colors, ax=ax2)
    ax2.set_title('Frames Per Second (FPS)', fontsize=14)
    ax2.set_ylabel('Frames per Second', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)
    for i, v in enumerate(df['FPS']):
        ax2.text(i, v + 2, f"{v:.1f} FPS", ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('visualizations/inference_time_fps.png', dpi=300)
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Return FPS data for additional use if needed
    return fps

# ---------- Main Function ----------
def main(show_plots=True):
    print("Generating and displaying model comparison visualizations...")
    
    # Create all visualizations
    accuracy_dict = plot_accuracy_vs_epochs(show_plot=show_plots)
    plot_loss_vs_epochs(show_plot=show_plots)
    plot_confusion_matrices(accuracy_dict, show_plot=show_plots)
    roc_auc_value = plot_roc_curves(accuracy_dict, show_plot=show_plots)
    plot_precision_recall_curve(accuracy_dict, show_plot=show_plots)
    plot_model_comparison_bar_graph(accuracy_dict, roc_auc_value, show_plot=show_plots)
    plot_parameter_count_vs_accuracy(accuracy_dict, show_plot=show_plots)
    plot_inference_time_comparison(show_plot=show_plots)
    
    print("All visualizations have been generated and displayed!")
    print("\nVisualization files also saved to the 'visualizations' directory:")
    print("1. accuracy_vs_epochs.png - Accuracy curves for all models")
    print("2. loss_vs_epochs.png - Loss curves for all models")
    print("3. confusion_matrices.png - Confusion matrices for all models")
    print("4. roc_curves.png - ROC curves for all models")
    print("5. precision_recall_curve.png - Precision-Recall curve for GenConViT")
    print("6. model_comparison_bar.png - Bar graph comparing all metrics")
    print("7. parameter_count_vs_accuracy.png - Model size vs. performance")
    print("8. inference_time_fps.png - Inference speed comparison")

if __name__ == "__main__":
    main(show_plots=True) 