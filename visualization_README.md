# Deepfake Detection Model Comparison Visualizations

This project contains scripts to generate and visualize performance comparisons between different deepfake detection models:
- GenConViT
- Swin Transformer
- Vision Transformer (ViT)
- MesoNet

## Generated Visualizations

The script generates the following visualizations:

1. **Accuracy vs Epochs** - Shows how model accuracy improves during training
2. **Loss vs Epochs** - Displays the training loss over time
3. **Confusion Matrices** - Visualizes classification results for each model
4. **ROC Curves** - Shows receiver operating characteristic for all models
5. **Precision-Recall Curve** - Displays PR curve for GenConViT
6. **Model Comparison Bar Graph** - Compares Accuracy, Precision, Recall, F1-Score, and AUC
7. **Model Size vs Accuracy** - Shows relationship between parameter count and performance
8. **Inference Time / FPS** - Compares model speed for real-time applications

## Requirements

The script requires the following Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- torch
- PIL

You can install these with:
```
pip install numpy pandas matplotlib seaborn scikit-learn torch Pillow
```

## How to Run

1. Make sure you have all the required packages installed.
2. Run the visualization generation script:
   ```
   python model_comparison_visualizations.py
   ```
3. All visualizations will be saved to the `visualizations/` directory.
4. To view all visualizations in an organized manner, open the `view_results.html` file in your web browser.

## Customization

You can customize the visualizations by modifying the parameters in the script:

- Change the number of epochs by modifying `num_epochs`
- Adjust model performance by changing the simulated accuracy and loss curves
- Modify parameter counts in `param_counts` dictionary to reflect your model sizes
- Change inference times in `inference_times` dictionary to match your hardware

## Example Output

The script will output:
```
Generating model comparison visualizations...
All visualizations have been saved to the 'visualizations' directory!

Visualization files:
1. accuracy_vs_epochs.png - Accuracy curves for all models
2. loss_vs_epochs.png - Loss curves for all models
3. confusion_matrices.png - Confusion matrices for all models
4. roc_curves.png - ROC curves for all models
5. precision_recall_curve.png - Precision-Recall curve for GenConViT
6. model_comparison_bar.png - Bar graph comparing all metrics
7. parameter_count_vs_accuracy.png - Model size vs. performance
8. inference_time_fps.png - Inference speed comparison
```

## Notes

- The visualizations use simulated data for demonstration purposes
- Replace the simulated data with your actual model training and evaluation metrics for accurate results
- You can modify the plot styles by changing the Matplotlib and Seaborn parameters at the top of the script 