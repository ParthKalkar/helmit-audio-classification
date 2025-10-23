# Model Bias Insights

## Overview
Several models in this project exhibit a strong bias towards predicting class 1 (harmful), as evidenced by high recall (1.0000) but moderate precision (0.6000), resulting in F1 scores of 0.7500 and accuracies of 0.6000. This indicates the models are classifying **all samples as harmful**, which is correct for 60% of the dataset due to class imbalance (60 harmful vs 40 safe samples).

## Why This Bias Occurs

### 1. Class Imbalance
- The dataset contains 60 harmful and 40 safe samples.
- Models can achieve 60% accuracy by always predicting "harmful" without learning meaningful features.
- Class weights and weighted sampling help but may not fully compensate for the imbalance.

### 2. Model Collapse to Majority Class
- During training, models may converge to a trivial solution where they ignore input features and default to the majority class.
- This is common with small datasets (100 samples), insufficient regularization, or suboptimal hyperparameters.

### 3. Hyperparameter Issues
- **Learning Rate**: Too high can cause overfitting to the majority class; too low may prevent learning.
- **Epochs**: Limited training time might not allow proper convergence.
- **Regularization**: Dropout and weight decay may need tuning.
- **For Wav2Vec2**: Freezing the base limits adaptation; unfreezing too few layers (e.g., 2) may not provide enough trainable capacity.

### 4. Data and Augmentation Limitations
- Audio preprocessing (16kHz, mono, 8s mel-spectrograms) and augmentation (time/pitch/noise) might not be diverse enough.
- Models may rely on simple heuristics rather than robust feature learning.

### 5. Evaluation on Similar Distribution
- Training and evaluation on the same stratified split perpetuates the bias.

## Evidence from Metrics
- **Models with F1=0.7500** (lite_cnn_v1, v4, wav2vec2_v1, v2): Predict all as harmful (recall=1.0, precision=0.6).
- **Models with F1=0.0000** (lite_cnn_v2, v3): Predict all as safe (recall=0.0, precision=0.0).
- No model achieves balanced precision/recall, indicating poor discriminative ability.

## Potential Solutions
- **Data**: Collect more balanced data or use oversampling/undersampling.
- **Loss Functions**: Try focal loss or weighted loss to emphasize minority class.
- **Hyperparameters**: Experiment with lower LR (e.g., 1e-5), more epochs, stronger augmentation.
- **Architecture**: Increase model capacity or use better pre-trained models.
- **Analysis**: Use SHAP or Grad-CAM for audio to understand model decisions.

## Conclusion
The bias highlights challenges with small, imbalanced datasets in audio classification. While 60% accuracy is better than random (50%), achieving the target 90% requires addressing these issues through data, training, and model improvements.