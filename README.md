# Helmit Audio Classification Project

## Overview
This project implements audio classification models to detect harmful vs. safe content in audio clips, such as harassment, bullying, or grooming vs. neutral conversations. It includes custom lightweight CNN architectures and pre-trained Wav2Vec2 models from Hugging Face, trained on a small dataset of 100 audio samples (60 harmful, 40 safe). The goal is to achieve high accuracy (>90%) for real-time deployment, but current models reach ~60% due to class imbalance, limited data, and training challenges.

Models are exported to ONNX for efficient inference across platforms, with quantization support for reduced model size and faster inference where possible. The project leverages PyTorch for training, torchaudio for audio processing, and ONNX Runtime for deployment.

### Key Features
- **Dual Architectures**: Custom CNN for lightweight inference, Wav2Vec2 for state-of-the-art performance.
- **Audio Preprocessing**: Robust handling of variable-length audio with mel-spectrograms and raw waveforms.
- **Data Augmentation**: Time stretching, pitch shifting, and noise injection to improve generalization.
- **ONNX Export & Quantization**: Cross-platform deployment with optional INT8 quantization.
- **Evaluation Metrics**: Comprehensive metrics including F1, precision, recall, and confusion matrices.
- **Bias Analysis**: Insights into model behavior and recommendations for improvement.

### Motivation
Audio content moderation is crucial for platforms dealing with user-generated audio. Traditional methods rely on manual review, but ML can automate detection. This project explores the trade-offs between custom models (fast, small) and pre-trained models (accurate, large) on limited data.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repo-url>
   cd Helmit
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux; Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies:
   - `torch>=2.0`: PyTorch for model training.
   - `torchaudio`: Audio processing.
   - `librosa`: Audio loading and augmentation.
   - `pandas`, `numpy`: Data handling.
   - `scikit-learn`: Metrics and splitting.
   - `transformers`: For Wav2Vec2.
   - `onnx`, `onnxruntime`: Export and inference.

4. **Optional: ONNX Quantization**:
   ```bash
   pip install onnxruntime
   ```
   Note: Quantization may fail on CPU for complex models; use GPU if available.

5. **Verify Installation**:
   ```bash
   python -c "import torch, torchaudio, librosa; print('Ready')"
   ```

## Dataset
The dataset consists of 100 audio clips (WAV/MP3) categorized into harmful (e.g., harassment, bullying) and safe (neutral) content.

### Source and Preparation
- **Download**: Run `python download_from_drive.py` to fetch `harmful.zip` and `safe.zip` from Google Drive.
- **Unzip**: Extract to `data/harmful/` and `data/safe/`.
- **Labels**: `python prep_dataset.py` generates `data/labels.csv` with columns: `path` (e.g., "harmful/file.mp3"), `label` (0=safe, 1=harmful).

### Statistics
- Total samples: 100 (60 harmful, 40 safe).
- Duration: Variable, cropped/padded to 8 seconds.
- Format: 16kHz, mono.
- Split: Stratified 80/20 train/validation (80 train, 20 val).

### Preprocessing Details
- **Loading**: `librosa.load` with resampling to 16kHz, mono conversion.
- **Cropping/Padding**: Ensure exact 8-second length using `utils_audio.crop_or_pad`.
- **For CNN**: Convert to mel-spectrogram (128 mels, 512 FFT, 256 hop, fmin=60, fmax=8000).
- **For Wav2Vec2**: Use raw 1D waveform tensor.
- **Normalization**: Mel-spectrograms are log-scaled; waveforms are float32.

### Augmentation
Applied during training (70% probability):
- **Time Stretch**: Rate 0.85-1.2x.
- **Pitch Shift**: Steps -3 to +3.
- **Noise Addition**: SNR 18-32 dB.
- **SpecAugment**: Frequency/time masking for mel-spectrograms.

## Models
All models output logits for 2 classes, trained with CrossEntropyLoss, class weights (inverse to frequency), and optional weighted sampling.

### Custom CNN (LiteAudioCNN)
A lightweight convolutional neural network optimized for mobile/edge deployment.

#### Architecture
- **Input**: Mel-spectrogram [1, 128, T] (T~250 for 8s audio).
- **Stem**: Conv2d (3x3, stride 2x2) + BatchNorm + ReLU.
- **Blocks**: 4 depthwise separable conv blocks with increasing channels (32→64→128→160→192), each with stride 2x1, BatchNorm, ReLU.
- **Pooling**: Global average pooling.
- **Head**: Dropout + Linear (192 → 2).
- **Parameters**: `width_mult` scales all channels (e.g., 1.0 = base, 1.5 = 50% more).
- **Total Params**: ~100k-500k depending on width_mult.

#### Why This Architecture?
- Depthwise separable convs reduce parameters for efficiency.
- Global pooling handles variable time dimensions.
- Suitable for ONNX quantization.

#### Trained Variants
- `lite_cnn_v1.pt`: width_mult=1.0, dropout=0.2, lr=5e-4, epochs=50
- `lite_cnn_v2.pt`: width_mult=1.2, dropout=0.5, lr=1e-4, epochs=50
- `lite_cnn_v3.pt`: width_mult=0.8, dropout=0.3, lr=5e-5, epochs=50
- `lite_cnn_v4.pt`: width_mult=1.5, dropout=0.4, lr=1e-3, epochs=50

### Pre-trained Wav2Vec2
Leverages self-supervised pre-training for superior performance on limited data.

#### Architecture
- **Base Model**: `facebook/wav2vec2-base` (95M params, frozen by default).
- **Input**: Raw waveform [T] (T=128000 for 8s).
- **Encoder**: 12 transformer layers, processes to hidden states [T, 768].
- **Pooling**: Mean pooling over time → [768].
- **Head**: Linear (768 → 2).
- **Unfreezing**: Optionally unfreeze last N layers for fine-tuning.

#### Why Wav2Vec2?
- Pre-trained on 960 hours of speech; captures acoustic features well.
- Handles raw audio directly, no manual feature engineering.
- Fine-tuning adapts to task with few samples.

#### Trained Variants
- `wav2vec2_v1.pt`: lr=3e-4, epochs=30, num_unfrozen_layers=0 (frozen base)
- `wav2vec2_v2.pt`: lr=5e-4, epochs=20, num_unfrozen_layers=2 (last 2 layers trainable)

## Training
Training is handled by `train.py`, supporting both architectures with flexible hyperparameters.

### Key Components
- **Optimizer**: AdamW (lr, weight_decay=1e-4).
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5, monitors val F1).
- **Loss**: CrossEntropyLoss with class weights (total / (2 * counts)).
- **Sampling**: Optional WeightedRandomSampler for balanced batches.
- **Early Stopping**: Patience=12 on val F1.
- **Device**: Auto-detects MPS (Apple Silicon), CUDA, or CPU.
- **Batch Size**: Adjusted for memory (e.g., 4 for MPS, 8 for CPU).
- **Accumulation**: Gradient accumulation for larger effective batch sizes.

### Command-Line Flags
- **Common**: `--epochs`, `--batch_size`, `--lr`, `--ckpt`, `--use_sampler`, `--accumulate_steps`, `--use_autocast`
- **CNN**: `--width_mult`, `--dropout`
- **Wav2Vec2**: `--use_wav2vec2`, `--num_unfrozen_layers`

### Examples
```bash
# Train CNN v1 with default settings
python train.py --width_mult 1.0 --dropout 0.2 --lr 5e-4 --batch_size 4 --epochs 50 --ckpt lite_cnn_v1.pt

# Train Wav2Vec2 v2 with unfreezing
python train.py --use_wav2vec2 --lr 5e-4 --epochs 20 --batch_size 2 --num_unfrozen_layers 2 --ckpt wav2vec2_v2.pt

# With sampler and autocast
python train.py --use_wav2vec2 --lr 3e-4 --use_sampler --use_autocast --ckpt wav2vec2_custom.pt
```

### Monitoring
- Logs: Loss, accuracy, F1, precision, recall per epoch.
- RAM: Reported per epoch.
- Time: Per epoch and total.

## Evaluation
Evaluate models using prediction CSVs generated by inference.

### Computing Metrics
Run `compute_metrics.py` to generate `model_metrics.txt` with accuracy, precision, recall, F1 for all models.

Manual computation:
```python
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv('data/lite_cnn_v1_predictions.csv')
print(classification_report(df['label'], df['pred']))
```

### Validation vs. Full Dataset
- Training uses 20-sample validation set.
- Inference uses full 100-sample dataset for final metrics.

## Inference
Supports both PyTorch and ONNX for flexibility.

### PyTorch Inference
```python
import torch
from model import LiteAudioCNN
from utils_audio import load_audio, to_logmel

# Load model
model = LiteAudioCNN(n_mels=128, n_classes=2, width_mult=1.0, dropout=0.2)
model.load_state_dict(torch.load('lite_cnn_v1.pt'))
model.eval()

# Preprocess
y = load_audio('path/to/audio.wav', sr=16000, mono=True, target_sec=8.0)
mel = to_logmel(y, 16000, 128, 512, 256, 60, 8000)
x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # [1,1,128,T]

# Predict
with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(1).item()
    prob = torch.softmax(logits, 1).max(1)[0].item()
print(f"Prediction: {pred}, Confidence: {prob:.3f}")
```

### ONNX Inference
1. **Export**:
   ```bash
   python export_onnx.py --ckpt lite_cnn_v1.pt --onnx lite_cnn_v1_fixed.onnx --width_mult 1.0 --dropout 0.2
   # For Wav2Vec2: add --use_wav2vec2
   ```

2. **Quantize** (CNN only):
   ```bash
   python quantize_onnx.py --onnx_in lite_cnn_v1_fixed.onnx --onnx_out lite_cnn_v1_int8.onnx
   ```

3. **Batch Inference**:
   ```bash
   python infer_dataset.py --onnx lite_cnn_v1_fixed.onnx --output_csv data/lite_cnn_v1_predictions.csv
   ```

4. **Single File**:
   ```bash
   python infer.py --audio_path path/to/audio.wav --onnx lite_cnn_v1_fixed.onnx
   ```

### Performance Notes
- FP32 ONNX: ~10-50ms inference on CPU.
- INT8: ~5-20ms, 50-70% size reduction (CNN only).
- Wav2Vec2: Larger (380MB), slower (~100ms), but more accurate.

## File Structure
Detailed breakdown of all files:

- **`config.py`**: Defines `CFG` dataclass with hyperparameters (e.g., sample_rate=16000, batch_size=8). Includes `seed_all()` for reproducibility and `get_device()` for MPS/CUDA/CPU detection.
- **`dataset.py`**: `HelmitAudioDS` class loads audio, applies preprocessing/augmentation. Supports mel-spec or raw audio based on `return_raw_audio`.
- **`model.py`**: Defines `LiteAudioCNN` and `Wav2Vec2Classifier`. Includes forward passes and optional feature extraction.
- **`train.py`**: Main training script. Handles data loading, optimization, logging, and checkpointing.
- **`export_onnx.py`**: Exports PyTorch models to ONNX with dynamic shapes. Requires model-specific flags.
- **`quantize_onnx.py`**: Applies INT8 quantization using ONNX Runtime. Cleans model graph to avoid shape issues.
- **`infer.py`**: Single-file inference with ONNX. Loads audio, preprocesses, predicts, and prints result.
- **`infer_dataset.py`**: Batch inference on full dataset. Saves predictions to CSV for evaluation.
- **`compute_metrics.py`**: Computes and saves metrics (accuracy, F1, etc.) from prediction CSVs.
- **`utils_audio.py`**: Helper functions: `load_audio()`, `to_logmel()`, `crop_or_pad()`, `add_noise()`.
- **`prep_dataset.py`**: Scans `data/` folders and generates `labels.csv`.
- **`download_from_drive.py`**: Downloads dataset zips from Google Drive links.
- **`requirements.txt`**: List of Python packages with versions.
- **`model_configs.txt`**: Summary of training configs for each model.
- **`model_metrics.txt`**: Performance metrics for all models.
- **`insights.md`**: Analysis of model bias and improvement suggestions.
- **`data/`**: 
  - `labels.csv`: Ground truth labels.
  - `*_predictions.csv`: Inference results (path, label, pred, prob).
- **`*.pt`**: PyTorch model checkpoints.
- **`*_fixed.onnx`**: FP32 ONNX models (dynamic batch/time).
- **`*_int8.onnx`**: Quantized INT8 ONNX (CNN only).

## Results
### Performance Summary
| Model          | Accuracy | Precision | Recall | F1 Score | Notes |
|----------------|----------|-----------|--------|----------|-------|
| lite_cnn_v1   | 0.6000  | 0.6000   | 1.0000| 0.7500  | Predicts all harmful |
| lite_cnn_v2   | 0.4000  | 0.0000   | 0.0000| 0.0000  | Predicts all safe |
| lite_cnn_v3   | 0.4000  | 0.0000   | 0.0000| 0.0000  | Predicts all safe |
| lite_cnn_v4   | 0.6000  | 0.6000   | 1.0000| 0.7500  | Predicts all harmful |
| wav2vec2_v1   | 0.6000  | 0.6000   | 1.0000| 0.7500  | Predicts all harmful |
| wav2vec2_v2   | 0.6000  | 0.6000   | 1.0000| 0.7500  | Predicts all harmful |

- **Top Models**: lite_cnn_v1/v4, wav2vec2_v1/v2 (F1=0.75).
- **Bias Issue**: Models favor majority class (harmful) due to imbalance. See `insights.md` for details.
- **Training RAM/Time**: CNN ~250-300MB, 40-50s/epoch; Wav2Vec2 ~400-500MB, 35-40s/epoch.

### Confusion Matrices (Example for lite_cnn_v1)
- True Safe: 0 predicted safe, 40 predicted harmful.
- True Harmful: 60 predicted harmful, 0 predicted safe.

## Troubleshooting
- **Import Errors**: Ensure all packages in `requirements.txt` are installed. Use `pip install --upgrade` if needed.
- **ONNX Export Fails**: Check model loading; ensure correct `width_mult`/`dropout` for CNN.
- **Quantization Fails**: Common for Wav2Vec2; use FP32. Clean model graph if issues persist.
- **Out of Memory**: Reduce `batch_size` (e.g., to 1), use CPU, or enable gradient checkpointing.
- **Low Accuracy/Bias**: Increase data, tune LR (try 1e-5), add more augmentation, use focal loss.
- **Wav2Vec2 Slow**: Freeze more layers or use smaller batch size.
- **Audio Loading Issues**: Ensure `librosa` and `soundfile` are installed; check file formats.
- **MPS Issues**: Apple Silicon may have bugs; switch to CPU if training fails.

## FAQ
- **Why 60% accuracy?** Class imbalance causes models to predict majority class.
- **Can I use GPU?** Yes, set `device='cuda'` in `config.py`.
- **How to add more data?** Place in `data/harmful/` or `data/safe/`, rerun `prep_dataset.py`.
- **ONNX vs PyTorch?** ONNX for deployment; PyTorch for development.
- **Why not higher F1?** Small dataset; need 1000+ samples for robust training.

## Contributing
- **Data**: Contribute more labeled audio samples.
- **Models**: Experiment with HuBERT, Whisper, or larger Wav2Vec2 variants.
- **Features**: Add cross-validation, hyperparameter sweeps, or multi-class support.
- **Code**: Follow PEP8, add docstrings, test on multiple devices.
- **Issues**: Report bugs with logs; suggest features via issues.

## License
This project is open-source under the MIT License. See LICENSE file for details.

## Changelog
- v1.0: Initial release with CNN and Wav2Vec2 models, ONNX export, basic inference.