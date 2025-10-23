from dataclasses import dataclass
from typing import Optional, Tuple
import torch, random, numpy as np

@dataclass
class CFG:
    # Data
    DATA_ROOT: str = "./data"
    AUDIO_DIR: str = "."            # we read from harmful/ and safe/ via labels.csv paths
    LABELS_CSV: str = "labels.csv"

    # Audio
    SAMPLE_RATE: int = 16_000
    MONO: bool = True
    TARGET_SEC: float = 8.0         # 8-10s recommended for long clips
    N_MELS: int = 128
    N_FFT: int = 512
    HOP_LENGTH: int = 256           # ~16ms hop for lighter compute
    FMIN: int = 60
    FMAX: Optional[int] = 8000

    # Training
    EPOCHS: int = 40
    BATCH_SIZE: int = 8
    LR: float = 1e-3
    WD: float = 1e-4
    VAL_SPLIT: float = 0.2
    SEED: int = 42
    DEVICE: str = "auto"            # "auto" will pick mps/cuda/cpu automatically
    NUM_WORKERS: int = 0            # macOS + libsndfile: 0 is safest

    # Augmentation
    AUG_PROB: float = 0.7
    AUG_TIME_STRETCH: Tuple[float, float] = (0.85, 1.2)
    AUG_PITCH_STEPS: Tuple[int, int] = (-3, 3)
    AUG_NOISE_SNR_DB: Tuple[int, int] = (18, 32)
    SPEC_FREQ_MASK: int = 15
    SPEC_TIME_MASK: int = 35

    # Artifacts
    CKPT_PATH: str = "./lite_cnn.pt"
    ONNX_PATH: str = "./lite_cnn.onnx"
    ONNX_INT8_PATH: str = "./lite_cnn_int8.onnx"
    ONNX_OPSET: int = 13

    # Explainability
    TOP_TIME_SEGMENTS: int = 3

def seed_all(cfg: CFG):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

def get_device(cfg: CFG):
    """Return a torch.device depending on cfg.DEVICE (auto/mps/cuda/cpu)."""
    dev = cfg.DEVICE
    if dev == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if dev == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
