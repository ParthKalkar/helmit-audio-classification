import numpy as np
import librosa

def load_audio(path: str, sr=16_000, mono=True, target_sec=None):
    y, src_sr = librosa.load(path, sr=None, mono=mono)
    if src_sr != sr:
        y = librosa.resample(y, orig_sr=src_sr, target_sr=sr)
    y = librosa.util.normalize(y)
    if target_sec is not None:
        target_len = int(sr * target_sec)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
    return y.astype(np.float32)

def crop_or_pad(y: np.ndarray, target_len: int, center: bool):
    if len(y) == target_len:
        return y
    if len(y) < target_len:
        return np.pad(y, (0, target_len-len(y)))
    if center:
        start = (len(y)-target_len)//2
    else:
        start = np.random.randint(0, len(y)-target_len+1)
    return y[start:start+target_len]

def to_logmel(y: np.ndarray, sr: int, n_mels=128, n_fft=512, hop=256, fmin=60, fmax=8000):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop, fmin=fmin, fmax=fmax)
    logS = librosa.power_to_db(S + 1e-10)
    m, s = logS.mean(), logS.std() + 1e-6
    return ((logS - m) / s).astype(np.float32)

def add_noise(y, snr_db):
    rms = np.sqrt(np.mean(y**2) + 1e-12)
    noise_rms = rms / (10**(snr_db/20))
    noise = np.random.normal(0, noise_rms, y.shape).astype(np.float32)
    return (y + noise).astype(np.float32)
