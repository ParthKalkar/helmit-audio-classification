import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
import librosa
from utils_audio import load_audio, crop_or_pad, to_logmel, add_noise

class HelmitAudioDS(Dataset):
    """
    Expects labels.csv with two columns: filepath,label
    where filepath is a relative path from DATA_ROOT (e.g., harmful/x.mp3)
    """
    def __init__(self, data_root, labels_csv, cfg, augment: bool, return_raw_audio: bool = False):
        self.data_root = data_root
        df = pd.read_csv(os.path.join(data_root, labels_csv))
        # Support either (filename,label) OR (filepath,label)
        if "filepath" in df.columns:
            self.paths = df["filepath"].tolist()
        else:
            # fallback for (filename,label) relative to AUDIO_DIR; not used by default
            audio_dir = getattr(cfg, "AUDIO_DIR", ".")
            self.paths = [os.path.join(audio_dir, fn) for fn in df["filename"].tolist()]
        self.labels = df["label"].astype(int).tolist()
        self.cfg = cfg
        self.augment = augment
        self.return_raw_audio = return_raw_audio

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        rel_path = self.paths[idx]
        label = self.labels[idx]
        path = os.path.join(self.data_root, rel_path)

        # load full clip, crop/pad window
        y = load_audio(path, sr=self.cfg.SAMPLE_RATE, mono=self.cfg.MONO, target_sec=None)
        target_len = int(self.cfg.SAMPLE_RATE * self.cfg.TARGET_SEC)
        y = crop_or_pad(y, target_len, center=(not self.augment))

        # augment (train only)
        if self.augment and np.random.rand() < self.cfg.AUG_PROB:
            if np.random.rand() < 0.5:
                rate = np.random.uniform(*self.cfg.AUG_TIME_STRETCH)
                # librosa.effects.time_stretch requires rate as a keyword in newer versions
                y = librosa.effects.time_stretch(y, rate=rate)
                y = crop_or_pad(y, target_len, center=False)
            if np.random.rand() < 0.5:
                steps = np.random.randint(self.cfg.AUG_PITCH_STEPS[0], self.cfg.AUG_PITCH_STEPS[1]+1)
                if steps != 0:
                    y = librosa.effects.pitch_shift(y, sr=self.cfg.SAMPLE_RATE, n_steps=steps)
                    y = crop_or_pad(y, target_len, center=False)
            if np.random.rand() < 0.5:
                snr = np.random.uniform(*self.cfg.AUG_NOISE_SNR_DB)
                y = add_noise(y, snr)

        if self.return_raw_audio:
            x = torch.from_numpy(y).float()  # raw audio waveform
        else:
            mel = to_logmel(y, self.cfg.SAMPLE_RATE, self.cfg.N_MELS, self.cfg.N_FFT,
                            self.cfg.HOP_LENGTH, self.cfg.FMIN, self.cfg.FMAX)

            # SpecAugment
            if self.augment and np.random.rand() < 0.5:
                mel = mel.copy()
                fmask = np.random.randint(0, self.cfg.SPEC_FREQ_MASK+1)
                if fmask > 0:
                    f0 = np.random.randint(0, max(1, mel.shape[0]-fmask))
                    mel[f0:f0+fmask, :] = 0
                tmask = np.random.randint(0, self.cfg.SPEC_TIME_MASK+1)
                if tmask > 0:
                    t0 = np.random.randint(0, max(1, mel.shape[1]-tmask))
                    mel[:, t0:t0+tmask] = 0

            x = torch.from_numpy(mel).unsqueeze(0)  # [1, n_mels, T]

        y = torch.tensor(label, dtype=torch.long)
        return x, y, rel_path
