import argparse, numpy as np, onnxruntime as ort
from config import CFG
from utils_audio import load_audio, to_logmel

LABEL = {0:"safe", 1:"harmful"}

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=1, keepdims=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_path", type=str, required=True)
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--window_sec", type=float, default=8.0)
    ap.add_argument("--stride_sec", type=float, default=4.0)
    ap.add_argument("--hop_length", type=int, default=256)
    args = ap.parse_args()

    cfg = CFG(HOP_LENGTH=args.hop_length)
    y = load_audio(args.audio_path, sr=cfg.SAMPLE_RATE, mono=cfg.MONO, target_sec=None)
    sr = cfg.SAMPLE_RATE
    win = int(args.window_sec * sr)
    stride = int(args.stride_sec * sr)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    probs, times = [], []
    for i in range(0, max(1, len(y)-win+1), stride):
        seg = y[i:i+win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win-len(seg)))
        mel = to_logmel(seg, sr, cfg.N_MELS, cfg.N_FFT, cfg.HOP_LENGTH, cfg.FMIN, cfg.FMAX)
        x = mel[None, None, ...].astype(np.float32)
        out = sess.run(None, {"input": x})[0]
        p = softmax(out)[0,1]
        probs.append(float(p))
        times.append((i/sr, min((i+win)/sr, len(y)/sr)))

    avg_p = float(np.mean(probs)) if probs else 0.0
    label = LABEL[1] if avg_p >= 0.5 else LABEL[0]
    print(f"Sliding-window result: {label} (avg harmful prob={avg_p:.3f}, windows={len(probs)})")

if __name__ == "__main__":
    main()
