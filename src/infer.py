import argparse, numpy as np
import onnxruntime as ort
import torch
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
    ap.add_argument("--target_sec", type=float, default=8.0)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    cfg = CFG(TARGET_SEC=args.target_sec, HOP_LENGTH=args.hop_length)

    y = load_audio(args.audio_path, sr=cfg.SAMPLE_RATE, mono=cfg.MONO, target_sec=cfg.TARGET_SEC)
    mel = to_logmel(y, cfg.SAMPLE_RATE, cfg.N_MELS, cfg.N_FFT, cfg.HOP_LENGTH, cfg.FMIN, cfg.FMAX)
    x = mel[None, None, ...].astype(np.float32)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x})[0]  # [1,2]
    p = softmax(out)[0]
    pred = int(np.argmax(p)); conf = float(np.max(p))
    expl = "Classified as safe." if pred==0 else "Flagged as harmful based on acoustic patterns."
    print(f"Prediction: {LABEL[pred]}  prob={conf:.3f}")
    print(f"Explanation: {expl}")

if __name__ == "__main__":
    main()
