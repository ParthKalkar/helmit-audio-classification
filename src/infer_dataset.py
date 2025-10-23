import argparse, os, numpy as np, pandas as pd
import onnxruntime as ort
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from config import CFG, seed_all
from dataset import HelmitAudioDS

LABEL = {0: "safe", 1: "harmful"}

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--audio_dir", type=str, default=".")
    ap.add_argument("--labels_csv", type=str, default="labels.csv")
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--use_wav2vec2", action="store_true", help="Model is Wav2Vec2")
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    cfg = CFG(DATA_ROOT=args.data_root, AUDIO_DIR=args.audio_dir, LABELS_CSV=args.labels_csv)
    seed_all(cfg)

    # Load full dataset
    df = pd.read_csv(os.path.join(cfg.DATA_ROOT, cfg.LABELS_CSV))
    indices = list(df.index.values)

    ds = HelmitAudioDS(cfg.DATA_ROOT, cfg.LABELS_CSV, cfg, augment=False, return_raw_audio=args.use_wav2vec2)
    ds = Subset(ds, indices)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load ONNX model
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    results = []
    for x, y_true, rel_paths in loader:
        x = x.numpy()
        try:
            out = sess.run(None, {"input": x})[0]  # [batch, 2]
        except Exception as e:
            raise RuntimeError(f"ONNX runtime failed: {e}")
        probs = softmax(out)
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)

        for i in range(len(rel_paths)):
            results.append({
                "path": rel_paths[i],
                "label": int(y_true[i].item()),
                "pred": int(preds[i]),
                "prob": float(confs[i])
            })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")

    # Calculate accuracy
    acc = (results_df["label"] == results_df["pred"]).mean()
    print(f"Dataset Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
