import argparse, os, torch, onnx, numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

from config import CFG, seed_all
from dataset import HelmitAudioDS
from model import LiteAudioCNN, Wav2Vec2Classifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--audio_dir", type=str, default=".")
    ap.add_argument("--labels_csv", type=str, default="labels.csv")
    ap.add_argument("--ckpt", type=str, default="./lite_cnn.pt")
    ap.add_argument("--onnx", type=str, default="./lite_cnn.onnx")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--use_wav2vec2", action="store_true", help="Model is Wav2Vec2")
    ap.add_argument("--width_mult", type=float, default=0.6, help="Width multiplier for CNN")
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout for CNN")
    args = ap.parse_args()

    cfg = CFG(DATA_ROOT=args.data_root, AUDIO_DIR=args.audio_dir, LABELS_CSV=args.labels_csv,
              CKPT_PATH=args.ckpt, ONNX_PATH=args.onnx, ONNX_OPSET=args.opset)
    seed_all(cfg)

    # small loader to get a real dummy shape
    df = pd.read_csv(os.path.join(cfg.DATA_ROOT, cfg.LABELS_CSV))
    y = df["label"].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.SEED)
    train_idx, _ = next(sss.split(df.index, y))

    ds = HelmitAudioDS(cfg.DATA_ROOT, cfg.LABELS_CSV, cfg, augment=False, return_raw_audio=args.use_wav2vec2)
    ds = Subset(ds, train_idx[:1])  # one item is enough
    x, _, _ = ds[0]
    dummy = x.unsqueeze(0)          # [1,1,128,T] for CNN or [1, time] for Wav2Vec2

    if args.use_wav2vec2:
        model = Wav2Vec2Classifier(n_classes=2)
    else:
        model = LiteAudioCNN(n_mels=cfg.N_MELS, n_classes=2, width_mult=args.width_mult, dropout=args.dropout)
    model.load_state_dict(torch.load(cfg.CKPT_PATH, map_location="cpu"))
    model.eval()

    # provide an example output so the exporter doesn't freeze the batch dim
    example_tensor = torch.tensor(np.asarray(dummy))
    example_out = model(example_tensor).detach()

    torch.onnx.export(
        model, (dummy,), cfg.ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        opset_version=cfg.ONNX_OPSET,
        do_constant_folding=True,
        keep_initializers_as_inputs=False
    )
    print("Saved ONNX:", cfg.ONNX_PATH)
    onnx.checker.check_model(onnx.load(cfg.ONNX_PATH))
    print("ONNX check: OK")

if __name__ == "__main__":
    main()
