import os, time, argparse, psutil
import torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

from config import CFG, seed_all, get_device
from dataset import HelmitAudioDS
from model import LiteAudioCNN, Wav2Vec2Classifier

def measure_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024*1024)

class EarlyStopper:
    def __init__(self, patience=12, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.count = 0
    def step(self, metric):
        if self.best is None or (metric - self.best) > self.min_delta:
            self.best = metric; self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    crit = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x,y,_ in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += crit(logits, y).item() * x.size(0)
            n += x.size(0)
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
    prec, rec, f1, _ = precision_recall_fscore_support(ys, ps, average='binary', pos_label=1, zero_division=0)
    rpt = classification_report(ys, ps, digits=4, zero_division=0)
    cm = confusion_matrix(ys, ps)
    return loss_sum/max(1,n), prec, rec, f1, rpt, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--audio_dir", type=str, default=".")
    ap.add_argument("--labels_csv", type=str, default="labels.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--target_sec", type=float, default=8.0)
    ap.add_argument("--hop_length", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="./lite_cnn.pt")
    ap.add_argument("--width_mult", type=float, default=0.6, help="Model width multiplier")
    ap.add_argument("--dropout", type=float, default=0.3, help="Dropout probability for the model")
    ap.add_argument("--use_sampler", action="store_true", help="Use WeightedRandomSampler for balanced training")
    ap.add_argument("--accumulate_steps", type=int, default=1, help="Gradient accumulation steps")
    ap.add_argument("--use_autocast", action="store_true", help="Use autocast during evaluation (mps/cuda)")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--use_wav2vec2", action="store_true", help="Use Wav2Vec2 pre-trained model instead of custom CNN")
    ap.add_argument("--num_unfrozen_layers", type=int, default=0, help="Number of last layers to unfreeze in Wav2Vec2")
    args = ap.parse_args()

    cfg = CFG(DATA_ROOT=args.data_root, AUDIO_DIR=args.audio_dir, LABELS_CSV=args.labels_csv,
              EPOCHS=args.epochs, BATCH_SIZE=args.batch_size, TARGET_SEC=args.target_sec,
              HOP_LENGTH=args.hop_length, CKPT_PATH=args.ckpt if not args.use_wav2vec2 else "./wav2vec2.pt")
    seed_all(cfg)
    # allow CLI override of learning rate
    if args.lr is not None:
        cfg.LR = float(args.lr)

    # Resolve device and adapt some defaults for MPS
    device = get_device(cfg)
    # If using MPS, lower default batch size for memory constraints
    if device.type == "mps":
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        if args.use_wav2vec2:
            cfg.BATCH_SIZE = min(cfg.BATCH_SIZE, 2)  # Wav2Vec2 is more memory intensive
        elif cfg.BATCH_SIZE > 4:
            print("Adjusting batch size down to 4 for MPS device")
            cfg.BATCH_SIZE = 4

    # Stratified split
    df = pd.read_csv(os.path.join(cfg.DATA_ROOT, cfg.LABELS_CSV))
    y = df["label"].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED)
    train_idx, val_idx = next(sss.split(df.index, y))

    # Show label distributions to diagnose imbalance
    train_labels = y[train_idx]
    val_labels = y[val_idx]
    print("Train labels distribution:", np.bincount(train_labels))
    print("Val labels distribution:  ", np.bincount(val_labels))

    train_ds = HelmitAudioDS(cfg.DATA_ROOT, cfg.LABELS_CSV, cfg, augment=True, return_raw_audio=args.use_wav2vec2)
    val_ds   = HelmitAudioDS(cfg.DATA_ROOT, cfg.LABELS_CSV, cfg, augment=False, return_raw_audio=args.use_wav2vec2)

    train_ds = Subset(train_ds, train_idx)
    val_ds   = Subset(val_ds,   val_idx)

    # optionally use sampler for balanced training
    use_sampler = bool(args.use_sampler)

    # pin_memory is helpful for CUDA; not needed for MPS/CPU
    pin_memory = True if device.type == "cuda" else False
    if use_sampler:
        # compute per-sample weights inversely proportional to class frequency
        sample_weights = np.array([1.0 / max(1, np.bincount(train_labels)[lbl]) for lbl in train_labels], dtype=np.float32)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, sampler=sampler,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.NUM_WORKERS, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=pin_memory)

    if args.use_wav2vec2:
        model = Wav2Vec2Classifier(n_classes=2, num_unfrozen_layers=args.num_unfrozen_layers).to(device)
    else:
        model = LiteAudioCNN(n_mels=cfg.N_MELS, n_classes=2, width_mult=float(args.width_mult), dropout=float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    # Compute class weights from training set to mitigate imbalance
    try:
        counts = np.bincount(train_labels)
        total = counts.sum()
        if len(counts) < 2:
            counts = np.array(list(counts) + [1])
        # weight inversely proportional to class frequency
        weights = total / (2.0 * counts.astype(np.float32))
        weight_t = torch.tensor(weights, dtype=torch.float32).to(device)
        crit = nn.CrossEntropyLoss(weight=weight_t)
        print("Using class weights:", weights.tolist())
    except Exception:
        crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5)

    accum_steps = max(1, int(args.accumulate_steps))
    use_autocast = bool(args.use_autocast)

    best_f1, best_state = -1.0, None
    es = EarlyStopper(patience=12)

    print(f"RAM start: {measure_ram_mb():.1f} MB")
    for epoch in range(1, cfg.EPOCHS+1):
        t0 = time.time()
        # train
        model.train()
        total, correct, n = 0.0, 0, 0
        opt.zero_grad()
        for step, (x,y,_) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)
            # forward (optionally no-autocast for training; autocast is used for eval/test only)
            logits = model(x)
            loss_raw = crit(logits, y)
            loss = loss_raw / accum_steps
            loss.backward()

            # update when we've accumulated enough gradients
            if (step + 1) % accum_steps == 0:
                opt.step()
                opt.zero_grad()

            total += loss_raw.item()*x.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            n += x.size(0)
        tr_loss, tr_acc = total/max(1,n), correct/max(1,n)

        # eval
        # evaluation: optionally use autocast for faster inference on mps/cuda
        if use_autocast and device.type in ("mps", "cuda"):
            try:
                with torch.autocast(device_type=device.type):
                    val_loss, P, R, F1, rpt, cm = evaluate(model, val_loader, device)
            except Exception:
                val_loss, P, R, F1, rpt, cm = evaluate(model, val_loader, device)
        else:
            val_loss, P, R, F1, rpt, cm = evaluate(model, val_loader, device)
        sched.step(F1)
        dt = time.time()-t0
        print(f"[{epoch:02d}] tr_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={val_loss:.4f} P={P:.3f} R={R:.3f} F1={F1:.3f} | {dt:.1f}s | RAM {measure_ram_mb():.1f}MB")

        if F1 > best_f1:
            best_f1 = F1
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            torch.save(best_state, cfg.CKPT_PATH)
        if es.step(F1):
            print("Early stopping.")
            break

    print("\nBest F1:", best_f1)
    print("Saved:", cfg.CKPT_PATH)

if __name__ == "__main__":
    main()
