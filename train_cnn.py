"""
train_cnn.py - basic CNN (built from scratch) for the BD-PaveSurface V2 dataset.

Examples (run from the project folder, with the conda environment active):
    python train_cnn.py --data data --quick          # 1-2 minute test of the whole pipeline
    python train_cnn.py --data data                  # full run (with + without augmentation)
    python train_cnn.py --data data --img-size 64 --epochs 15      # lighter run for a CPU-only PC

Everything is saved in the --out folder (default: outputs).
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # save figures to files (no pop-up windows)
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# Class / split / augmented-file detection from folder names. Edit if your folders are named differently.
CLASS_PATTERNS = {"Crack": r"crack", "Pothole": r"pothole", "Good": r"good|normal|smooth|intact|healthy"}
SPLIT_PATTERNS = {"train": r"^train", "val": r"^val", "test": r"^test"}
# A file counts as augmented if its FILE NAME starts with / contains "aug" as a separate word (e.g. aug__0001__...),
# or if one of its folders is named exactly like "aug", "augmented", "augmentation".
# (A folder such as "Processed Augmented Data" must NOT count, it holds the whole dataset.)
AUG_FILE_PATTERN = r"(^|[_\-\s.])aug(ment|mented|mentation)?([_\-\s.\d]|$)"
AUG_DIR_PATTERN = r"^aug(ment|mented|mentation)?s?$"


# --------------------------------------------------------------------------- data
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def detect(parts, patterns):
    hits = {name for name, pat in patterns.items() for part in parts if re.search(pat, part, re.I)}
    return hits.pop() if len(hits) == 1 else None


def build_table(root, path_include=None, path_exclude=None, use_provided_aug=True):
    rows = []
    for p in root.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in IMG_EXTS) or "__MACOSX" in p.parts:
            continue
        rel = p.relative_to(root)
        s = rel.as_posix()
        if path_include and not re.search(path_include, s, re.I):
            continue
        if path_exclude and re.search(path_exclude, s, re.I):
            continue
        parts = [x.lower() for x in rel.parts]
        dirs, stem = parts[:-1], Path(parts[-1]).stem
        rows.append(dict(path=str(p), rel=s,
                         cls=detect(dirs, CLASS_PATTERNS) or detect([stem], CLASS_PATTERNS),
                         split=detect(dirs, SPLIT_PATTERNS),
                         is_aug=bool(re.search(AUG_FILE_PATTERN, stem)) or
                                any(re.search(AUG_DIR_PATTERN, d) for d in dirs)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No images found under {root.resolve()}. Check the --data path "
                         f"(run inspect_dataset.py first).")

    unk = df[df.cls.isna()]
    if len(unk):
        print(f"WARNING: {len(unk)} images have no detectable class and are ignored. Examples:")
        print("   ", unk.rel.head(5).tolist())
        print("   -> edit CLASS_PATTERNS at the top of this file to match your folder names.")
    df = df[df.cls.notna()].copy()
    if not use_provided_aug:
        df = df[~df.is_aug].copy()
    if df.empty:
        raise SystemExit("No images with a detectable class. Edit CLASS_PATTERNS in train_cnn.py.")

    class_names = sorted(df.cls.unique())
    df["label"] = df.cls.map({c: i for i, c in enumerate(class_names)})
    return df.reset_index(drop=True), class_names


def make_split(df, seed):
    if df.split.notna().all():
        tr = np.where(df.split.values == "train")[0]
        va = np.where(df.split.values == "val")[0]
        te = np.where(df.split.values == "test")[0]
        note = "provided split (from the dataset authors)"
        if len(df.iloc[np.concatenate([va, te])].query("is_aug")):
            raise SystemExit("Augmented images found in val/test - check PATH_INCLUDE / folder names.")
    else:
        print("WARNING: no complete train/val/test split found in the folder names -> using a RANDOM split.")
        print("         Near-duplicate photos of the same scene can end up in train AND test and inflate accuracy.")
        note = "random stratified split (image level) - possible leakage"
        base = np.where(~df.is_aug.values)[0]
        aug = np.where(df.is_aug.values)[0]
        tr_b, tmp = train_test_split(base, test_size=0.30, stratify=df.label.values[base], random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=df.label.values[tmp], random_state=seed)
        tr = np.concatenate([tr_b, aug])
    if min(len(tr), len(va), len(te)) == 0:
        raise SystemExit("One of train/val/test is empty. Check the split detection output above.")
    return tr, va, te, note


def load_images(paths, size):
    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            out[i] = np.array(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
        if (i + 1) % 2000 == 0:
            print(f"   loaded {i + 1}/{len(paths)} images")
    return out


class PaveDS(Dataset):
    def __init__(self, X, y, mean, std, augment=False):
        self.X = torch.from_numpy(X).permute(0, 3, 1, 2).contiguous()      # N x 3 x H x W (uint8)
        self.y = torch.from_numpy(y).long()
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.augment = augment
        self.tf = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            v2.ColorJitter(brightness=0.25, contrast=0.25),
        ])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = self.tf(x)
        return (x.float() / 255.0 - self.mean) / self.std, self.y[i]


# --------------------------------------------------------------------------- model
class BasicCNN(nn.Module):
    """4 blocks of (Conv-BN-ReLU) x2 + MaxPool, global average pooling, small classifier. No pretrained weights."""

    def __init__(self, n_classes, p_drop=0.4):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(p_drop),
            nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- training
def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    tot_loss = tot_correct = n = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tot_loss += loss.item() * len(yb)
            tot_correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return tot_loss / n, tot_correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for xb, yb in loader:
        P.append(model(xb.to(device)).argmax(1).cpu())
        Y.append(yb)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def fit_and_eval(tag, augment, data, n_classes, device, args):
    set_seed(args.seed)
    Xtr, ytr, Xva, yva, Xte, yte, mean, std = data
    bs = max(2, min(args.batch_size, len(ytr) // 2))
    train_dl = DataLoader(PaveDS(Xtr, ytr, mean, std, augment), batch_size=bs, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(PaveDS(Xva, yva, mean, std), batch_size=bs, num_workers=args.workers)
    test_dl = DataLoader(PaveDS(Xte, yte, mean, std), batch_size=bs, num_workers=args.workers)

    model = BasicCNN(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

    hist = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc"]}
    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, opt)
        vl, va = run_epoch(model, val_dl, device)
        sched.step(va)
        for k, v in zip(hist, [tl, ta, vl, va]):
            hist[k].append(v)
        print(f"[{tag}] epoch {ep:02d}/{args.epochs} | train loss {tl:.3f} acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {time.time() - t0:.0f}s")
        if va > best:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{tag}] early stopping at epoch {ep} (no val improvement for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    pred, true = predict(model, test_dl, device)
    res = dict(tag=tag, model=model, hist=hist, pred=pred, true=true, best_val_acc=float(best),
               test_acc=float(accuracy_score(true, pred)),
               test_macro_f1=float(f1_score(true, pred, average="macro")),
               epochs_run=len(hist["val_acc"]), seconds=round(time.time() - t0, 1))
    print(f"[{tag}] TEST accuracy {res['test_acc']:.4f} | macro F1 {res['test_macro_f1']:.4f}\n")
    return res


# --------------------------------------------------------------------------- plots
def plot_curves(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for tag, r in results.items():
        axes[0].plot(r["hist"]["train_loss"], label=f"{tag} train")
        axes[0].plot(r["hist"]["val_loss"], "--", label=f"{tag} val")
        axes[1].plot(r["hist"]["train_acc"], label=f"{tag} train")
        axes[1].plot(r["hist"]["val_acc"], "--", label=f"{tag} val")
    axes[0].set_title("Loss")
    axes[1].set_title("Accuracy")
    for a in axes:
        a.set_xlabel("epoch")
        a.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, class_names, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(class_names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, M, title, fmt in [(axes[0], cm, "counts", "d"), (axes[1], cm_norm, "row-normalized", ".2f")]:
        ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(n))
        ax.set_xticklabels(class_names)
        ax.set_yticks(range(n))
        ax.set_yticklabels(class_names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"Confusion matrix ({title})")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(M[i, j], fmt), ha="center", va="center",
                        color="white" if M[i, j] > M.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_misclassified(X, true, pred, class_names, path, n=12):
    wrong = np.where(pred != true)[0][:n]
    if len(wrong) == 0:
        return
    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(X[i])
        ax.set_title(f"true: {class_names[true[i]]}\npred: {class_names[pred[i]]}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Basic CNN from scratch on BD-PaveSurface V2")
    ap.add_argument("--data", required=True, help="folder containing the unzipped dataset")
    ap.add_argument("--out", default="outputs", help="folder for results (created automatically)")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (keep 0 on Windows)")
    ap.add_argument("--path-include", default=None, help="regex: keep only image paths matching this")
    ap.add_argument("--path-exclude", default=None, help="regex: drop image paths matching this")
    ap.add_argument("--no-provided-aug", action="store_true", help="ignore the dataset's augmented images")
    ap.add_argument("--skip-ablation", action="store_true", help="train only the augmented model")
    ap.add_argument("--quick", action="store_true", help="tiny subset + 3 epochs to test the pipeline")
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "|", torch.cuda.get_device_name(0) if device.type == "cuda" else
          "no NVIDIA GPU found - training on CPU (use --img-size 64 --epochs 15 if it is too slow)")

    # 1) find images, classes, split
    df, class_names = build_table(Path(args.data), args.path_include, args.path_exclude, not args.no_provided_aug)
    if args.quick:
        parts = [g.sample(min(len(g), 40), random_state=args.seed)
                 for _, g in df.groupby([df["cls"], df["split"].fillna("none")])]
        df = pd.concat(parts).reset_index(drop=True)
        args.epochs = min(args.epochs, 3)
        print("QUICK MODE: small subset, 3 epochs (results are NOT meaningful)")
    n_classes = len(class_names)
    print(f"\nImages: {len(df)} | classes: {class_names}")
    print(df.fillna({"split": "none"}).groupby(["cls", "split", "is_aug"]).size().unstack(fill_value=0))
    print("Course minimum of 2,000 images:", "OK" if len(df) >= 2000 else "NOT MET")

    tr, va, te, split_note = make_split(df, args.seed)
    print(f"\nSplit: {split_note}\ntrain / val / test = {len(tr)} / {len(va)} / {len(te)}")
    df["split_used"] = "train"
    df.loc[va, "split_used"] = "val"
    df.loc[te, "split_used"] = "test"
    df.drop(columns=["path"]).to_csv(out / "split_used.csv", index=False)

    # 2) load images once (resized, RGB)
    print(f"\nLoading images at {args.img_size}x{args.img_size} ...")
    t0 = time.time()
    X = load_images(df.path.tolist(), args.img_size)
    y = df.label.values
    print(f"Done in {time.time() - t0:.0f}s ({X.nbytes / 1e6:.0f} MB in memory)\n")
    Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]
    mean = (Xtr.reshape(-1, 3).mean(0) / 255.0).tolist()
    std = (Xtr.reshape(-1, 3).std(0) / 255.0).tolist()
    data = (Xtr, ytr, Xva, yva, Xte, yte, mean, std)

    n_params = sum(p.numel() for p in BasicCNN(n_classes).parameters())
    print(f"CNN trainable parameters: {n_params:,}\n")

    # 3) train
    results = {"with_aug": fit_and_eval("with_aug", True, data, n_classes, device, args)}
    if not args.skip_ablation:
        results["no_aug"] = fit_and_eval("no_aug", False, data, n_classes, device, args)

    # 4) analysis + saving
    main_res = results["with_aug"]
    pred, true = main_res["pred"], main_res["true"]
    report = classification_report(true, pred, target_names=class_names, digits=3, zero_division=0)
    print("Classification report (main model, test set):\n" + report)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(true, pred, labels=list(range(n_classes)))
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                print(f"  {cm[i, j]} {class_names[i]} images predicted as {class_names[j]}")

    plot_curves(results, out / "learning_curves.png")
    plot_confusion(cm, class_names, out / "confusion_matrix.png")
    plot_misclassified(Xte, true, pred, class_names, out / "misclassified_examples.png")
    torch.save(main_res["model"].state_dict(), out / "cnn_with_aug.pt")

    summary = {r["tag"]: {k: r[k] for k in ["best_val_acc", "test_acc", "test_macro_f1", "epochs_run", "seconds"]}
               for r in results.values()}
    summary["setup"] = dict(split=split_note, img_size=args.img_size, seed=args.seed, n_images=int(len(df)),
                            n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                            classes=class_names, device=str(device), quick=args.quick)
    (out / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved to:", out.resolve())
    for f in sorted(out.iterdir()):
        print("   ", f.name)


if __name__ == "__main__":
    main()
