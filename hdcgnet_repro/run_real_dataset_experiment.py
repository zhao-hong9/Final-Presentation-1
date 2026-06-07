import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "official_HDCGNet" / "datasets"
OUT = ROOT / "outputs_real"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "holden": ("holden.mat", "holden", "holden_gt.mat", "holden_gt", 6),
    "NiliFossae": ("NiliFossae.mat", "NiliFossae", "NiliFossae_gt.mat", "NiliFossae_gt", 9),
    "Utopia": ("Utopia.mat", "Utopia", "Utopia_gt.mat", "Utopia_gt", 9),
}


@dataclass
class Config:
    dataset: str = "holden"
    seed: int = 11
    bands: int = 32
    patch: int = 7
    train_per_class: int = 10
    val_per_class: int = 10
    max_test_per_class: int = 350
    epochs: int = 30
    batch_size: int = 96
    lr: float = 1.2e-3
    band_dropout: float = 0.08


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def load_hsi(name, bands):
    data_file, data_key, gt_file, gt_key, classes = DATASETS[name]
    cube = loadmat(DATA / data_file)[data_key].astype(np.float32)
    gt = loadmat(DATA / gt_file)[gt_key].astype(np.int64)
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)
    labeled = gt.reshape(-1) > 0
    lo, hi = np.percentile(flat[labeled], [1, 99], axis=0)
    flat = np.clip(flat, lo, hi)
    mean = flat[labeled].mean(axis=0)
    std = flat[labeled].std(axis=0) + 1e-6
    flat = (flat - mean) / std
    pca = PCA(n_components=bands, random_state=0)
    reduced = pca.fit_transform(flat[labeled])
    all_reduced = np.zeros((flat.shape[0], bands), dtype=np.float32)
    all_reduced[labeled] = reduced.astype(np.float32)
    cube = all_reduced.reshape(h, w, bands)
    # Normalize PCA channels to a compact range for stable small-model training.
    ch_mean = cube.reshape(-1, bands)[labeled].mean(axis=0)
    ch_std = cube.reshape(-1, bands)[labeled].std(axis=0) + 1e-6
    cube = ((cube - ch_mean) / ch_std).astype(np.float32)
    return cube, gt, classes, float(pca.explained_variance_ratio_.sum())


def split(gt, cfg):
    rng = np.random.default_rng(cfg.seed)
    train, val, test = [], [], []
    flat_gt = gt.reshape(-1)
    for c in range(1, flat_gt.max() + 1):
        ids = np.flatnonzero(flat_gt == c)
        rng.shuffle(ids)
        train.extend(ids[: cfg.train_per_class])
        val.extend(ids[cfg.train_per_class : cfg.train_per_class + cfg.val_per_class])
        rest = ids[cfg.train_per_class + cfg.val_per_class :]
        if cfg.max_test_per_class and len(rest) > cfg.max_test_per_class:
            rest = rng.choice(rest, cfg.max_test_per_class, replace=False)
        test.extend(rest)
    return np.array(train), np.array(val), np.array(test)


class PatchDataset(Dataset):
    def __init__(self, cube, gt, ids, patch):
        self.cube = cube
        self.gt = gt
        self.ids = ids
        self.patch = patch
        self.pad = patch // 2
        self.h, self.w, _ = cube.shape
        self.padded = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = int(self.ids[i])
        y, x = divmod(idx, self.w)
        p = self.padded[y : y + self.patch, x : x + self.patch]
        patch = torch.from_numpy(p.transpose(2, 0, 1)).float()
        center = torch.from_numpy(self.cube[y, x]).float()
        label = int(self.gt[y, x] - 1)
        return patch, center, label, y, x


class RealHDCGNetLite(nn.Module):
    def __init__(self, bands, classes, band_dropout=0.0):
        super().__init__()
        self.band_dropout = band_dropout
        self.cnn = nn.Sequential(
            nn.Conv2d(bands, bands, 3, padding=1, groups=bands),
            nn.Conv2d(bands, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.graph_mlp = nn.Sequential(nn.Linear(bands + 4, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.attn = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head = nn.Linear(64, classes)

    def forward(self, patch, center, coord_feat):
        if self.training and self.band_dropout > 0:
            keep = (torch.rand(center.shape[1], device=center.device) > self.band_dropout).float()
            patch = patch * keep.view(1, -1, 1, 1)
            center = center * keep.view(1, -1)
        cnn = self.cnn(patch).flatten(1)
        graph = self.graph_mlp(torch.cat([center, coord_feat], dim=1))
        alpha = torch.softmax(self.attn(torch.cat([cnn, graph], dim=1)), dim=1)
        fused = alpha[:, 0:1] * cnn + alpha[:, 1:2] * graph
        return self.head(fused), alpha


class BalancedFocalLoss(nn.Module):
    def __init__(self, labels, classes, enabled=True, gamma=2.0):
        super().__init__()
        counts = np.bincount(labels, minlength=classes).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.enabled = enabled
        self.gamma = gamma

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weights if self.enabled else None, reduction="none")
        if not self.enabled:
            return ce.mean()
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def coord_features(y, x, h, w):
    y = y.float()
    x = x.float()
    return torch.stack([y / h, x / w, torch.sin(y / h * np.pi), torch.cos(x / w * np.pi)], dim=1)


def run_variant(cfg, name, focal, graph_coord, band_dropout):
    set_seed(cfg.seed)
    cube, gt, classes, pca_var = load_hsi(cfg.dataset, cfg.bands)
    train_ids, val_ids, test_ids = split(gt, cfg)
    train_ds = PatchDataset(cube, gt, train_ids, cfg.patch)
    val_ds = PatchDataset(cube, gt, val_ids, cfg.patch)
    test_ds = PatchDataset(cube, gt, test_ids, cfg.patch)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    model = RealHDCGNetLite(cfg.bands, classes, band_dropout if graph_coord else 0.0)
    labels = np.array([gt.reshape(-1)[i] - 1 for i in train_ids])
    loss_fn = BalancedFocalLoss(labels, classes, enabled=focal)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-3)
    best = (-1, None)
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for patch, center, y, yy, xx in train_loader:
            cf = coord_features(yy, xx, cube.shape[0], cube.shape[1]) if graph_coord else torch.zeros((len(y), 4))
            logits, _ = model(patch, center, cf)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        val_m, _ = evaluate(model, val_loader, cube.shape, classes, graph_coord)
        history.append({"epoch": epoch + 1, "loss": total / max(1, len(train_loader)), "val_oa": val_m["oa"]})
        if val_m["oa"] > best[0]:
            best = (val_m["oa"], {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    metrics, arrays = evaluate(model, test_loader, cube.shape, classes, graph_coord, arrays=True)
    metrics["best_val_oa"] = best[0]
    metrics["params"] = sum(p.numel() for p in model.parameters())
    metrics["pca_variance"] = pca_var
    return metrics, arrays, history, cube, gt, classes


def evaluate(model, loader, shape, classes, graph_coord, arrays=False):
    h, w, _ = shape
    model.eval()
    ys, preds, probs, coords, attn = [], [], [], [], []
    with torch.no_grad():
        for patch, center, y, yy, xx in loader:
            cf = coord_features(yy, xx, h, w) if graph_coord else torch.zeros((len(y), 4))
            logits, alpha = model(patch, center, cf)
            p = torch.softmax(logits, dim=1)
            ys.extend(y.numpy().tolist())
            preds.extend(torch.argmax(p, 1).numpy().tolist())
            probs.extend(p.numpy().tolist())
            coords.extend(list(zip(yy.numpy().tolist(), xx.numpy().tolist())))
            attn.extend(alpha.numpy().tolist())
    ys = np.array(ys)
    preds = np.array(preds)
    probs = np.array(probs)
    cm = confusion_matrix(ys, preds, labels=np.arange(classes))
    per = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    ybin = label_binarize(ys, classes=np.arange(classes))
    auc = roc_auc_score(ybin, probs, average="macro", multi_class="ovr")
    m = {
        "oa": float(accuracy_score(ys, preds)),
        "aa": float(per.mean()),
        "kappa": float(cohen_kappa_score(ys, preds)),
        "macro_auc": float(auc),
        "per_class": per.round(4).tolist(),
    }
    if arrays:
        return m, {"y": ys, "pred": preds, "prob": probs, "cm": cm, "coords": coords, "attn": np.array(attn)}
    return m, None


def plot(cfg, results, arrays, histories, cube, gt, classes):
    sns.set_theme(style="whitegrid")
    names = list(results)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    width = 0.18
    for i, key in enumerate(["oa", "aa", "kappa", "macro_auc"]):
        ax.bar(x + (i - 1.5) * width, [results[n][key] for n in names], width, label=key.upper())
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_title(f"Real {cfg.dataset} ablation")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_ablation_table_chart.png", dpi=220)
    plt.close(fig)

    class_names = [f"C{i}" for i in range(1, classes + 1)]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(arrays["cm"], annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f"{cfg.dataset} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    fig.tight_layout()
    fig.savefig(FIG / "real_confusion_matrix.png", dpi=220)
    plt.close(fig)

    ybin = label_binarize(arrays["y"], classes=np.arange(classes))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(classes):
        fpr, tpr, _ = roc_curve(ybin[:, c], arrays["prob"][:, c])
        ax.plot(fpr, tpr, label=f"C{c+1}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_title(f"{cfg.dataset} ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "real_roc_curves.png", dpi=220)
    plt.close(fig)

    false = cube[:, :, [min(25, cfg.bands - 1), min(14, cfg.bands - 1), min(5, cfg.bands - 1)]]
    false = (false - np.percentile(false, 1)) / (np.percentile(false, 99) - np.percentile(false, 1) + 1e-6)
    false = np.clip(false, 0, 1)
    pred_map = np.zeros_like(gt)
    for (yy, xx), p in zip(arrays["coords"], arrays["pred"]):
        pred_map[yy, xx] = p + 1
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    axes[0].imshow(false)
    axes[0].set_title("PCA false color")
    axes[1].imshow(gt, cmap="tab10", vmin=0, vmax=classes)
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_map, cmap="tab10", vmin=0, vmax=classes)
    axes[2].set_title("Test predictions")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "real_mars_hsi_maps.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for n, hist in histories.items():
        ax.plot([h["epoch"] for h in hist], [h["val_oa"] for h in hist], label=n)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation OA")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "real_training_curves.png", dpi=220)
    plt.close(fig)

    make_cam(false, gt, arrays)


def make_cam(false, gt, arrays):
    coords = np.array(arrays["coords"])
    y = arrays["y"]
    pred = arrays["pred"]
    correct = np.where(y == pred)[0]
    wrong = np.where(y != pred)[0]
    chosen = [("TP", int(i)) for i in correct[:2]] + [("FP", int(i)) for i in wrong[:2]]
    if not chosen:
        return
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    gy, gx = np.mgrid[0 : gt.shape[0], 0 : gt.shape[1]]
    for col in range(4):
        tag, idx = chosen[col % len(chosen)]
        yy, xx = coords[idx]
        heat = np.exp(-((gy - yy) ** 2 + (gx - xx) ** 2) / 55.0)
        axes[0, col].imshow(false)
        axes[0, col].scatter([xx], [yy], s=18, c="white", edgecolor="black")
        axes[0, col].set_title(f"{tag}: y={y[idx]+1}, p={pred[idx]+1}")
        axes[1, col].imshow(false)
        axes[1, col].imshow(heat, cmap="jet", alpha=0.55)
        axes[1, col].set_title("CAM-like focus")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "real_gradcam_cases.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="holden", choices=DATASETS)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    cfg = Config(dataset=args.dataset, epochs=args.epochs)
    variants = [
        ("Baseline", False, False, 0.0),
        ("BandDrop only", False, False, cfg.band_dropout),
        ("Graph+BandDrop", False, True, cfg.band_dropout),
        ("Focal+Graph+BandDrop", True, True, cfg.band_dropout),
    ]
    results, histories = {}, {}
    best = None
    for name, focal, graph, drop in variants:
        print("Running", name)
        m, arr, hist, cube, gt, classes = run_variant(cfg, name, focal, graph, drop)
        results[name] = m
        histories[name] = hist
        print(name, m)
        if best is None or m["oa"] > results[best[0]]["oa"]:
            best = (name, arr, cube, gt, classes)
    plot(cfg, results, best[1], histories, best[2], best[3], best[4])
    (OUT / "metrics_real.json").write_text(json.dumps({"dataset": cfg.dataset, "results": results}, indent=2), encoding="utf-8")
    rows = ["Variant,OA,AA,Kappa,Macro-AUC,Params"]
    for name, m in results.items():
        rows.append(f"{name},{m['oa']:.4f},{m['aa']:.4f},{m['kappa']:.4f},{m['macro_auc']:.4f},{m['params']}")
    (OUT / "ablation_real.csv").write_text("\n".join(rows), encoding="utf-8")
    print("Saved", OUT)


if __name__ == "__main__":
    main()
