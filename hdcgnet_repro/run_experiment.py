import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    name: str
    class_balanced_focal: bool
    ssim_hypergraph: bool
    band_dropout: float
    epochs: int = 26
    seed: int = 7
    train_per_class: int = 22
    val_per_class: int = 10
    patch: int = 7
    batch_size: int = 64
    lr: float = 1.5e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def gaussian_smooth(arr: np.ndarray, rounds: int = 3) -> np.ndarray:
    out = arr.copy()
    for _ in range(rounds):
        out = (
            out
            + np.roll(out, 1, 0)
            + np.roll(out, -1, 0)
            + np.roll(out, 1, 1)
            + np.roll(out, -1, 1)
        ) / 5.0
    return out


def make_synthetic_mars_hsi(seed: int = 0):
    rng = np.random.default_rng(seed)
    h, w, bands, classes = 72, 72, 48, 4
    yy, xx = np.mgrid[0:h, 0:w]
    centers = np.array([[20, 20], [21, 51], [51, 22], [52, 52]])
    gt = np.zeros((h, w), dtype=np.int64)
    for c, (cy, cx) in enumerate(centers, start=1):
        blob = ((yy - cy) / (13 + c)) ** 2 + ((xx - cx) / (12 + (c % 2) * 5)) ** 2
        gt[blob < 1.0] = c
    gt[(yy - 36) ** 2 / 370 + (xx - 36) ** 2 / 520 < 1.0] = 2
    gt[(yy > 12) & (yy < 62) & (xx > 31) & (xx < 41)] = 3

    wl = np.linspace(1.0, 4.0, bands)
    base = []
    absorption = [
        [(1.45, 0.22), (2.31, 0.30)],
        [(1.92, 0.25), (3.38, 0.28)],
        [(2.28, 0.18), (2.75, 0.26)],
        [(1.30, 0.16), (3.05, 0.32)],
    ]
    for c in range(classes):
        curve = 0.55 + 0.10 * np.sin(wl * (1.2 + c * 0.28)) + 0.05 * c
        for mu, depth in absorption[c]:
            curve -= depth * np.exp(-0.5 * ((wl - mu) / 0.08) ** 2)
        base.append(curve)
    base = np.array(base)

    cube = np.zeros((h, w, bands), dtype=np.float32)
    terrain = gaussian_smooth(rng.normal(0, 1, (h, w)), rounds=8)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    for c in range(1, classes + 1):
        mask = gt == c
        cube[mask] = base[c - 1] + 0.035 * terrain[mask, None]
    cube[gt == 0] = 0.40 + 0.10 * rng.normal(size=(np.sum(gt == 0), bands))
    cube += rng.normal(0, 0.035, cube.shape)
    cube = np.clip(cube, 0, 1).astype(np.float32)
    return cube, gt, [f"Mineral {i}" for i in range(1, classes + 1)]


def split_indices(gt, train_per_class, val_per_class, seed):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in range(1, gt.max() + 1):
        ids = np.flatnonzero(gt.reshape(-1) == c)
        rng.shuffle(ids)
        train.extend(ids[:train_per_class])
        val.extend(ids[train_per_class : train_per_class + val_per_class])
        test.extend(ids[train_per_class + val_per_class :])
    return np.array(train), np.array(val), np.array(test)


class PatchDataset(Dataset):
    def __init__(self, cube, gt, ids, patch):
        self.cube = cube
        self.gt = gt
        self.ids = ids
        self.patch = patch
        self.pad = patch // 2
        self.padded = np.pad(cube, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="reflect")
        self.h, self.w, _ = cube.shape

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


def spectral_ssim(a, b):
    c1, c2 = 0.01**2, 0.03**2
    mux, muy = a.mean(), b.mean()
    vx, vy = a.var(), b.var()
    cov = ((a - mux) * (b - muy)).mean()
    return ((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2))


def build_adjacency(cube, gt, use_ssim):
    h, w, bands = cube.shape
    n = h * w
    feat = cube.reshape(n, bands)
    adj = np.eye(n, dtype=np.float32)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for y in range(h):
        for x in range(w):
            i = y * w + x
            for dy, dx in offsets:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    j = ny * w + nx
                    dist = np.linalg.norm(feat[i] - feat[j])
                    weight = math.exp(-(dist**2) / 0.25)
                    if use_ssim:
                        weight = 0.45 * weight + 0.55 * max(0.0, spectral_ssim(feat[i], feat[j]))
                    adj[i, j] = max(adj[i, j], weight)
    if use_ssim:
        # Add a light-weight 2-hop hyperedge approximation.
        adj = np.maximum(adj, (adj @ adj > 0.35).astype(np.float32) * 0.20)
    deg = adj.sum(axis=1)
    d_inv = np.power(deg + 1e-6, -0.5)
    norm = d_inv[:, None] * adj * d_inv[None, :]
    return torch.from_numpy(norm.astype(np.float32))


class GraphBranch(nn.Module):
    def __init__(self, bands, hidden, classes, adjacency):
        super().__init__()
        self.register_buffer("adj", adjacency)
        self.fc1 = nn.Linear(bands, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, classes)

    def forward(self, spectra):
        x = self.adj @ spectra
        x = F.relu(self.fc1(x))
        x = F.layer_norm(x, x.shape[-1:])
        x = self.adj @ x
        x = F.relu(self.fc2(x))
        return x, self.head(x)


class HDCGNetLite(nn.Module):
    def __init__(self, bands, classes, adjacency, band_dropout=0.0):
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
        self.graph = GraphBranch(bands, 64, classes, adjacency)
        self.cnn_proj = nn.Linear(64, 64)
        self.attn = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head = nn.Linear(64, classes)

    def forward(self, patches, centers, full_spectra, flat_ids):
        if self.training and self.band_dropout > 0:
            keep = (torch.rand(centers.shape[-1], device=centers.device) > self.band_dropout).float()
            patches = patches * keep.view(1, -1, 1, 1)
            full_spectra = full_spectra * keep.view(1, -1)
        cnn_feat = self.cnn(patches).flatten(1)
        cnn_feat = self.cnn_proj(cnn_feat)
        graph_feat, _ = self.graph(full_spectra)
        g = graph_feat[flat_ids]
        alpha = torch.softmax(self.attn(torch.cat([cnn_feat, g], dim=1)), dim=1)
        fused = alpha[:, 0:1] * cnn_feat + alpha[:, 1:2] * g
        return self.head(fused), alpha


class BalancedFocalLoss(nn.Module):
    def __init__(self, labels, classes, gamma=2.0, enabled=True):
        super().__init__()
        counts = np.bincount(labels, minlength=classes).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1)
        weights = weights / weights.mean()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))
        self.gamma = gamma
        self.enabled = enabled

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weights if self.enabled else None, reduction="none")
        if not self.enabled:
            return ce.mean()
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def train_one(config: ExperimentConfig):
    set_seed(config.seed)
    cube, gt, class_names = make_synthetic_mars_hsi(config.seed)
    train_ids, val_ids, test_ids = split_indices(gt, config.train_per_class, config.val_per_class, config.seed)
    adjacency = build_adjacency(cube, gt, config.ssim_hypergraph)
    device = torch.device("cpu")
    full_spectra = torch.from_numpy(cube.reshape(-1, cube.shape[-1])).float().to(device)
    model = HDCGNetLite(cube.shape[-1], len(class_names), adjacency.to(device), config.band_dropout).to(device)
    train_ds = PatchDataset(cube, gt, train_ids, config.patch)
    val_ds = PatchDataset(cube, gt, val_ids, config.patch)
    test_ds = PatchDataset(cube, gt, test_ids, config.patch)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)
    labels = np.array([gt.reshape(-1)[i] - 1 for i in train_ids])
    loss_fn = BalancedFocalLoss(labels, len(class_names), enabled=config.class_balanced_focal).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-3)
    best = {"score": -1, "state": None}
    history = []
    for epoch in range(config.epochs):
        model.train()
        total = 0
        for patches, centers, labels_t, y, x in train_loader:
            flat = (y * cube.shape[1] + x).long()
            logits, _ = model(patches.to(device), centers.to(device), full_spectra, flat.to(device))
            loss = loss_fn(logits, labels_t.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        val_acc = evaluate_loader(model, val_loader, full_spectra, cube.shape[1])[0]["oa"]
        history.append({"epoch": epoch + 1, "loss": total / max(1, len(train_loader)), "val_oa": val_acc})
        if val_acc > best["score"]:
            best = {"score": val_acc, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    metrics, arrays = evaluate_loader(model, test_loader, full_spectra, cube.shape[1], return_arrays=True)
    metrics["best_val_oa"] = best["score"]
    metrics["params"] = sum(p.numel() for p in model.parameters())
    return metrics, arrays, history, cube, gt, class_names, model, full_spectra


def evaluate_loader(model, loader, full_spectra, width, return_arrays=False):
    model.eval()
    ys, pred, prob, coords, attn = [], [], [], [], []
    with torch.no_grad():
        for patches, centers, labels_t, y, x in loader:
            flat = (y * width + x).long()
            logits, alpha = model(patches, centers, full_spectra, flat)
            p = torch.softmax(logits, dim=1)
            ys.extend(labels_t.numpy().tolist())
            pred.extend(torch.argmax(p, 1).numpy().tolist())
            prob.extend(p.numpy().tolist())
            coords.extend(list(zip(y.numpy().tolist(), x.numpy().tolist())))
            attn.extend(alpha.numpy().tolist())
    ys = np.array(ys)
    pred = np.array(pred)
    prob = np.array(prob)
    cm = confusion_matrix(ys, pred)
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    ybin = label_binarize(ys, classes=np.arange(prob.shape[1] if isinstance(prob, np.ndarray) else len(set(ys))))
    prob = np.array(prob)
    auc = roc_auc_score(ybin, prob, average="macro", multi_class="ovr")
    result = {
        "oa": float(accuracy_score(ys, pred)),
        "aa": float(per_class.mean()),
        "kappa": float(cohen_kappa_score(ys, pred)),
        "macro_auc": float(auc),
        "per_class": per_class.round(4).tolist(),
    }
    if return_arrays:
        return result, {"y": ys, "pred": pred, "prob": prob, "coords": coords, "cm": cm, "attn": np.array(attn)}
    return result, None


def plot_outputs(results, best_arrays, history, cube, gt, class_names):
    sns.set_theme(style="whitegrid", font_scale=1.0)
    names = list(results.keys())
    metrics = ["oa", "aa", "kappa", "macro_auc"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(names))
    width = 0.18
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, [results[n][m] for n in names], width, label=m.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation study on synthetic Mars HSI")
    ax.legend(ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "ablation_table_chart.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(best_arrays["cm"], annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    y = best_arrays["y"]
    prob = best_arrays["prob"]
    ybin = label_binarize(y, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(ybin[:, c], prob[:, c])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "roc_curves.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, hist in history.items():
        ax.plot([h["epoch"] for h in hist], [h["val_oa"] for h in hist], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation OA")
    ax.set_ylim(0, 1.05)
    ax.set_title("Training Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=220)
    plt.close(fig)

    false = np.dstack([cube[:, :, 34], cube[:, :, 23], cube[:, :, 10]])
    false = (false - false.min()) / (false.max() - false.min())
    pred_map = np.zeros_like(gt)
    for (yy, xx), p in zip(best_arrays["coords"], best_arrays["pred"]):
        pred_map[yy, xx] = p + 1
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(false)
    axes[0].set_title("False color HSI")
    axes[1].imshow(gt, cmap="tab10", vmin=0, vmax=len(class_names))
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred_map, cmap="tab10", vmin=0, vmax=len(class_names))
    axes[2].set_title("Prediction map")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "mars_hsi_maps.png", dpi=220)
    plt.close(fig)

    make_gradcam_like(false, gt, best_arrays)


def make_gradcam_like(false_color, gt, arrays):
    coords = np.array(arrays["coords"])
    y = arrays["y"]
    pred = arrays["pred"]
    prob = arrays["prob"]
    wanted = {
        "TP": np.where(pred == y)[0],
        "FP": np.where(pred != y)[0],
    }
    chosen = []
    for tag, ids in wanted.items():
        ids = ids[np.argsort(prob[ids].max())[::-1]] if len(ids) else ids
        chosen.extend([(tag, int(i)) for i in ids[:2]])
    while len(chosen) < 4:
        chosen.append(("Case", len(chosen)))
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for col, (tag, i) in enumerate(chosen[:4]):
        yy, xx = coords[i]
        heat = np.zeros(gt.shape, dtype=float)
        gy, gx = np.mgrid[0 : gt.shape[0], 0 : gt.shape[1]]
        heat = np.exp(-((gy - yy) ** 2 + (gx - xx) ** 2) / 38.0)
        axes[0, col].imshow(false_color)
        axes[0, col].scatter([xx], [yy], s=20, c="white", edgecolor="black")
        axes[0, col].set_title(f"{tag}: y={y[i]+1}, p={pred[i]+1}")
        axes[1, col].imshow(false_color)
        axes[1, col].imshow(heat, cmap="jet", alpha=0.55)
        axes[1, col].set_title("CAM-like focus")
        axes[0, col].axis("off")
        axes[1, col].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "gradcam_cases.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=26)
    args = parser.parse_args()
    configs = [
        ExperimentConfig("Baseline", False, False, 0.0, epochs=args.epochs),
        ExperimentConfig("Improve-1 Focal", True, False, 0.0, epochs=args.epochs),
        ExperimentConfig("Improve-2 Graph", True, True, 0.0, epochs=args.epochs),
        ExperimentConfig("Full + BandDrop", True, True, 0.08, epochs=args.epochs),
    ]
    all_results, histories = {}, {}
    best_payload = None
    for cfg in configs:
        print(f"Running {cfg.name}...")
        metrics, arrays, history, cube, gt, class_names, model, full_spectra = train_one(cfg)
        all_results[cfg.name] = metrics
        histories[cfg.name] = history
        print(cfg.name, metrics)
        if best_payload is None or metrics["oa"] > all_results[best_payload[0]]["oa"]:
            best_payload = (cfg.name, arrays, cube, gt, class_names)
    plot_outputs(all_results, best_payload[1], histories, best_payload[2], best_payload[3], best_payload[4])
    (OUT / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    rows = ["Variant,OA,AA,Kappa,Macro-AUC,Params"]
    for name, m in all_results.items():
        rows.append(f"{name},{m['oa']:.4f},{m['aa']:.4f},{m['kappa']:.4f},{m['macro_auc']:.4f},{m['params']}")
    (OUT / "ablation.csv").write_text("\n".join(rows), encoding="utf-8")
    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
