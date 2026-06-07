import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
real = ROOT / "outputs_real"
out = ROOT / "outputs"
fig_real = real / "figures"
fig = out / "figures"
fig.mkdir(parents=True, exist_ok=True)

payload = json.loads((real / "metrics_real.json").read_text(encoding="utf-8"))
(out / "metrics.json").write_text(json.dumps(payload["results"], indent=2), encoding="utf-8")

rows = ["Variant,OA,AA,Kappa,Macro-AUC,Params"]
for name, m in payload["results"].items():
    rows.append(f"{name},{m['oa']:.4f},{m['aa']:.4f},{m['kappa']:.4f},{m['macro_auc']:.4f},{m['params']}")
(out / "ablation.csv").write_text("\n".join(rows), encoding="utf-8")

mapping = {
    "real_ablation_table_chart.png": "ablation_table_chart.png",
    "real_confusion_matrix.png": "confusion_matrix.png",
    "real_roc_curves.png": "roc_curves.png",
    "real_gradcam_cases.png": "gradcam_cases.png",
    "real_mars_hsi_maps.png": "mars_hsi_maps.png",
    "real_training_curves.png": "training_curves.png",
}
for src, dst in mapping.items():
    shutil.copy2(fig_real / src, fig / dst)

print("Promoted real Holden outputs into", out)
