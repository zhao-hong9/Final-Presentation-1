# HDCGNet partial reproduction

This folder contains a runnable, lightweight reproduction inspired by:

HDCGNet: A Hypergraph Dual-Branch CNN-GCN Network for Mars Hyperspectral Image Classification.

The inspected official repository did not include the final `model.py` implementation. After the original data files were downloaded, this project uses the Holden Crater Mars HSI dataset for a CPU-feasible partial reproduction of the core idea: CNN local branch + graph/GCN branch + adaptive fusion.

## Environment

All commands were run with the Anaconda environment requested by the user:

```powershell
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\run_real_dataset_experiment.py --dataset holden --epochs 60
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\promote_real_outputs.py
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\make_deliverables.py
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\make_template_report.py
```

## Improvements

1. Graph context with band dropout for spectral robustness.
2. Class-balanced focal loss in the full graph-regularized model.

## Main result

Holden Crater, 10 train samples/class, 10 validation samples/class, 60 epochs.

| Variant | OA | AA | Kappa | Macro-AUC |
|---|---:|---:|---:|---:|
| Baseline | 0.8548 | 0.8548 | 0.8257 | 0.9764 |
| BandDrop only | 0.8548 | 0.8548 | 0.8257 | 0.9764 |
| Graph+BandDrop | 0.8571 | 0.8571 | 0.8286 | 0.9784 |
| Focal+Graph+BandDrop | 0.8724 | 0.8724 | 0.8469 | 0.9792 |

## Deliverables

Final files are in `deliverables/`:

- `HDCGNet_Final_Presentation.pptx`
- `HDCGNet_Final_Report.pdf`
- `HDCGNet_Final_Report.tex`
