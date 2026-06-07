# HDCGNet Final Presentation Reproduction

This repository contains a final presentation project for reproducing and improving:

**HDCGNet: A Hypergraph Dual-Branch CNN-GCN Network for Mars Hyperspectral Image Classification**

The project focuses on a CPU-feasible partial reproduction of the paper's core idea:

- CNN branch for local spectral-spatial feature extraction
- graph / GCN-style branch for spatial-spectral context
- adaptive fusion of CNN and graph features
- two improvements for the reproduced model

## Student

**ZHAO HONG WEI**  
M11417025@yuntech.edu.tw  
Department of Computer Science and Information Engineering  
National Yunlin University of Science and Technology

## Paper

Tian et al.,  
"HDCGNet: A Hypergraph Dual-Branch CNN-GCN Network for Mars Hyperspectral Image Classification,"  
IEEE Transactions on Geoscience and Remote Sensing, 2026.

Official repository inspected during this project:

https://github.com/Ctao0820/HDCGNet

Dataset DOI:

https://doi.org/10.57760/sciencedb.19732

## Dataset

The final experiment uses the original **Holden Crater** Mars hyperspectral dataset from the paper data release.

Expected files:

```text
official_HDCGNet/datasets/
├── holden.mat
├── holden_gt.mat
├── NiliFossae.mat
├── NiliFossae_gt.mat
├── Utopia.mat
├── Utopia_gt.mat
└── data_description.xlsx
```

Only Holden Crater is used for the final reported experiment:

```text
holden.mat     shape: 418 x 595 x 440
holden_gt.mat  shape: 418 x 595
classes: 6
```

Large `.mat` files should not be committed to GitHub. Download them from the DOI above and place them in `official_HDCGNet/datasets/`.

## Method Summary

The inspected official repository did not include the complete final `model.py`, so this project implements a lightweight HDCGNet-inspired reproduction.

The reproduced model includes:

1. **CNN branch**  
   Extracts local spectral-spatial features from HSI patches.

2. **Graph-context branch**  
   Uses spectral features and coordinate context to approximate the graph/hypergraph branch.

3. **Adaptive fusion**  
   Learns attention weights between CNN and graph features.

4. **Improvement 1: Graph context with band dropout**  
   Adds graph-style spatial-spectral context and regularizes spectral bands.

5. **Improvement 2: Class-balanced focal loss**  
   Improves the full graph-regularized model under few-shot and imbalanced settings.

## Results

Experiment setting:

- Dataset: Holden Crater
- Training samples: 10 pixels per class
- Validation samples: 10 pixels per class
- Spectral preprocessing: PCA to 32 components
- Training: 60 epochs
- Runtime environment: Anaconda `final1`, CPU

| Variant | OA | AA | Kappa | Macro-AUC | Params |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.8548 | 0.8548 | 0.8257 | 0.9764 | 45,272 |
| BandDrop only | 0.8548 | 0.8548 | 0.8257 | 0.9764 | 45,272 |
| Graph+BandDrop | 0.8571 | 0.8571 | 0.8286 | 0.9784 | 45,272 |
| Focal+Graph+BandDrop | **0.8724** | **0.8724** | **0.8469** | **0.9792** | 45,272 |

The full improved model raises OA from **0.8548** to **0.8724** on Holden Crater.

## Repository Structure

```text
.
├── HDCGNet_A_Hypergraph_Dual-Branch_CNNGCN_Network_for_Mars_Hyperspectral_Image_Classification.pdf
├── hdcgnet_repro/
│   ├── run_real_dataset_experiment.py
│   ├── run_experiment.py
│   ├── promote_real_outputs.py
│   ├── make_deliverables.py
│   ├── make_template_report.py
│   ├── outputs_real/
│   │   ├── ablation_real.csv
│   │   ├── metrics_real.json
│   │   └── figures/
│   └── outputs/
├── official_HDCGNet/
│   └── datasets/
└── deliverables/
    ├── HDCGNet_Final_Presentation_updated.pptx
    ├── HDCGNet_Final_Report.pdf
    ├── HDCGNet_Final_Report.tex
    └── tex/
```

## Environment

The project was run with the user's Anaconda environment:

```text
C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe
```

Required Python packages:

```text
torch
numpy
scipy
scikit-learn
matplotlib
seaborn
pillow
python-pptx
reportlab
pypdf
pymupdf
```

## How to Run

Run the real Holden Crater experiment:

```powershell
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\run_real_dataset_experiment.py --dataset holden --epochs 60
```

Promote the real Holden outputs into the deliverable pipeline:

```powershell
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\promote_real_outputs.py
```

Generate the presentation:

```powershell
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\make_deliverables.py
```

Generate the YunTech LaTeX-style report PDF/TEX:

```powershell
& 'C:\Users\USER\AppData\Local\anaconda3\envs\final1\python.exe' hdcgnet_repro\make_template_report.py
```

## Deliverables

Final files:

```text
deliverables/
├── HDCGNet_Final_Presentation_updated.pptx
├── HDCGNet_Final_Report.pdf
├── HDCGNet_Final_Report.tex
└── tex/
    └── HDCGNet_tex_<timestamp>/
        ├── HDCGNet_Final_Report.tex
        ├── figures/
        └── HDCGNet_tex_<timestamp>.zip
```

## Limitations

- The full official HDCGNet model implementation was not available in the inspected GitHub repository.
- This project reproduces the core CNN-GCN fusion idea with a lightweight implementation.
- The final experiment uses the original Holden Crater dataset only; Nili Fossae and Utopia Planitia are left for future work.
- Reported numbers are not directly comparable to the full paper because the architecture and hardware setting are simplified.

## License / Usage

This repository is for coursework and academic reproduction only. Please cite the original HDCGNet paper and download the dataset from the official DOI.
