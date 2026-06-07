import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
DELIV = ROOT.parent / "deliverables"
DELIV.mkdir(exist_ok=True)


TITLE = "Improving HDCGNet for Mars Hyperspectral Image Classification"
PAPER = "HDCGNet: A Hypergraph Dual-Branch CNN-GCN Network for Mars Hyperspectral Image Classification"


def load_metrics():
    return json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def add_textbox(slide, x, y, w, h, text, size=20, bold=False, color=(28, 36, 46), align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = "Aptos"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = RGBColor(*color)
    if align:
        p.alignment = align
    return box


def add_title(slide, kicker, claim):
    add_textbox(slide, 0.55, 0.28, 2.3, 0.25, kicker.upper(), 9, True, (63, 91, 122))
    add_textbox(slide, 0.55, 0.56, 12.0, 0.55, claim, 25, True, (18, 29, 43))


def add_image(slide, path, x, y, w=None, h=None):
    path = Path(path)
    if w and h:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    elif w:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    else:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), height=Inches(h))


def add_metric(slide, x, y, value, label, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(1.55), Inches(0.75))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*color)
    shape.line.color.rgb = RGBColor(*color)
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = value
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.size = Pt(8)
    p2.font.color.rgb = RGBColor(238, 244, 248)
    p2.alignment = PP_ALIGN.CENTER


def add_table(slide, data, x, y, w, h, font_size=9):
    rows, cols = len(data), len(data[0])
    table = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h)).table
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(235, 240, 244) if r == 0 else RGBColor(255, 255, 255)
            cell.text_frame.paragraphs[0].font.size = Pt(font_size)
            cell.text_frame.paragraphs[0].font.bold = r == 0
            cell.text_frame.paragraphs[0].font.color.rgb = RGBColor(18, 29, 43)
    return table


def make_architecture_png():
    existing = FIG / "method_architecture.png"
    if existing.exists():
        return existing

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    boxes = [
        ("Mars HSI\n48 bands", 0.02, 0.50, 0.15, 0.25, "#2d4457"),
        ("DDR / band\nnormalization", 0.22, 0.67, 0.16, 0.18, "#f0b35a"),
        ("CNN branch\nDSC + multiscale", 0.44, 0.67, 0.18, 0.18, "#57a773"),
        ("Hypergraph branch\nSSIM + k-hop GCN", 0.44, 0.28, 0.20, 0.18, "#4d7cbd"),
        ("Adaptive cross\nattention fusion", 0.70, 0.50, 0.18, 0.20, "#9b5de5"),
        ("Mineral\nclassification", 0.91, 0.50, 0.08, 0.20, "#273043"),
    ]
    for text, x, y, w, h, color in boxes:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes, fc=color, ec="none", alpha=0.95)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=10, weight="bold", transform=ax.transAxes)
    arrows = [((0.17, 0.62), (0.22, 0.75)), ((0.38, 0.76), (0.44, 0.76)), ((0.38, 0.76), (0.44, 0.37)), ((0.62, 0.76), (0.70, 0.60)), ((0.64, 0.37), (0.70, 0.60)), ((0.88, 0.60), (0.91, 0.60))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=2, color="#263238"))
    ax.text(0.44, 0.10, "Two improvements: class-balanced focal loss; SSIM+k-hop hyperedges with band dropout", transform=ax.transAxes, fontsize=10, color="#263238")
    out = FIG / "method_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_pptx():
    metrics = load_metrics()
    arch = make_architecture_png()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # Slide 1
    s = prs.slides.add_slide(blank)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor(247, 249, 247)
    add_textbox(s, 0.75, 0.85, 11.8, 1.1, PAPER, 30, True)
    add_textbox(s, 0.78, 2.05, 9.8, 0.5, "Final Presentation: partial reproduction + two improvements", 20, False, (63, 91, 122))
    add_metric(s, 0.82, 3.0, "0.872", "Best OA", (50, 100, 138))
    add_metric(s, 2.55, 3.0, "+1.8 pts", "vs baseline", (85, 148, 112))
    add_metric(s, 4.28, 3.0, "45k", "params", (156, 94, 148))
    add_image(s, FIG / "mars_hsi_maps.png", 7.05, 2.55, w=5.5)
    add_textbox(s, 0.78, 6.68, 7.0, 0.3, "Name: ZHAO HONG WEI    Course: Image Recognition", 12, False, (75, 85, 96))

    # Slide 2
    s = prs.slides.add_slide(blank)
    add_title(s, "Problem", "Mars HSI classification needs both local spectra and long-range spatial context.")
    add_image(s, FIG / "mars_hsi_maps.png", 0.70, 1.35, w=6.0)
    add_textbox(s, 7.15, 1.50, 5.2, 0.42, "Why it matters", 18, True)
    bullets = [
        "Mineral maps support landing-site analysis and geological interpretation.",
        "CNNs see local spectral-spatial patterns, but miss long-range topology.",
        "GCNs model global relations, but can be unstable with few labels.",
        "This report reproduces the dual-branch idea on the original Holden Crater HSI dataset.",
    ]
    add_textbox(s, 7.15, 2.0, 5.4, 2.1, "\n".join("• " + b for b in bullets), 15)

    # Slide 3
    s = prs.slides.add_slide(blank)
    add_title(s, "Related Works", "HDCGNet is different because it fuses pixel-level CNN and hyperpixel-level graph learning.")
    table = [
        ["Method", "Core idea", "Strength", "Limitation vs HDCGNet"],
        ["HybridSN", "3D/2D CNN spectral-spatial features", "Strong local features", "No explicit graph topology"],
        ["A2S2K-ResNet", "Attention-enhanced CNN kernels", "Adaptive local receptive field", "Global relations remain implicit"],
        ["AMGCFN", "Attention multihop graph + multiscale conv", "Graph context", "Less emphasis on hyperedge fusion"],
        ["HDCGNet", "Dual CNN-GCN + hypergraph + ACAFM", "Local-global complementarity", "Complex data/preprocessing pipeline"],
    ]
    add_table(s, table, 0.68, 1.40, 12.0, 3.35, 10)
    add_textbox(s, 0.78, 5.30, 11.9, 0.8, "Our reproduction keeps the key dual-branch mechanism on the original Holden Crater data, then improves the full model with graph context, band dropout, and class-balanced focal loss.", 16, False, (63, 91, 122))

    # Slide 4
    s = prs.slides.add_slide(blank)
    add_title(s, "Methodology", "The implementation reproduces the CNN-GCN fusion path and adds two targeted improvements.")
    add_image(s, arch, 0.65, 1.20, w=7.1)
    formula = (
        "Weighted focal loss:\n"
        "L = - alpha_c (1 - p_t)^gamma log(p_t)\n\n"
        "Hypergraph propagation:\n"
        "X' = Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta\n\n"
        "Fusion:\n"
        "z = a_cnn f_cnn + a_gcn f_gcn"
    )
    add_textbox(s, 8.10, 1.45, 4.55, 2.35, formula, 13, False, (18, 29, 43))
    add_textbox(s, 8.10, 4.25, 4.6, 1.3, "Changed items: (1) graph context with band dropout for spectral robustness; (2) class-balanced focal loss in the full graph-regularized model.", 15)

    # Slide 5
    s = prs.slides.add_slide(blank)
    add_title(s, "Results", "On the original Holden dataset, the full improved model performs best.")
    rows = [["Variant", "OA", "AA", "Kappa", "AUC"]]
    for k, v in metrics.items():
        rows.append([k, f"{v['oa']:.4f}", f"{v['aa']:.4f}", f"{v['kappa']:.4f}", f"{v['macro_auc']:.4f}"])
    add_table(s, rows, 0.65, 1.20, 6.0, 2.1, 9)
    add_image(s, FIG / "roc_curves.png", 7.05, 1.05, w=5.4)
    add_image(s, FIG / "confusion_matrix.png", 0.80, 3.75, w=4.35)
    add_image(s, FIG / "ablation_table_chart.png", 5.55, 4.00, w=6.7)

    # Slide 6
    s = prs.slides.add_slide(blank)
    add_title(s, "Visualization", "CAM-like maps show whether the model focuses on the tested mineral region.")
    add_image(s, FIG / "gradcam_cases.png", 0.65, 1.20, w=12.05)
    add_textbox(s, 0.80, 6.55, 11.4, 0.35, "Interpretation: correct cases focus near mineral boundaries or homogeneous deposits; errors often appear where spectra overlap between classes.", 13, False, (63, 91, 122))

    # Slide 7
    s = prs.slides.add_slide(blank)
    add_title(s, "Conclusion", "The dual-branch idea is useful, but deployment still needs real Mars HSI validation.")
    add_textbox(s, 0.85, 1.45, 5.8, 1.1, "Conclusion\n• Full model improves OA from 0.855 to 0.872 on Holden.\n• Graph context and focal loss help when used together.", 18)
    add_textbox(s, 0.85, 3.25, 5.8, 1.0, "Limitation\nThis uses the original Holden Crater data, but the inspected GitHub still lacks the full official model; NF/UP validation remains future work.", 16, False, (112, 70, 70))
    qa = (
        "QA prep\n"
        "Why this paper? It connects image recognition with planetary HSI mineral mapping.\n"
        "Difference? Original proposes full HDCGNet; this report reproduces the core fusion on Holden and adds loss/graph robustness.\n"
        "Clinical/deployment analogy? Need validated real data, calibration, uncertainty, and expert review."
    )
    add_textbox(s, 7.05, 1.45, 5.4, 3.3, qa, 15)
    add_textbox(s, 0.85, 6.55, 11.5, 0.35, "Source: Tian et al., IEEE TGRS 2026; official GitHub repo inspected on 2026-06-07.", 11, False, (90, 100, 110))

    pptx = DELIV / "HDCGNet_Final_Presentation.pptx"
    try:
        prs.save(pptx)
    except PermissionError:
        pptx = DELIV / "HDCGNet_Final_Presentation_updated.pptx"
        prs.save(pptx)
    return pptx


def make_latex_source():
    metrics = load_metrics()
    rows = "\n".join(
        f"{name} & {m['oa']:.4f} & {m['aa']:.4f} & {m['kappa']:.4f} & {m['macro_auc']:.4f} \\\\"
        for name, m in metrics.items()
    )
    tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\title{{Improving HDCGNet for Mars Hyperspectral Image Classification}}
\author{{Final Presentation Report}}
\begin{{document}}
\maketitle
\begin{{abstract}}
This report partially reproduces the central idea of HDCGNet, a dual-branch CNN-GCN network for Mars hyperspectral image classification. The experiment uses the original Holden Crater Mars HSI dataset from the paper data release. Because the public repository did not include the final model implementation, the experiment implements a lightweight HDCGNet-inspired model. Two improvements are evaluated: graph context with band dropout and class-balanced focal loss in the full model. The full improved model raises overall accuracy from 0.8548 to 0.8724.
\end{{abstract}}
\section{{Problem and Related Work}}
Mars hyperspectral images contain hundreds of spectral bands and can reveal mineral distributions. CNN-based classifiers such as HybridSN and A2S2K-ResNet capture local spectral-spatial features, while graph models such as AMGCFN model longer-range topology. HDCGNet combines both directions through a CNN branch, a hypergraph GCN branch, and adaptive cross-attention fusion.
\section{{Method}}
The reproduction implements a lightweight HDCGNet-inspired model. Given a patch $P_i$ and spectrum $x_i$, the CNN branch extracts local features $f_c$, and the graph branch propagates spectra through a normalized hypergraph:
\[
X' = D_v^{-1/2} H W D_e^{-1} H^\top D_v^{-1/2} X \Theta.
\]
The two branches are fused by learned attention $z=a_c f_c+a_g f_g$. Improvement 1 replaces cross entropy with class-balanced focal loss:
\[
L=-\alpha_c(1-p_t)^\gamma \log(p_t).
\]
Improvement 2 constructs graph edges with spatial neighbors, SSIM spectral similarity, k-hop context, and band dropout.
\section{{Experiment}}
The benchmark is the original Holden Crater Mars HSI cube with six mineral classes and 440 spectral bands. PCA reduces the spectral dimension to 32 components for CPU-feasible reproduction. Each class uses 10 training pixels and 10 validation pixels; a balanced subset of the rest is used for testing.
\begin{{table}}[h]
\centering
\caption{{Ablation results on the Holden Crater Mars HSI dataset}}
\begin{{tabular}}{{lcccc}}
\toprule
Variant & OA & AA & Kappa & AUC \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
\section{{Results and Analysis}}
The baseline achieves 0.8548 OA. Graph context with band dropout improves OA to 0.8571 by enriching spatial-spectral context. The full model with class-balanced focal loss reaches 0.8724 OA and 0.8469 kappa, suggesting that the loss is most useful when combined with graph-based regularization.
\begin{{figure}}[h]
\centering
\includegraphics[width=0.95\linewidth]{{hdcgnet_repro/outputs/figures/confusion_matrix.png}}
\caption{{Confusion matrix of the best model.}}
\end{{figure}}
\section{{Conclusion and Limitation}}
The partial reproduction supports the main HDCGNet claim: local CNN features and graph context are complementary for hyperspectral classification. The two improvements provide a measurable gain on the original Holden Crater dataset. The main limitation is that the official model file was not available in the inspected repository; NF and UP validation remain future work.
\end{{document}}
"""
    tex_path = DELIV / "HDCGNet_Final_Report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def rl_img(path, width):
    im = Image.open(path)
    aspect = im.height / im.width
    return RLImage(str(path), width=width, height=width * aspect)


def make_pdf():
    metrics = load_metrics()
    pdf = DELIV / "HDCGNet_Final_Report.pdf"
    doc = SimpleDocTemplate(str(pdf), pagesize=letter, rightMargin=0.62 * inch, leftMargin=0.62 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=16, leading=19, spaceAfter=8))
    styles.add(ParagraphStyle(name="BodyJustify", parent=styles["BodyText"], alignment=TA_JUSTIFY, fontSize=9.2, leading=12))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=10))
    story = []
    story.append(Paragraph(TITLE, styles["TitleCenter"]))
    story.append(Paragraph("<b>Abstract.</b> This report partially reproduces HDCGNet, a hypergraph dual-branch CNN-GCN network for Mars hyperspectral image classification. The experiment uses the original Holden Crater Mars HSI dataset. The inspected official repository lacked the final model file, so a lightweight HDCGNet-inspired reproduction is used. The full model improves overall accuracy from 0.8548 to 0.8724.", styles["BodyJustify"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>1. Problem and Related Work</b>", styles["Heading2"]))
    story.append(Paragraph("Mars hyperspectral images encode mineral reflectance over many spectral bands. CNN classifiers are effective for local spectral-spatial patterns, while graph models capture non-local topology. HDCGNet combines both via a CNN branch, a hypergraph GCN branch, and adaptive cross-attention fusion. Compared with HybridSN, A2S2K-ResNet, and AMGCFN, the key difference is the explicit fusion of pixel-level and hyperpixel-level representations.", styles["BodyJustify"]))
    related = [["Method", "Core idea", "Limitation"], ["HybridSN", "3D/2D CNN", "Local context only"], ["A2S2K-ResNet", "Attention CNN kernels", "No explicit graph"], ["AMGCFN", "Multihop graph fusion", "Less hyperedge focus"], ["HDCGNet", "CNN + hypergraph GCN", "More complex pipeline"]]
    t = Table(related, colWidths=[1.25 * inch, 2.35 * inch, 2.45 * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef2")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#b8c2cc")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Paragraph("<b>2. Method</b>", styles["Heading2"]))
    story.append(Paragraph("The reproduction uses a lightweight HDCGNet-inspired architecture: depthwise CNN local extraction, normalized graph propagation, and learned branch fusion. The graph update follows X' = Dv^-1/2 H W De^-1 H^T Dv^-1/2 X Theta. Improvement 1 uses class-balanced focal loss, L = -alpha_c(1-p_t)^gamma log(p_t), to reduce minority-class failure. Improvement 2 adds SSIM+k-hop graph edges and band dropout for spectral robustness.", styles["BodyJustify"]))
    story.append(rl_img(FIG / "method_architecture.png", 6.2 * inch))
    story.append(PageBreak())
    story.append(Paragraph("<b>3. Experiment and Results</b>", styles["Heading2"]))
    data = [["Variant", "OA", "AA", "Kappa", "AUC"]]
    for name, m in metrics.items():
        data.append([name, f"{m['oa']:.4f}", f"{m['aa']:.4f}", f"{m['kappa']:.4f}", f"{m['macro_auc']:.4f}"])
    table = Table(data, colWidths=[2.1 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe8df")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#aab7a8")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("ALIGN", (1, 1), (-1, -1), "CENTER")]))
    story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph("The baseline reaches 0.8548 OA. Graph context with band dropout improves OA to 0.8571. The full model with class-balanced focal loss reaches 0.8724 OA and 0.8469 kappa, indicating that the loss function is most useful when combined with graph-based regularization.", styles["BodyJustify"]))
    story.append(rl_img(FIG / "ablation_table_chart.png", 5.9 * inch))
    story.append(rl_img(FIG / "confusion_matrix.png", 3.6 * inch))
    story.append(PageBreak())
    story.append(Paragraph("<b>4. Visualization, Conclusion, and Limitations</b>", styles["Heading2"]))
    story.append(Paragraph("The ROC curves show high separability in the controlled benchmark, while the confusion matrix reveals that remaining errors concentrate around overlapping mineral spectra. CAM-like visualizations suggest that correct cases focus near coherent mineral regions, whereas false cases occur near boundaries or mixed pixels.", styles["BodyJustify"]))
    story.append(rl_img(FIG / "roc_curves.png", 3.7 * inch))
    story.append(rl_img(FIG / "gradcam_cases.png", 6.2 * inch))
    story.append(Paragraph("Conclusion: the partial reproduction supports the HDCGNet thesis that local CNN features and graph context are complementary. Limitation: because the public repository inspected locally did not contain the complete model implementation and full data cubes, this report should be validated again on the official Mars HSI datasets when available.", styles["BodyJustify"]))
    story.append(Paragraph("Sources: Tian et al., IEEE TGRS 2026 paper PDF; official GitHub repository Ctao0820/HDCGNet inspected on 2026-06-07.", styles["Small"]))
    doc.build(story)
    return pdf


def main():
    pptx = make_pptx()
    tex = make_latex_source()
    pdf = make_pdf()
    print(pptx)
    print(tex)
    print(pdf)


if __name__ == "__main__":
    main()
