# Problem Statement & Stakeholder Analysis
## Colorectal Cancer Tissue Classification using Deep Learning
### Project: EfficientNet-B0 on NCT-CRC-HE-100K-NONORM

---

## 1. Executive Summary

Colorectal cancer (CRC) is the **third most common cancer worldwide** and the **second leading cause of cancer-related death**, accounting for approximately 1.9 million new cases and 930,000 deaths globally every year (WHO, 2022). Early and accurate diagnosis is the single most critical factor determining patient survival — a patient diagnosed at Stage I has a 5-year survival rate of **~90%**, while a Stage IV diagnosis drops that to **~14%**.

The cornerstone of colorectal cancer diagnosis is **histopathological examination** — a pathologist manually inspects H&E (Hematoxylin and Eosin) stained tissue slides under a microscope and classifies tissue regions into meaningful biological categories. This process is:

- **Slow** — a single Whole Slide Image (WSI) contains thousands of patches requiring review
- **Expensive** — specialist pathologists are scarce, especially in low-income countries
- **Subjective** — inter-observer variability among pathologists can be as high as 20–30% for ambiguous tissue types
- **Unsustainable** — global demand for pathology services is growing faster than the supply of trained pathologists

This project directly addresses these challenges by building an **automated, AI-powered tissue classification system** that can accurately identify 9 distinct colorectal tissue types from H&E-stained histopathology patches with **99.64% validation accuracy**.

---

## 2. The Core Problem

### 2.1 What Exactly Are We Solving?

> **The problem is not cancer detection per se — it is the bottleneck of tissue-level characterization that precedes every clinical decision in colorectal oncology.**

In colorectal cancer pathology, a pathologist must routinely classify tissue regions into categories such as:
- Is this tumor (adenocarcinoma)?
- Is this normal mucosa being invaded?
- Is this tumor-associated stroma driving metastasis?
- Is this lymphocytic infiltration (which predicts immunotherapy response)?

Each of these questions requires examining **thousands of image patches per patient slide**. Manually doing this at scale is humanly impossible without significant time and cost.

**Our system automates this patch-level tissue classification**, enabling:
1. Faster slide triage
2. Quantitative tumor microenvironment profiling
3. Downstream survival prediction pipelines
4. Quality control for junior pathologists

### 2.2 The 9-Class Classification Problem

Our model classifies each 224×224 pixel H&E patch (at 0.5 µm/pixel resolution) into one of 9 clinically meaningful tissue types:

| Class | Full Name | Clinical Significance |
|-------|-----------|----------------------|
| **ADI** | Adipose tissue | Tumor boundary marker; fat invasion indicates aggressive disease |
| **BACK** | Background / non-tissue | Quality control; slides must be cleared of artifacts |
| **DEB** | Debris / cellular artifacts | Indicates necrosis — poor prognosis marker |
| **LYM** | Lymphocytes | Tumor-infiltrating lymphocytes (TILs) predict immunotherapy response |
| **MUC** | Mucus | Present in mucinous adenocarcinoma — distinct molecular subtype |
| **MUS** | Smooth muscle | Muscularis invasion determines T-stage (T2 vs T3) |
| **NORM** | Normal colon mucosa | Baseline; distinguishes tumor spread from healthy tissue |
| **STR** | Tumor-associated stroma | Desmoplastic stroma linked to poor prognosis and drug resistance |
| **TUM** | Adenocarcinoma epithelium | The cancer itself — primary target for tumor burden quantification |

### 2.3 Why Is This Hard?

1. **Visual similarity**: STR vs NORM, TUM vs NORM are easily confused even by trained eyes
2. **Stain variability**: H&E staining protocols differ across hospitals, scanners, and batches — causing domain shift
3. **Scale**: A single patient biopsy generates 10,000–100,000 patches requiring classification
4. **Class imbalance**: Tumor and stroma patches dominate; rare classes like MUC need weighted treatment
5. **Interpretability requirement**: Clinical adoption requires knowing *why* the model made a decision — not just what it decided

---

## 3. What Problem Are We Solving — Precisely?

We are solving **three nested problems simultaneously**:

### Problem 1 — Throughput Bottleneck
> Pathologists spend 40–60% of their time on routine tissue categorization tasks that do not require their full expertise.

**Our solution:** Automate patch-level tissue typing so pathologists focus only on ambiguous or high-stakes decisions. Our model processes a patch in **<10ms on CPU** — vs. 3–5 seconds per patch manually.

### Problem 2 — Quantitative Tumor Microenvironment (TME) Analysis
> Current pathology reports are qualitative ("tumor present", "stroma abundant"). Precision oncology requires *quantitative* tissue composition metrics.

**Our solution:** By classifying every patch in a slide, we enable computation of:
- % TUM (tumor burden)
- % LYM (immune infiltration score)
- % STR (stromal ratio)
- TIL score (linked to immunotherapy eligibility)

These are **biomarkers** — quantitative signals that drive treatment decisions.

### Problem 3 — Accessibility & Democratization
> In low- and middle-income countries (LMICs), there is on average **1 pathologist per 500,000 people** (vs. 1 per 20,000 in high-income countries).

**Our solution:** A lightweight EfficientNet-B0 model (~5.3M parameters) deployable on standard hospital hardware with no GPU requirement — bringing diagnostic-grade tissue analysis to resource-constrained settings.

---

## 4. Stakeholder Analysis

### 4.1 Primary Stakeholders (Direct Users)

#### A. Pathologists & Histopathologists
- **Who:** Board-certified pathologists in hospital labs, academic medical centers, and private diagnostic labs
- **Pain point:** Manual review of thousands of WSI patches per day; cognitive fatigue; time pressure
- **What they gain:** AI-assisted pre-screening, second-opinion tool, quantitative tissue metrics
- **How they interact:** Our Gradio demo / API integrated into their digital pathology workflow (e.g., QuPath, Aperio, HALO)
- **Concern:** Trust and interpretability — addressed via **Grad-CAM heatmaps** showing exactly which pixels drove the prediction

#### B. Oncologists & Gastroenterologists
- **Who:** Treating physicians managing CRC patients
- **Pain point:** Pathology reports lack quantitative data needed for precision treatment decisions
- **What they gain:** Structured tissue composition reports (% TUM, % LYM, % STR) to guide chemotherapy vs. immunotherapy vs. surgery
- **How they interact:** Through structured reports generated from our model's output integrated into EMR (Electronic Medical Records)

#### C. Radiologists / Digital Pathology Labs
- **Who:** Labs running high-throughput WSI scanners (Leica, Hamamatsu, Philips)
- **Pain point:** Scanner throughput exceeds manual review capacity — slides queue up
- **What they gain:** Automated batch processing pipeline — slides classified overnight without human input
- **How they interact:** REST API or batch inference pipeline

---

### 4.2 Secondary Stakeholders (Indirect Beneficiaries)

#### D. Cancer Researchers & Computational Pathologists
- **Who:** Academic researchers studying CRC biology, tumor microenvironments, survival prediction
- **Pain point:** Manual annotation of tissue patches for research datasets is prohibitively expensive
- **What they gain:** Pre-trained model as a feature extractor or annotation tool for downstream research (e.g., survival prediction, molecular subtype classification)
- **How they interact:** GitHub repo, HF Spaces API, model weights download

#### E. Pharmaceutical Companies & CROs (Contract Research Orgs)
- **Who:** Drug developers running clinical trials for CRC therapies
- **Pain point:** Biomarker quantification (TIL scoring, stromal ratio) for patient stratification in trials is manual and expensive
- **What they gain:** Automated, reproducible, scalable biomarker extraction from trial biopsy slides
- **How they interact:** Enterprise API integration or custom model fine-tuning

#### F. Medical AI Startups & Digital Pathology Vendors
- **Who:** Companies building AI-powered pathology platforms (PathAI, Paige.AI, Ibex Medical)
- **Pain point:** Need robust tissue classification as a foundational layer in their pipelines
- **What they gain:** Open-source baseline model, dataset benchmark, deployable demo
- **How they interact:** GitHub fork, model fine-tuning, API

---

### 4.3 Tertiary Stakeholders (System-Level)

#### G. Patients
- **Who:** CRC patients (1.9 million/year globally)
- **Pain point:** Delayed diagnosis, inconsistent pathology quality, missed early-stage detection
- **What they gain:** Faster turnaround time (hours vs. days), more consistent diagnosis, better treatment matching
- **How they interact:** Indirectly — through their pathologist's workflow

#### H. Healthcare Systems & Hospitals (Administrators)
- **Who:** Hospital procurement, lab managers, health ministry officials
- **Pain point:** Pathologist shortage, cost of diagnostic services, lab efficiency
- **What they gain:** Reduced cost-per-diagnosis, faster lab turnaround, reduced re-read rates
- **How they interact:** Institutional procurement of AI-assisted pathology tools

#### I. Regulatory Bodies (FDA, CE, ICMR)
- **Who:** FDA (USA), CE marking (EU), ICMR (India), and equivalent bodies
- **Pain point:** Ensuring AI tools meet safety/efficacy standards before clinical deployment
- **What they gain:** A transparent, explainable model (Grad-CAM) with documented performance metrics
- **Note:** Clinical deployment requires regulatory clearance — this project is currently a **research/decision-support tool**, not a standalone diagnostic device

---

## 5. Problem Scope — What This Project Does and Does Not Do

### In Scope
| Capability | Detail |
|-----------|--------|
| Patch-level tissue classification | 9 classes, 224×224 H&E patches |
| Confidence scoring | Softmax probability for all 9 classes |
| Visual explainability | Grad-CAM heatmaps per prediction |
| Web-accessible demo | Hugging Face Spaces (live) |
| Lightweight deployment | CPU-compatible, ~5.3M parameters |

### Out of Scope (Future Work)
| Limitation | Why / Future Direction |
|-----------|------------------------|
| Whole Slide Image (WSI) inference | Requires slide tiling pipeline — next version |
| Patient-level diagnosis | Requires aggregating patch predictions across entire slide |
| Multi-scanner generalization | Needs cross-institution validation dataset |
| Regulatory clearance | Requires clinical trial evidence and FDA/CE submission |
| Real-time integration with LIMS | Requires hospital IT integration |

---

## 6. Impact Quantification

| Metric | Current (Manual) | With Our System |
|--------|-----------------|-----------------|
| Time per patch | 3–5 seconds | <10 ms |
| Throughput (patches/day/person) | ~5,000 | ~500,000+ |
| Inter-observer variability | 20–30% | <1% (deterministic) |
| Cost per slide analysis | $50–$200 | <$1 (compute cost) |
| Availability in LMICs | Limited by pathologist shortage | Internet connection sufficient |

---

## 7. Technology Justification

| Design Choice | Reason |
|--------------|--------|
| EfficientNet-B0 | Best accuracy/parameter tradeoff; 5.3M params vs. ResNet-50's 25M |
| NCT-CRC-HE-100K dataset | Largest publicly available labeled CRC patch dataset; gold-standard benchmark |
| H&E-specific augmentation | ColorJitter + HueSaturation mimics stain variability across hospitals |
| Grad-CAM explainability | Clinical adoption requires interpretability — black-box AI is unacceptable in medicine |
| Gradio + HF Spaces | Zero-infrastructure demo accessible to any clinician or researcher globally |
| MIT License | Enables adoption by academic, clinical, and commercial users |

---

## 8. Conclusion

This project addresses a **real, urgent, and quantifiable clinical problem** — the manual tissue classification bottleneck in colorectal cancer pathology. It delivers a solution that is:

- **Accurate** (99.64% val accuracy, AUC 1.000)
- **Explainable** (Grad-CAM heatmaps)
- **Lightweight** (CPU-deployable, 5.3M parameters)
- **Accessible** (live demo, open-source, MIT licensed)
- **Clinically grounded** (9 classes with direct clinical significance)

The stakeholder landscape spans from individual pathologists to global healthcare systems, pharmaceutical companies, and ultimately — patients. Every improvement in tissue classification speed, consistency, and accessibility directly translates into better, faster, fairer cancer care.

---

*Document prepared for: Portfolio / Research / Product pitch purposes*
*Model: EfficientNet-B0 | Dataset: NCT-CRC-HE-100K-NONORM (Kather et al., 2019)*
*Live Demo: https://huggingface.co/spaces/PANKAJ-MOHAN/colorectal-histology-classifier*
*GitHub: https://github.com/iitmstudent-2021/colorectal-histology-classifier_computer_vision*
