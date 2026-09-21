# Complete Implementation Plan: Alzheimer's Disease Progression Prediction

## STAGE 1: ARCHITECTURE & ASSUMPTIONS CONFIRMATION

### ✅ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LONGITUDINAL PATIENT DATA                     │
│              Multiple visits per patient (T0, T1, T2...)         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
    MRI Sequence              Clinical Sequence
   (T0,T1,T2...)            (T0,T1,T2...)
         │                           │
         ▼                           ▼
    ┌─────────────┐            ┌──────────────┐
    │  3D CNN     │            │ Clinical     │
    │  (local)    │            │ Encoder      │
    │  features   │            │ (MLP-based)  │
    └──────┬──────┘            └──────┬───────┘
           │                          │
           ▼                          ▼
    ┌─────────────┐            ┌──────────────┐
    │  3D ViT     │            │ Embedding    │
    │  (global)   │            │ (D=64)       │
    │  attention  │            └──────┬───────┘
    └──────┬──────┘                   │
           │                          │
           └────────────┬─────────────┘
                        │
                        ▼
            ┌───────────────────────────┐
            │ Bidirectional Cross-      │
            │ Attention (MRI ↔ Clinical)│
            │ Fusion                    │
            └────────────┬──────────────┘
                        │
            Visit Multimodal Embeddings
            [emb_t0, emb_t1, emb_t2, ...]
                        │
                        ▼
            ┌───────────────────────────┐
            │  Temporal Transformer     │
            │  (temporal self-attention)│
            │  + position encoding      │
            └────────────┬──────────────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │   Patient-Level Embedding    │
        └────────────┬─────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
    Classification Head      Progression Head
   (CN/MCI/AD softmax)    (MCI→AD binary sigmoid)
        │                         │
        ▼                         ▼
    CN/MCI/AD              Progression Risk
    Probabilities          (0 = stable, 1 = converter)
```

### 📊 Data Flow Specification

**Input Format (Per Patient):**
- Patient ID (anonymized)
- Variable number of visits (typically 2-5)
- Per visit:
  - MRI: 3D volume [1, D, H, W] or preprocessed features
  - Clinical: vector of tabular features [n_features]
  - Visit timestamp/time-from-baseline
  - Diagnosis label (CN, MCI, AD)

**Sequence Representation:**
- All patients padded to max_visits (configurable)
- Attention masks for actual visits
- Time information: months/years from baseline

**Target Labels:**
- Classification: CN=0, MCI=1, AD=2
- Progression: 
  - Only for MCI patients at baseline
  - 1 if converts to AD within prediction horizon
  - 0 if remains MCI
  - -1 (masked) if insufficient follow-up

### 🔧 Key Design Decisions

1. **MRI Encoding Strategy**
   - 3D CNN: Captures local anatomical patterns
   - 3D ViT: Captures global relationships via attention
   - Concatenation at embedding level

2. **Clinical Handling**
   - Separate feature extraction per visit
   - LSTM-based temporal encoding of clinical data
   - Normalization: fit ONLY on training data

3. **Cross-Attention Mechanism**
   - Bidirectional: MRI→Clinical AND Clinical→MRI
   - Multi-head attention (default 8 heads)
   - Residual connections + Layer Norm

4. **Temporal Modeling**
   - Transformer-based (NOT LSTM at sequence level)
   - Positional encoding: learnable embeddings
   - Padding mask: ignore padded visits

5. **Multi-task Learning**
   - Loss 1: Classification (CN/MCI/AD) - weight λ₁
   - Loss 2: Progression (MCI→AD) - weight λ₂
   - Per-sample masking for invalid targets

6. **Explainability Strategy**
   - MRI: Grad-CAM heatmaps (3D → axial/sagittal/coronal slices)
   - Clinical: SHAP feature importance
   - Cross-attention: Weight visualization
   - Temporal: Per-visit attribution changes

### ⚡ Data Leakage Prevention

**Critical Rules:**
1. ✅ Train/test split is PATIENT-level (not visit-level)
2. ✅ ALL visits from one patient in same split
3. ✅ Preprocessing fit ONLY on training set
4. ✅ No future diagnosis used as input feature
5. ✅ Target label construction from actual longitudinal data only
6. ✅ No synthetic labels unless explicitly for testing

### 📁 Directory Structure (Confirmed)

```
Alzheimer_Longitudinal_AI/
├── data/
│   ├── raw/              # Original data (user provides)
│   ├── processed/        # Preprocessed data
│   └── metadata/         # CSV with patient info
├── preprocessing/        # Data processing pipelines
├── models/               # Model components
├── training/             # Training scripts
├── evaluation/           # Metrics & analysis
├── explainability/       # Attribution methods
├── experiments/          # Baselines & ablation
├── app/                  # Streamlit dashboard
├── config/               # Configuration files
├── utils/                # Helper functions
├── scripts/              # Entry point scripts
├── models_saved/         # Trained checkpoints
├── results/              # Output metrics/plots
├── tests/                # Unit tests
├── requirements.txt
├── config.yaml
└── README.md
```

### 🎯 Core Tasks & Objectives

**Primary Task (Progression Prediction):**
- Input: Baseline + follow-up MRI + clinical data
- Output: P(MCI → AD within prediction horizon)
- Evaluation: AUC, sensitivity, specificity, F1

**Auxiliary Task (Diagnosis Classification):**
- Input: Single-visit or latest-visit multimodal data
- Output: P(CN), P(MCI), P(AD)
- Evaluation: Accuracy, macro F1, confusion matrix

**Explainability Objectives:**
1. Identify MRI regions contributing to prediction
2. Identify clinical features (MMSE, CDR, etc.)
3. Show how each modality is weighted via cross-attention
4. Show progression trajectory across visits

### 🔐 Reproducibility Guarantees

- Random seed: 42 (configurable)
- Deterministic: torch, numpy, sklearn
- GPU: optional, graceful fallback to CPU
- Results: same seed = same results (within floating-point precision)

### 📊 Dataset Expectations

**Minimum Requirements:**
- At least 20-30 patients (can start with synthetic)
- MCI subset with longitudinal AD conversions
- Clinical features: age, MMSE, CDR minimum
- MRI: preprocessed 3D volumes (NIfTI or numpy)

**Expected Outputs:**
- Train/validation/test metrics (CSV + JSON)
- Plots: losses, accuracy, ROC curves, confusion matrices
- Explanations: Grad-CAM, SHAP, attention visualizations
- Model checkpoint: best.pth + config

---

## Next Steps
→ STAGE 2: Create complete config.yaml with all parameters
→ STAGE 3: Implement data validation script
→ STAGE 4-8: Data preprocessing pipelines
→ STAGE 9-14: Model components
→ STAGE 15-27: Training, evaluation, explainability, app
