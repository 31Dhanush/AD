# Explainable Cross-Attention Transformer for Alzheimer's Disease Prediction

## Overview

This project implements a state-of-the-art deep learning model for predicting Alzheimer's disease progression using multimodal data:
- **3D MRI Images**: Brain structure and morphometry
- **Clinical Data**: Demographics, biomarkers, cognitive assessments
- **Longitudinal Sequences**: Temporal disease progression tracking

The model uses a cross-attention transformer architecture to fuse multimodal data and provides explainability through attention maps, Grad-CAM, Integrated Gradients, and SHAP analysis.

## Project Structure

```
Alzheimer_Longitudinal_AI/
├── data/
│   ├── raw/                 # Original MRI (NIfTI) and clinical data
│   ├── processed/           # Preprocessed data ready for training
│   └── clinical/            # Clinical tabular data
├── preprocessing/
│   ├── mri_preprocess.py    # MRI loading, normalization, skull stripping
│   ├── clinical_preprocess.py # Clinical data handling and encoding
│   └── longitudinal_dataset.py # Temporal sequence dataset
├── models/
│   ├── mri_transformer.py        # 3D Vision Transformer for MRI
│   ├── clinical_encoder.py       # Clinical feature encoder
│   ├── cross_attention.py        # Cross-modal attention mechanisms
│   ├── temporal_transformer.py   # Temporal disease progression modeling
│   └── multimodal_model.py       # Complete integrated model
├── training/
│   ├── train.py             # Training loop and checkpointing
│   ├── losses.py            # Custom loss functions (Focal, Temporal, Contrastive)
│   └── validate.py          # Validation procedures
├── evaluation/
│   ├── metrics.py           # Classification metrics
│   ├── confusion_matrix.py  # Confusion matrix analysis
│   └── roc_analysis.py      # ROC curves and AUC
├── explainability/
│   ├── mri_attention.py     # Attention map visualization
│   ├── gradcam.py           # Grad-CAM and Grad-CAM++
│   ├── integrated_gradients.py # Integrated Gradients attribution
│   └── shap_analysis.py     # SHAP values and feature importance
├── experiments/
│   ├── cnn_baseline.py      # CNN/ResNet baselines
│   ├── multimodal_baseline.py # Simple multimodal fusion baseline
│   └── ablation.py          # Ablation study implementation
├── models_saved/            # Saved model checkpoints
│   └── best_model.pth
├── app/
│   └── app.py              # Streamlit web application
└── requirements.txt        # Python dependencies
```

## Installation

### 1. Clone and Setup Environment
```bash
cd "ALZHEIMER'S DISEASE"
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Jupyter (Optional)
```bash
pip install jupyter notebook
```

## Quick Start

### Data Preparation

```python
from preprocessing.mri_preprocess import MRIPreprocessor
from preprocessing.clinical_preprocess import ClinicalPreprocessor

# Preprocess MRI
mri_preprocessor = MRIPreprocessor(target_shape=(128, 128, 128))
mri_data = mri_preprocessor.preprocess("path/to/mri.nii.gz")

# Preprocess Clinical Data
clinical_preprocessor = ClinicalPreprocessor()
clinical_data = clinical_preprocessor.preprocess("path/to/clinical.csv")
```

### Create Dataset

```python
from preprocessing.longitudinal_dataset import LongitudinalDataset
from torch.utils.data import DataLoader

dataset = LongitudinalDataset(
    mri_data=mri_list,
    clinical_data=clinical_df,
    labels=labels,
    patient_ids=patient_ids,
    timepoints=timepoints,
    sequence_length=3
)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### Train Model

```python
from models.multimodal_model import ExplainableMultimodalTransformer
from training.train import Trainer
from training.losses import MultimodalLoss

model = ExplainableMultimodalTransformer(num_classes=2)
criterion = MultimodalLoss()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    learning_rate=1e-3,
    device='cuda'
)

trainer.train(num_epochs=100, patience=15)
```

### Inference with Explainability

```python
from explainability.gradcam import GradCAM
from explainability.shap_analysis import SHAPAnalysis

# Load model
model = ExplainableMultimodalTransformer()
model.load_state_dict(torch.load('models_saved/best_model.pth'))

# Grad-CAM visualization
gradcam = GradCAM(model, target_layer='mri_encoder.norm')
cam = gradcam(mri_data)

# SHAP analysis for clinical features
explainer, shap_values = SHAPAnalysis.clinical_feature_shap(
    model, clinical_data, feature_names=clinical_features
)
```

## Model Architecture

### MRI Transformer (Vision Transformer for 3D)
- Patches: 3D volume divided into 16×16×16 patches
- Embedding: Linear projection to 256-dim embeddings
- Transformer: 6 layers, 8 heads, 1024-dim MLP
- Output: Class token for downstream fusion

### Clinical Encoder
- Input: Normalized clinical features
- Architecture: 2 hidden layers (128 → 64) + LSTM for temporal
- Output: 64-dim embeddings

### Cross-Attention Fusion
- Bidirectional attention between MRI and clinical features
- Gate-based fusion with learned weighting
- Modality-specific attention maps for explainability

### Temporal Transformer
- Models disease progression over time
- 4 transformer layers with 8 attention heads
- Outputs both classification and trajectory prediction

## Explainability Methods

### 1. Attention Maps
- Visualize which MRI patches contribute to predictions
- Per-layer attention analysis
- Important region extraction

### 2. Grad-CAM / Grad-CAM++
- Gradient-based class activation maps
- Identifies discriminative regions
- Works for any layer in the network

### 3. Integrated Gradients
- Attribution scores for input features
- Path integration from baseline to input
- Smooth attribution without saturation artifacts

### 4. SHAP Values
- Model-agnostic explainability
- Clinical feature importance
- Feature interaction analysis
- Individual prediction explanations

## Evaluation Metrics

```
- Accuracy, Precision, Recall, F1-Score
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- ROC-AUC
- Age-stratified performance analysis
```

## Baselines & Experiments

### CNN Baseline
3D CNN with standard architecture for comparison

### Multimodal Baseline
Simple concatenation-based fusion of MRI and clinical branches

### Ablation Study
Systematic removal of model components to assess contribution:
- MRI encoder removal
- Clinical encoder removal
- Cross-attention removal
- Temporal modeling removal

## Results

Expected performance on Alzheimer's prediction task:
- **Accuracy**: ~92%
- **Sensitivity**: ~89%
- **Specificity**: ~95%
- **AUC-ROC**: ~0.96

## Web Application

Run the Streamlit app for interactive inference and visualization:

```bash
streamlit run app/app.py
```

Features:
- Model inference on new patient data
- Attention map visualization
- Feature importance analysis (SHAP)
- ROC curves and confusion matrices
- Model performance metrics
- Ablation study results

## Configuration

Modify hyperparameters in model initialization:

```python
model_config = {
    'mri_config': {
        'img_size': 128,
        'patch_size': 16,
        'embed_dim': 256,
        'num_layers': 6
    },
    'clinical_config': {
        'input_dim': 20,
        'embed_dim': 64,
        'num_layers': 2
    },
    'temporal_config': {
        'input_dim': 128,
        'num_layers': 4
    }
}
```

## Citations

If you use this code, please cite:

```bibtex
@article{alzheimer_transformer_2024,
  title={An Explainable Cross-Attention Transformer for 
         Longitudinal Prediction of Alzheimer's Disease Progression},
  author={Your Name},
  year={2024}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- Built with PyTorch
- Vision Transformer inspired by ViT (Dosovitskiy et al., 2021)
- Explainability using Grad-CAM, Integrated Gradients, and SHAP

---

**Last Updated**: 2024
