import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from models.longitudinal_multimodal import LongitudinalMultimodalModel

# Test 1: Model instantiation
print("Test 1: Model instantiation...")
try:
    model = LongitudinalMultimodalModel(
        mri_dim=4,
        clinical_dim=8,
        hidden_dim=24,
        fusion_dim=24,
        num_timepoints=2,
        num_classes=3,
        progression_classes=2,
    )
    print("✓ Model created successfully")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: predict_patient with single visit
print("\nTest 2: predict_patient with single visit...")
try:
    clinical_history = [
        {
            "age": 72.0,
            "education_years": 12.0,
            "mmse": 24.0,
            "apoe4": 1.0,
            "cdrsb": 1.5,
            "hippocampus_volume": 2.4,
            "ventricle_volume": 1.2,
            "brain_volume": 1200.0,
        }
    ]
    mri_history = [
        {
            "mri_mean_intensity": 100.0,
            "mri_std_intensity": 25.0,
            "mri_voxel_count": 1200000.0,
            "mri_signal_score": 65.0,
        }
    ]
    result = model.predict_patient(clinical_history, mri_history)
    print(f"✓ predict_patient result: {result['diagnosis_label']}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 3: predict_patient with two visits (longitudinal)
print("\nTest 3: predict_patient with two visits (longitudinal)...")
try:
    clinical_history = [
        {
            "age": 70.0,
            "education_years": 12.0,
            "mmse": 26.0,
            "apoe4": 1.0,
            "cdrsb": 0.5,
            "hippocampus_volume": 2.5,
            "ventricle_volume": 1.0,
            "brain_volume": 1250.0,
        },
        {
            "age": 72.0,
            "education_years": 12.0,
            "mmse": 24.0,
            "apoe4": 1.0,
            "cdrsb": 1.5,
            "hippocampus_volume": 2.4,
            "ventricle_volume": 1.2,
            "brain_volume": 1200.0,
        },
    ]
    mri_history = [
        {
            "mri_mean_intensity": 105.0,
            "mri_std_intensity": 20.0,
            "mri_voxel_count": 1250000.0,
            "mri_signal_score": 70.0,
        },
        {
            "mri_mean_intensity": 100.0,
            "mri_std_intensity": 25.0,
            "mri_voxel_count": 1200000.0,
            "mri_signal_score": 65.0,
        },
    ]
    result = model.predict_patient(clinical_history, mri_history)
    print(f"✓ predict_patient (2 visits) result:")
    print(f"  - Diagnosis: {result['diagnosis_label']}")
    print(f"  - Progression: {result['progression_label']}")
    print(f"  - Diagnosis probs: {result['diagnosis_probabilities']}")
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
