"""Integration test for app with model."""

import sys
from pathlib import Path
import pandas as pd
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from models.longitudinal_multimodal import LongitudinalMultimodalModel

def test_app_integration():
    """Test the complete app workflow."""
    print("="*60)
    print("COMPREHENSIVE APP INTEGRATION TEST")
    print("="*60)
    
    # Test 1: Model loads correctly
    print("\n1. Testing model initialization...")
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
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False
    
    # Test 2: Single patient prediction (as would happen in Risk Prediction page)
    print("\n2. Testing single patient risk prediction (Risk Prediction page)...")
    try:
        clinical_data = {
            "age": 72.0,
            "education_years": 12.0,
            "mmse": 24.0,
            "apoe4": 1.0,
            "cdrsb": 1.5,
            "hippocampus_volume": 2.4,
            "ventricle_volume": 1.2,
            "brain_volume": 1200.0,
        }
        mri_data = {
            "mri_mean_intensity": 100.0,
            "mri_std_intensity": 25.0,
            "mri_voxel_count": 1200000.0,
            "mri_signal_score": 65.0,
        }
        result = model.predict_patient([clinical_data], [mri_data])
        
        assert "diagnosis_label" in result
        assert "progression_label" in result
        assert result["diagnosis_label"] in ["CN", "MCI", "AD"]
        assert result["progression_label"] in ["Stable", "MCI_to_AD"]
        
        print(f"✓ Prediction successful")
        print(f"  - Diagnosis: {result['diagnosis_label']}")
        print(f"  - Progression risk: {result['progression_label']}")
        print(f"  - Prog prob: {max(result['progression_probabilities'].values())*100:.1f}%")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Longitudinal patient progression (as would happen in Patient Progression page)
    print("\n3. Testing longitudinal patient progression (Patient Progression page)...")
    try:
        # Simulate multiple visits for one patient
        clinical_history = []
        mri_history = []
        
        visits = [
            {
                "clinical": {
                    "age": 70.0,
                    "education_years": 12.0,
                    "mmse": 26.0,
                    "apoe4": 1.0,
                    "cdrsb": 0.5,
                    "hippocampus_volume": 2.5,
                    "ventricle_volume": 1.0,
                    "brain_volume": 1250.0,
                },
                "mri": {
                    "mri_mean_intensity": 105.0,
                    "mri_std_intensity": 20.0,
                    "mri_voxel_count": 1250000.0,
                    "mri_signal_score": 70.0,
                }
            },
            {
                "clinical": {
                    "age": 72.0,
                    "education_years": 12.0,
                    "mmse": 24.0,
                    "apoe4": 1.0,
                    "cdrsb": 1.5,
                    "hippocampus_volume": 2.4,
                    "ventricle_volume": 1.2,
                    "brain_volume": 1200.0,
                },
                "mri": {
                    "mri_mean_intensity": 100.0,
                    "mri_std_intensity": 25.0,
                    "mri_voxel_count": 1200000.0,
                    "mri_signal_score": 65.0,
                }
            }
        ]
        
        for visit in visits:
            clinical_history.append(visit["clinical"])
            mri_history.append(visit["mri"])
        
        result = model.predict_patient(clinical_history, mri_history)
        
        assert "diagnosis_label" in result
        assert "progression_label" in result
        
        print(f"✓ Longitudinal prediction successful")
        print(f"  - Final diagnosis: {result['diagnosis_label']}")
        print(f"  - Progression risk: {result['progression_label']}")
        print(f"  - Diagnosis probabilities: {result['diagnosis_probabilities']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Data without MRI (fallback mode)
    print("\n4. Testing prediction without MRI data (fallback mode)...")
    try:
        clinical_data = {
            "age": 72.0,
            "education_years": 12.0,
            "mmse": 24.0,
            "apoe4": 1.0,
            "cdrsb": 1.5,
            "hippocampus_volume": 2.4,
            "ventricle_volume": 1.2,
            "brain_volume": 1200.0,
        }
        empty_mri = {
            "mri_mean_intensity": 0.0,
            "mri_std_intensity": 0.0,
            "mri_voxel_count": 0.0,
            "mri_signal_score": 0.0,
        }
        result = model.predict_patient([clinical_data], [empty_mri])
        
        assert "diagnosis_label" in result
        print(f"✓ Fallback mode works (no MRI data)")
        print(f"  - Diagnosis: {result['diagnosis_label']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_app_integration()
    sys.exit(0 if success else 1)
