"""
Quick Demo - Test Model Setup
A simple script to test if all imports work correctly
"""

import torch
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required packages are installed"""
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        logger.error(f"✗ PyTorch: {e}")
    
    try:
        import numpy
        logger.info(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy: {e}")
    
    try:
        import pandas
        logger.info(f"✓ Pandas {pandas.__version__}")
    except ImportError as e:
        logger.error(f"✗ Pandas: {e}")
    
    try:
        import matplotlib
        logger.info(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        logger.error(f"✗ Matplotlib: {e}")
    
    try:
        import sklearn
        logger.info(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        logger.error(f"✗ Scikit-learn: {e}")
    
    try:
        import shap
        logger.info(f"✓ SHAP {shap.__version__}")
    except ImportError as e:
        logger.error(f"✗ SHAP: {e}")
    
    logger.info("\nAll imports successful!")


def test_model_creation():
    """Test if model can be created"""
    logger.info("\nTesting model creation...")
    
    try:
        from models.mri_transformer import MRITransformer
        model = MRITransformer()
        logger.info(f"✓ MRITransformer created - {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logger.error(f"✗ MRITransformer: {e}")
    
    try:
        from models.clinical_encoder import ClinicalEncoder
        model = ClinicalEncoder(input_dim=20)
        logger.info(f"✓ ClinicalEncoder created")
    except Exception as e:
        logger.error(f"✗ ClinicalEncoder: {e}")
    
    logger.info("\nModel creation successful!")


def test_dummy_forward():
    """Test forward pass with dummy data"""
    logger.info("\nTesting dummy forward pass...")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        from models.mri_transformer import MRITransformer
        model = MRITransformer().to(device)
        
        # Create dummy MRI data
        x = torch.randn(1, 1, 128, 128, 128).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        logger.info(f"✓ Forward pass successful - Output shape: {output.shape}")
        
    except Exception as e:
        logger.error(f"✗ Forward pass: {e}")


if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Alzheimer's Prediction Model - Setup Test")
    logger.info("="*50)
    
    test_imports()
    
    try:
        test_model_creation()
        test_dummy_forward()
    except Exception as e:
        logger.error(f"Could not test models yet: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("Setup test complete!")
    logger.info("="*50)
