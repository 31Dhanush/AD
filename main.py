"""
Main Training Script - Complete end-to-end training pipeline
Usage: python main.py --clinical-data data/raw/clinical/clinical.csv --mri-dir data/raw/MRI
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import json
import argparse
import sys
import numpy as np

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from utils.seed import set_seed, configure_determinism
from utils.config_loader import ConfigLoader
from utils.logging_utils import setup_logging

from models.multimodal_model import ExplainableMultimodalTransformer
from training.losses import MultiTaskLoss
from training.train import Trainer
from preprocessing.longitudinal_dataset import LongitudinalDataLoader

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train Alzheimer's disease prediction model"
    )
    parser.add_argument('--clinical-data', type=str,
                       default='data/raw/clinical/clinical.csv',
                       help='Path to clinical data CSV')
    parser.add_argument('--mri-dir', type=str,
                       default='data/raw/MRI',
                       help='Path to MRI directory')
    parser.add_argument('--config', type=str,
                       default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def setup_device(device_str):
    """Setup device"""
    if device_str == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = device_str

    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")

    return device


def main():
    """Main training function"""

    # Parse arguments
    args = parse_args()

    # Setup
    set_seed(args.seed)
    configure_determinism(seed=args.seed)
    setup_logging("alzheimer_training", log_dir="logs")

    logger.info("="*80)
    logger.info("ALZHEIMER'S DISEASE PROGRESSION PREDICTION - TRAINING")
    logger.info("="*80)

    # Load config
    config = ConfigLoader(args.config)
    logger.info(f"Loaded config: {args.config}")

    # Setup device
    device = setup_device(args.device)

    # Create dataset
    logger.info("Creating dataset...")

    if not Path(args.clinical_data).exists() or not Path(args.mri_dir).exists():
        logger.warning(
            "Training data not found. Expected clinical CSV at '%s' and MRI directory at '%s'. "
            "The app can still be launched in dashboard mode for uploaded data.",
            args.clinical_data,
            args.mri_dir,
        )
        return 0

    data_loader = LongitudinalDataLoader(
        clinical_file=args.clinical_data,
        mri_directory=args.mri_dir,
        config=config
    )

    # Create splits (patient-level)
    train_dataset, val_dataset, test_dataset = data_loader.create_patient_splits(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Create model
    logger.info("Creating model...")

    model = ExplainableMultimodalTransformer(
        mri_shape=(96, 112, 96),
        mri_output_dim=config.get('model.mri_output_dim', 256),
        clinical_input_dim=config.get('model.clinical_input_dim', 20),
        clinical_output_dim=config.get('model.clinical_output_dim', 128),
        num_classes=config.get('model.num_classes', 3),
        dropout=config.get('model.dropout', 0.1)
    )

    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create loss function
    logger.info("Creating loss function...")

    criterion = MultiTaskLoss(
        num_classes=config.get('model.num_classes', 3),
        classification_weight=config.get('training.classification_weight', 1.0),
        progression_weight=config.get('training.progression_weight', 0.5),
        adaptive_weighting=config.get('training.adaptive_weighting', False)
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.get('training.weight_decay', 1e-5),
        betas=(0.9, 0.999)
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # Create trainer
    logger.info("Creating trainer...")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=config.get('training.early_stopping_patience', 15)
    )

    # Train
    logger.info("Starting training...")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")

    try:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs
        )

        logger.info("✓ Training completed successfully!")

        # Save history
        history_file = Path(args.checkpoint_dir) / 'training_history.json'
        with open(history_file, 'w') as f:
            history_json = {
                k: [float(v) if not isinstance(v, (int, float)) else v for v in history[k]]
                for k in history
            }
            json.dump(history_json, f, indent=4)
        logger.info(f"Saved training history to {history_file}")

        # Test on test set
        logger.info("\nEvaluating on test set...")
        test_metrics = trainer.validate(test_loader, compute_metrics=True)

        logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics.get('accuracy', 0):.4f}")
        logger.info(f"Test AUC: {test_metrics.get('auc', 0):.4f}")
        logger.info(f"Test F1 (macro): {test_metrics.get('f1_macro', 0):.4f}")

        # Save test results
        test_results_file = Path(args.checkpoint_dir) / 'test_results.json'
        with open(test_results_file, 'w') as f:
            test_results_json = {
                k: float(v) if isinstance(v, (float, np.floating)) else v
                for k, v in test_metrics.items()
                if k != 'confusion_matrix'
            }
            json.dump(test_results_json, f, indent=4)
        logger.info(f"Saved test results to {test_results_file}")

        return 0

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        description='Alzheimer\'s Disease Prediction Pipeline'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run preprocessing only')
    parser.add_argument('--train', action='store_true',
                       help='Run training')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only')
    
    args = parser.parse_args()
    
    pipeline = Pipeline(config_path=args.config)
    
    if args.preprocess:
        pipeline.preprocess_data()
    elif args.train:
        pipeline.run_full_pipeline()
    elif args.evaluate:
        logger.info("Evaluation mode - load model and data")
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
