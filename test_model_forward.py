import torch
from models.longitudinal_multimodal import LongitudinalMultimodalModel

model = LongitudinalMultimodalModel(mri_dim=4, clinical_dim=8, hidden_dim=32, fusion_dim=32, num_timepoints=2)
mri_seq = torch.randn(2, 2, 4)
clinical_seq = torch.randn(2, 2, 8)

try:
    output = model(mri_seq, clinical_seq)
    print("Forward pass OK")
    print(f"Output keys: {output.keys()}")
except Exception as e:
    print(f"ERROR in forward pass: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
