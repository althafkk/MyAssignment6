import torch
import torch.nn as nn
from model import Net
from torchsummary import summary
import sys

def check_model_requirements():
    # Initialize model
    model = Net()
    
    # Check 1: Parameter Count (should be < 20k)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"
    
    # Check 2: Batch Normalization
    has_batchnorm = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    print(f"Has BatchNorm: {has_batchnorm}")
    assert has_batchnorm, "Model should use Batch Normalization"
    
    # Check 3: Dropout
    has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
    print(f"Has Dropout: {has_dropout}")
    assert has_dropout, "Model should use Dropout"
    
    # Check 4: GAP or FC Layer
    has_gap = any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules())
    has_fc = any(isinstance(m, nn.Linear) for m in model.modules())
    print(f"Has GAP: {has_gap}, Has FC: {has_fc}")
    assert has_gap or has_fc, "Model should use either Global Average Pooling or Fully Connected layer"
    
    print("All checks passed successfully!")
    return True

if __name__ == "__main__":
    try:
        check_model_requirements()
        sys.exit(0)
    except AssertionError as e:
        print(f"Check failed: {str(e)}")
        sys.exit(1) 