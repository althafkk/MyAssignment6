[![Model Architecture Checks](https://github.com/althafkk/MyAssignment6/actions/workflows/model_checks.yml/badge.svg)](https://github.com/althafkk/MyAssignment6/actions/workflows/model_checks.yml)

# MNIST Classification with PyTorch

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification using PyTorch. The model achieves >99.4% test accuracy while maintaining less than 20k parameters.

## Model Architecture

The CNN architecture includes:
- Batch Normalization
- Dropout for regularization
- Global Average Pooling
- Channel reduction techniques
- No Fully Connected layers

Architecture Overview:
```
Input (28x28x1) →
conv1 (28x28x16) →
conv2 (28x28x32) →
maxpool (14x14x32) →
conv3 (14x14x32) [with channel reduction] →
maxpool (7x7x32) →
conv4 (7x7x16) →
GAP (1x1x16) →
final_conv (1x1x10)
```

## Project Structure

```
├── model.py           # Model architecture definition
├── train.py          # Training script and utilities
├── test_model.py     # Model testing and validation
├── .github/workflows # GitHub Actions for automated checks
└── README.md         # Project documentation
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- tqdm
- torchsummary

Install dependencies:
```bash
pip install torch torchvision tqdm torchsummary
```

## Model Requirements

The model meets the following requirements:
1. Parameters < 20,000
2. Uses Batch Normalization
3. Uses Dropout
4. Uses Global Average Pooling
5. Achieves >99.4% test accuracy

## Usage

1. Train the model:
```bash
python train.py
```

2. Test model requirements:
```bash
python test_model.py
```

## Training Features

- Data Augmentation:
  - Random Rotation
  - Random Affine Transforms
  - Color Jittering
- OneCycleLR scheduler
- Early stopping at 99.4% accuracy
- Best model checkpoint saving

## GitHub Actions

Automated checks run on every push and pull request to verify:
- Parameter count < 20k
- Use of Batch Normalization
- Use of Dropout
- Use of GAP/FC layer

## Results

The model achieves:
- Test Accuracy: >99.4%
- Total Parameters: <20k
- Training Time: ~20 epochs

## License

This project is open-sourced under the MIT license. 