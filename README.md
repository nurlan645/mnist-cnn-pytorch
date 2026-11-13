# MNIST CNN (PyTorch)

A learning project where I train a Convolutional Neural Network
on the MNIST dataset using PyTorch.

## Features
- 3 convolutional layers with BatchNorm and Dropout
- AdamW optimizer with LinearLR scheduler
- Train / validation loop with progress bars (tqdm)
- Logging to console and file
- Prediction visualization with matplotlib

## Setup
```bash
pip install torch torchvision tqdm matplotlib numpy
python train.py
```

## What I learned
- How a training loop works in PyTorch
- Why BatchNorm and Dropout are used
- How to log and visualize results

## Results
Test accuracy: (00:38:10 | INFO    | Epoch 1/2 | train_loss=0.3590 train_acc=0.8918 | val_loss=0.1425 val_acc=0.9597 | lr=5.05e-05                     
00:42:46 | INFO    | Epoch 2/2 | train_loss=0.1306 train_acc=0.9645 | val_loss=0.1160 val_acc=0.9682 | lr=1.00e-06  )
