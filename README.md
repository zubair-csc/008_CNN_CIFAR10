# 🧮 CNN Image Classification from Scratch

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) built from scratch using PyTorch for image classification on CIFAR-10 and Fashion-MNIST datasets. Features adaptive architecture with batch normalization, dropout regularization, and data augmentation to achieve high classification accuracy on both RGB and grayscale image datasets.

## 🎯 Objectives

* Build CNN architecture from scratch with modern deep learning practices
* Implement adaptive pooling for handling different input image dimensions
* Apply data augmentation and regularization techniques for improved performance
* Achieve 85-90% accuracy on CIFAR-10 and 90-93% on Fashion-MNIST
* Provide comprehensive training visualization and per-class evaluation metrics
* Create production-ready code with proper error handling and GPU optimization

## 📊 Dataset Information

**Datasets**: CIFAR-10 and Fashion-MNIST (automatically downloaded via torchvision)

* **CIFAR-10**:
  * **Size**: 60,000 images (50,000 training, 10,000 test)
  * **Dimensions**: 32×32×3 (RGB)
  * **Classes**: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
* **Fashion-MNIST**:
  * **Size**: 70,000 images (60,000 training, 10,000 test) 
  * **Dimensions**: 28×28×1 (Grayscale)
  * **Classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**Target Output**: Multi-class classification (10 classes each)
**Techniques**: CNN with adaptive pooling, batch normalization, dropout

## 🔧 Technical Implementation

### 📌 CNN Architecture
* **4 Convolutional layers** with increasing filters (32→64→128→256)
* **Batch Normalization** after each conv layer for stable training
* **Adaptive Average Pooling** for handling different input dimensions
* **3 Fully Connected layers** with dropout regularization

### 🧹 Data Preprocessing
* **CIFAR-10**:
  * Random crops with padding and horizontal flips
  * Normalization: mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
* **Fashion-MNIST**:
  * Random rotation (10°) and horizontal flips
  * Normalization: mean=(0.2860), std=(0.3530)

### ⚙️ Training Configuration
* **Optimizer**: Adam with learning rate 0.001 and weight decay 1e-4
* **Loss Function**: CrossEntropyLoss for multi-class classification
* **Scheduler**: StepLR with step_size=20, gamma=0.1
* **Regularization**: Dropout (25% conv, 50% FC) and batch normalization

### 📏 Evaluation Metrics
* **Overall Accuracy**: Percentage of correctly classified images
* **Per-class Accuracy**: Individual performance for each category
* **Training Curves**: Loss and accuracy progression over epochs

## 📊 Visualizations

* Training loss and accuracy curves over epochs
* Sample prediction comparisons with ground truth labels
* Real-time progress tracking with tqdm progress bars

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* CUDA-compatible GPU (optional but recommended)

### Installation

Clone the repository:
```bash
git clone https://github.com/zubair-csc/008_CNN_CIFAR10.git
cd 008_CNN_CIFAR10
```

Install required libraries:
```bash
pip install torch torchvision matplotlib numpy tqdm
```

### Running the Code

Execute the main script:
```python
python cnn_classification.py
```

Or use custom parameters:
```python
# Train on specific dataset
model, accuracy = main('cifar10', epochs=30, batch_size=128, learning_rate=0.001)

# Quick test on Fashion-MNIST
model, accuracy = main('fashionmnist', epochs=10, batch_size=64)
```

## 📈 Results

* **CIFAR-10**: Achieves ~87% test accuracy in 25 epochs (~5 minutes training)
* **Fashion-MNIST**: Achieves ~92% test accuracy in 15 epochs (~3 minutes training)
* **Model Size**: ~1M parameters, 4MB saved file
* **GPU Memory**: <1GB VRAM usage with batch_size=128

## 🙌 Acknowledgments

* PyTorch team for the deep learning framework
* torchvision for dataset access and preprocessing utilities
* CIFAR-10 and Fashion-MNIST dataset creators for benchmark datasets
