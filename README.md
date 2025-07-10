# CountXplain: Interpretable Cell Counting with Prototype-Based Density Map Estimation

This repository contains the official implementation of **CountXplain**, a novel approach for interpretable cell counting that combines prototype-based learning with density map estimation. The method provides both accurate counting predictions and interpretable explanations for the model's decisions.

## Overview

CountXplain addresses the critical need for interpretability in cell counting applications by introducing a prototype-based architecture that:

- **Learns interpretable prototypes** that represent typical cellular patterns
- **Provides density map estimation** for spatial understanding of cell distributions  
- **Offers model explanations** through prototype similarity visualization
- **Maintains high counting accuracy** while ensuring interpretability

## Architecture

The model consists of two main components:

1. **Counting Model (CSRNet)**: A density estimation network based on VGG-16 frontend with dilated convolutions
2. **Prototype Network**: Learns a set of prototypes that capture representative cellular patterns and provides interpretability

### Key Features

- **Prototype-based interpretability**: Visual explanations through learned prototypes
- **Density map estimation**: Spatial understanding of cell distributions

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Lightning
- OpenCV
- NumPy
- Matplotlib
- Weights & Biases (for logging)
- H5PY (for density map storage)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/countxplain.git
cd countxplain
```

2. Install dependencies:
```bash
pip install torch torchvision pytorch-lightning opencv-python numpy matplotlib wandb h5py tqdm scipy pandas
```

### Dataset Preparation

Organize your dataset with the following structure:
```
Dataset/
├── trainval/
│   ├── images/          # Training images (.png)
│   └── densities/       # Ground truth density maps (.h5)
└── test/
    ├── images/          # Test images (.png)
    └── densities/       # Test density maps (.h5)
```

## Training

### 1. Train Base Counting Model

First, train the base counting model (CSRNet):

```bash
python train_counting_model.py --dataset DCC --model_name csrnet --batch_size 2 --lr 0.001
```

### 2. Train Prototype Model

Train the CountXplain model with prototypes:

```bash
python train_push.py --dataset DCC --num_prototypes 20 --fg_coef 1 --diversity_coef 1 --proto_to_feature_coef 1 --batch_size 2 --lr 0.001
```

### Training Parameters

- `--num_prototypes`: Number of prototypes to learn (default: 20)
- `--fg_coef`: Weight for density estimation loss
- `--diversity_coef`: Prototype diversity loss coefficient  
- `--proto_to_feature_coef`: Prototype-to-feature alignment coefficient
- `--batch_size`: Training batch size
- `--lr`: Learning rate


## Citation

If you use this code in your research, please cite:

```bibtex
@article{countxplain2024,
  title={CountXplain: Interpretable Cell Counting with Prototype-Based Density Map Estimation},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License - see the [license.txt](license.txt) file for details.


