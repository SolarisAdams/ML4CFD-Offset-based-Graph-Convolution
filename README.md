# ML4CFD-Offset-based-Graph-Convolution
Code for Offset-based Graph Convolution for Mesh Graph in ML4CFD Competetion

## Introduction

This repository contains code for Offset-based Graph Neural Network model designed to predict fluid dynamics properties on airfoil meshes using graph neural networks.


## Model Overview

Our submission implements an **Offset-based Graph Neural Network (OffsetGNN)** designed specifically for airfoil RANS simulations. The model addresses the challenges of irregular mesh structures and spatial coordinate sensitivity through:

1. **Offset-driven kernel weight generation**:
   - Computes geometric offset vectors between nodes to capture spatial relationships
   - Uses lightweight MLPs to transform offsets into dynamic scaling weights

2. **Geometry-aware aggregation**:
   - Employs inverse-distance normalized attention weights
   - Combines self-feature updates with neighborhood aggregation

The architecture features three dedicated GNN models for different physical properties (velocity, pressure, and turbulent viscosity), each with specialized hidden dimensions and decoding layers. During training, the model utilizes boundary-aware sampling, noise injection, and a tailored loss function to balance prediction errors and incorporate surface-specific regularization.

## Environment Setup

To get started, first install the required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The required packages are:

- tqdm
- numpy
- torch>=2.0
- airfrans
- matplotlib
- lips
- dgl>=2.0
- vtk==9.3.0
- scikit-learn==1.5.0

## Project Structure

```
AirfoilRANS/
├── README.md           # This documentation file
├── requirements.txt    # Environment dependencies
├── src/
│   ├── confAirfoil.ini # Configuration file for the benchmark
│   └── example.py      # Example usage script
└── submission/
    ├── example.py      # Implementation code
    └── parameters.json # Configuration file required for submission
```

## Usage


### Getting the Dataset

Please download the Airfoil RANS dataset from the [AirfoilsRepository](https://airfrans.readthedocs.io/en/latest/notes/dataset.html#downloading-the-dataset). Extract the dataset to a directory of your choice, and modify the `DIRECTORY_NAME` variable in `src/example.py` to point to the extracted directory.

### Training and Evaluation

Both training and evaluation are performed using the `example.py` script.

To train and evaluate the model, run the following command:

```bash
python src/example.py --epochs 12 --batch_size 128 --lr 1.5e-4
```


## Example Parameters

The hyperparameters used for the submission are specified as default arguments in `submission/OffsetGNN.py`. You can modify this file to adjust training settings.




## Contact

For any questions or issues regarding this project, please contact the repository maintainer.