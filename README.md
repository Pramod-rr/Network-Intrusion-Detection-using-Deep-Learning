# Network-Intrusion-Detection-using-Deep-Learning

## Overview

This project implements a deep learning model for network intrusion detection using the UNSW-NB15 dataset. The model is designed to classify network traffic as normal or malicious based on various network features.

The approach combines numerical feature processing with an embedding layer for categorical protocol data, enabling the model to capture both structured and semantic relationships in the data.

## Dataset

The dataset used in this project is the UNSW-NB15 dataset, which contains modern network traffic data with labeled attack and normal activities.

The dataset is automatically downloaded from Kaggle using the opendatasets library.

## Features

### Automatic dataset download from Kaggle

### Data preprocessing:

One-hot encoding for categorical variables (service, state)

Standardization of numerical features

Label encoding for protocol (proto)

### Hybrid model architecture:

Embedding layer for protocol feature

Dense neural network for classification

### Performance metrics:
Accuracy

AUC

Precision

Recall

## Installation

Clone the repository:

git clone https://github.com/your-username/network-intrusion-detection.git
cd network-intrusion-detection

Install dependencies:

pip install tensorflow pandas numpy scikit-learn opendatasets pyarrow
## Usage

Run the training script:

python security.py

The script will:

Download the dataset from Kaggle

Preprocess the data

Train the deep learning model

Output training metrics

## Model Architecture

The model consists of two inputs:

Protocol input (proto): Passed through an embedding layer

Numerical features: Standardized and passed directly to the model

These inputs are concatenated and fed through multiple dense layers with dropout regularization.

## Architecture Summary

Embedding layer (protocol feature)

Fully connected layers with ReLU activation

Dropout layers for regularization

Sigmoid output layer for binary classification

## Training Details

Loss function: Binary Crossentropy

Optimizer: Adam

Epochs: 20

Batch size: 64

Validation split: 20%

## Evaluation Metrics

The model is evaluated using:

Accuracy

AUC (Area Under Curve)

Precision

Recall

These metrics provide a comprehensive view of classification performance, especially for imbalanced datasets.

## Requirements

Python 3.8+

TensorFlow

Pandas

NumPy

Scikit-learn

OpenDatasets

PyArrow

## Notes

Ensure you have Kaggle API credentials configured to download the dataset.

The embedding input dimension is set to 133 based on the dataset's protocol values.

The model uses stratified splitting to preserve class distribution.

## Future Improvements

Hyperparameter tuning

Model evaluation on test dataset

Use of advanced architectures (e.g., LSTM, Transformer)

Model deployment as an API

Integration with real-time network monitoring systems

## License

This project is open-source and available under the MIT License.
