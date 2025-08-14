# General Deep Learning Advertisements Project

This project provides a foundation for building deep learning models for advertisement-related tasks such as click-through rate (CTR) prediction or ad image classification. It includes modules for data loading, model definition, training, evaluation, and prediction.

## Dataset

The project expects tabular CSV data for CTR prediction or image folders for ad classification. Tabular datasets should include a target column indicating whether a user clicked an ad. Image datasets should be organized in subdirectories per class.

## Model Architectures

* **MLP**: A configurable Multi-Layer Perceptron for tabular data.
* **CNN**: A convolutional neural network built on popular backbones (ResNet, EfficientNet) for image-based tasks.

## Training and Evaluation

Training scripts allow selection between MLP and CNN models with customizable hyperparameters, optimizer choice, learning rate scheduling, and early stopping. Evaluation scripts compute accuracy, precision, recall, F1-score, and AUC, while saving confusion matrices and other metrics.

## Results

Trained model weights are stored in the `models/` directory, while evaluation metrics and plots reside in `results/`.

## Usage

Detailed usage examples are available in the docstrings of the scripts within `src/`.
