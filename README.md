# Polynomial Graph Degree Classification

## Table of Contents
- [Introduction](#introduction)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction
This project aims to classify polynomial graphs based on their degrees using Convolutional Neural Networks (CNNs). It contains code for generating graphs, as well as for training and evaluating the machine learning models.

## Technologies
### Graph Generation (`graph_generation.py`)
- **Python Libraries**: NumPy, Matplotlib, Pandas, OpenCV
- **Methods**: Graphs are generated using NumPy for mathematical calculations and Matplotlib for plotting.
- **Data Storage**: Graph images and metadata are saved to disk.

### Model Training (`model_training.py`)
- **Python Libraries**: NumPy, Pandas, Matplotlib, TensorFlow, scikit-learn
- **Methods**: The machine learning model uses a Convolutional Neural Network (CNN) implemented in TensorFlow. Data is split into training and testing sets using scikit-learn.
- **Data Loading**: Data is loaded from a pickled dataset.

## Installation

Clone the repository and navigate to the project directory. Install the required packages using pip:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt
```

## Usage

### Graph Generation

Run `graph_generation.py` to generate polynomial graphs.

```bash
python graph_generation.py
```

### Model Training

Run `model_training.py` to train the machine learning model.

```bash
python model_training.py
```

## Results

### Model Performance

#### Training & Validation Loss

![Graph of Training & Validation Loss](/content/loss.png)

#### Training & Validation Accuracy

![Graph of Training & Validation Accuracy](/content/accuracy.png)

#### Test Loss & Accuracy

- Test Loss: 0.20
- Test Accuracy: 0.94

#### Test Predictions

![Some Predictions from the Test Set](/content/predictions.png)