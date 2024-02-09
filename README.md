# Wind Speed Prediction using Neural Networks

## Overview
This project implements a neural network model in Python for predicting wind speeds based on historical data. The model utilizes Long Short-Term Memory (LSTM) cells for sequence prediction.

## Dataset
The wind speed dataset is sourced from the file `wind_dataset.csv` located in the `data` directory. It contains historical wind speed measurements along with corresponding dates.

## Preprocessing
- The dataset is preprocessed to extract features and targets for training, validation, and testing.
- Features are generated based on the dates, converting them into sinusoidal representations to capture seasonal patterns.
- Data is split into training, validation, and test sets.

## Model Architecture
- The neural network model consists of two LSTM layers followed by a linear layer for regression.
- The input to the model is a sequence of features representing dates.
- The output of the model is the predicted wind speed.

## Training
- The model is trained using Mean Absolute Error (L1 Loss) as the loss function.
- Stochastic Gradient Descent (SGD) is used as the optimizer.
- Training stops when the validation loss starts increasing, indicating overfitting.

## Evaluation
- The trained model is evaluated on the validation and test sets.
- Performance metrics such as loss values are computed for both validation and test sets.

## Visualization
- Predictions made by the model are visualized along with actual wind speed values.
- The visualization is saved as `NNpredictions.png` in the `diagrams` directory.

## Dependencies
- PyTorch
- NumPy
- Matplotlib

## Usage
1. Clone this repository to your local machine.
2. Ensure the dataset file `wind_dataset.csv` is located in the `data` directory.
3. Run the Python script to train the model and make predictions.
4. Check the generated visualization for predicted wind speed values.

## Acknowledgments
- This project was completed as part of my bachelor's thesis.
- **Citation Request**: [fedesoriano. (April 2022). Wind Speed Prediction Dataset. Retrieved 2023-04-15 from https://www.kaggle.com/datasets/fedesoriano/wind-speed-prediction-dataset]

