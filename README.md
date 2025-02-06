# Rock vs Mine Prediction

## Overview

This project implements a machine learning model to classify sonar signals as either "Rock" or "Mine." Using a dataset of sonar readings, the model employs logistic regression to predict the type of object based on input features.

## Table of Contents

- [Dependencies](#dependencies)
- [Data Collection and Processing](#data-collection-and-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Predictive System](#predictive-system)
- [Usage](#usage)

## Dependencies

This code requires the following Python libraries:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `sklearn`: For implementing the machine learning model and evaluation metrics.

To install these libraries, you can use pip:

```bash
pip install numpy pandas scikit-learn
```

## Data Collection and Processing

The dataset is loaded from a CSV file named `sonar data.csv` into a pandas DataFrame. The data contains sonar readings with features that describe each observation.

### Key Steps:

1. **Loading the Dataset:**
   ```python
   sonar_data = pd.read_csv('/content/sonar data.csv', header=None)
   ```

2. **Data Overview:**
   The shape and statistical description of the dataset are displayed.
   ```python
   sonar_data.head()
   sonar_data.describe()
   ```

3. **Label Distribution:**
   The dataset contains two labels:
   - `M` for Mine
   - `R` for Rock

   A count of these labels is provided for understanding the distribution.

4. **Separating Features and Labels:**
   Features (X) and labels (Y) are separated for model training.
   ```python
   X = sonar_data.drop(columns=60, axis=1)
   Y = sonar_data[60]
   ```

5. **Splitting the Data:**
   The dataset is split into training and test sets, with a test size of 10%:
   ```python
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
   ```

## Model Training

A logistic regression model is used for classification. The model is trained using the training data:

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Model Evaluation

The model's accuracy is evaluated using both training and test datasets:

- **Training Accuracy:**
  ```python
  X_train_prediction = model.predict(X_train)
  training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
  ```

- **Test Accuracy:**
  ```python
  X_test_prediction = model.predict(X_test)
  test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
  ```

## Predictive System

The model can make predictions based on new input data. An example input is provided as a tuple of features:

```python
input_data = (0.0307, 0.0523, 0.0653, ...)
```

The input is reshaped and fed into the model for prediction:

```python
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
```

The prediction is displayed to indicate whether the object is a Rock or a Mine.

## Usage

To use the model for predictions:

1. Ensure that the dataset is available as `sonar data.csv`.
2. Run the script to train the model and evaluate its performance.
3. Modify the `input_data` tuple with new sonar readings to classify other signals.

This project serves as an introduction to machine learning with practical applications in sonar data analysis and object classification.#   A I _ M L _ R o c k - v s - M i n e - P r e d i c t i o n 
 
 
