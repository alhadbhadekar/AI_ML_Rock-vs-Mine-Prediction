# Rock vs Mine Prediction

## Overview

This project utilizes machine learning techniques to classify sonar signals as either "Rock" or "Mine." The model is built using logistic regression, and it aims to predict the type of object based on sonar readings provided in a dataset.

## Table of Contents

- [Dependencies](#dependencies)
- [Data Collection and Processing](#data-collection-and-processing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Making Predictions](#making-predictions)
- [Usage](#usage)

## Dependencies

To run this project, ensure you have the following Python libraries installed:

- `numpy`: For numerical operations.
- `pandas`: For data manipulation and analysis.
- `scikit-learn`: For machine learning algorithms and metrics.

You can install these packages using pip:

```bash
pip install numpy pandas scikit-learn
```

## Data Collection and Processing

The dataset used in this project is loaded from a CSV file named `sonar data.csv` into a pandas DataFrame. This dataset consists of sonar readings, each containing features that help distinguish between rocks and mines.

### Key Steps:

1. **Load the Dataset:**
   The dataset is read into a pandas DataFrame.
   ```python
   sonar_data = pd.read_csv('/content/sonar data.csv', header=None)
   ```

2. **Explore the Data:**
   Display the first few rows and get basic statistics to understand the dataset.
   ```python
   sonar_data.head()
   sonar_data.describe()
   ```

3. **Label Distribution:**
   The target variable contains two labels:
   - `M` for Mine
   - `R` for Rock

   The distribution of these labels is checked for balance.
   ```python
   sonar_data[60].value_counts()
   ```

4. **Separating Features and Labels:**
   The feature set (X) is separated from the labels (Y).
   ```python
   X = sonar_data.drop(columns=60, axis=1)
   Y = sonar_data[60]
   ```

5. **Split the Data:**
   The dataset is divided into training and test sets, with 10% of the data reserved for testing.
   ```python
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
   ```

## Model Training

A logistic regression model is employed for classification. The model is trained using the training dataset.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

## Model Evaluation

After training, the model's accuracy is evaluated on both the training and test datasets.

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

Both accuracies are printed to provide insights into model performance.

## Making Predictions

The model can make predictions based on new sonar readings. An example input is provided in the form of a tuple of feature values:

```python
input_data = (0.0307, 0.0523, 0.0653, ...)
```

The input is converted into a NumPy array and reshaped for prediction:

```python
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
```

The model predicts the label, and the result is displayed:

```python
prediction = model.predict(input_data_reshaped)
```

If the predicted label is 'R', it indicates a Rock; otherwise, it indicates a Mine.

## Usage

1. Ensure the dataset `sonar data.csv` is in the specified path.
2. Run the script to train the model and evaluate its performance.
3. Modify the `input_data` tuple with new sonar readings to classify additional signals.

This project provides a foundational understanding of machine learning applications in sonar data classification and can be extended or modified for further experiments or enhancements.
