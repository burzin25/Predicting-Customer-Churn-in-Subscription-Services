
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, log_loss, roc_curve, auc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping


def split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=None):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        X (DataFrame or array-like): The feature matrix.
        y (Series or array-like): The target variable.
        test_size (float or int): The proportion of the dataset to include in the test split.
        val_size (float or int): The proportion of the dataset to include in the validation split.
        random_state (int or None): Random seed for reproducibility.

    Returns:
        Tuple: A tuple containing the following splits: X_train, X_val, X_test, y_train, y_val, y_test.
    """
    print("Splitting dataset into training, validation, and test sets...")
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Dataset split into training set ({len(X_train)} samples) and test set ({len(X_test)} samples)")
    
    # Split the remaining training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size / (1 - test_size),
                                                      random_state=random_state)
    print(f"Training set further split into training set ({len(X_train)} samples) and validation set ({len(X_val)} samples)")
    
    print("Splitting complete!")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_evaluate_neural_network(X_train, y_train, X_test, y_test, epochs=50, batch_size=64, validation_split=0.2, verbose=1):
    """
    Train and evaluate a feedforward neural network for binary classification tasks.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data labels.
    - epochs (int): Number of epochs for training. Default is 50.
    - batch_size (int): Batch size for training. Default is 64.
    - validation_split (float): Fraction of training data to be used as validation data. Default is 0.2.
    - verbose (int): Verbosity mode during training and evaluation. Default is 1.

    Returns:
    None

    The function trains a feedforward neural network with ReLU activation in the hidden layers
    and sigmoid activation in the output layer. It uses binary cross-entropy loss function,
    Adam optimizer, and early stopping with a patience of 5 epochs to prevent overfitting.
    After training, it evaluates the model on the test data and prints performance metrics
    including accuracy, precision, recall, F1 score, confusion matrix, and classification report.
    It also plots the ROC curve and training/validation loss curves to visualize model performance
    and checks for overfitting by comparing validation loss trends.
    """
    
    # Create a Sequential model
    model = Sequential()

    # Add layers to the model
    model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))  # Input layer
    model.add(Dense(units=64, activation='relu'))  # Hidden layer
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification

    # Define early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=verbose)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with early stopping
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, callbacks=[early_stopping], verbose=verbose)

    # Make predictions on the test set
    y_pred_proba = model.predict(X_test)
    y_pred = np.round(y_pred_proba)

    # Calculate log loss
    logloss = log_loss(y_test, y_pred_proba)
    print(f'Log Loss: {logloss:.4f}')

    # Calculate other performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Print performance metrics
    print("Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Plot training and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Check for overfitting
    if np.any(np.diff(history.history['val_loss']) > 0):
        print("The model is overfitting.")
    else:
        print("The model is not overfitting.")
