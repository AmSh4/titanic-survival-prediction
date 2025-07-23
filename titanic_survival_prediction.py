# titanic_survival_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import os

# --- Configuration ---
# Define the path where the dataset is expected to be found
DATA_DIR = 'data'
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv' # Note: For this project, we'll focus on predicting on the training set's test split.
                      # If you want to make predictions for the actual Kaggle test.csv,
                      # you'd apply the same preprocessing and then predict.

def load_data(data_dir, train_file):
    """
    Loads the training dataset from the specified directory.

    Args:
        data_dir (str): The directory where the dataset is located.
        train_file (str): The name of the training data CSV file.

    Returns:
        pd.DataFrame: The loaded training DataFrame.
    """
    file_path = os.path.join(data_dir, train_file)
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        print("Please download 'train.csv' from Kaggle (Titanic - Machine Learning from Disaster) and place it in a 'data' folder.")
        print("URL: https://www.kaggle.com/c/titanic/data")
        return None
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocesses the Titanic dataset.
    This includes handling missing values, encoding categorical features,
    and dropping irrelevant columns.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
        list: List of features used for training.
    """
    print("Starting data preprocessing...")

    # Drop columns that are not useful for prediction or have too many unique values
    # 'PassengerId' is just an identifier
    # 'Name' is unique for each passenger
    # 'Ticket' has too many unique values and complex patterns
    # 'Cabin' has a very high percentage of missing values and complex patterns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    # Handle missing 'Age' values using median imputation
    # Use SimpleImputer for robustness
    imputer_age = SimpleImputer(strategy='median')
    df['Age'] = imputer_age.fit_transform(df[['Age']])

    # Handle missing 'Embarked' values using mode imputation (most frequent)
    imputer_embarked = SimpleImputer(strategy='most_frequent')
    df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']])

    # Convert 'Sex' and 'Embarked' into numerical representations using one-hot encoding
    # This creates new columns for each category (e.g., Sex_male, Sex_female)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True) # drop_first avoids multicollinearity

    # Define features (X) and target (y)
    # The target variable is 'Survived'
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Ensure all feature columns are numeric
    # Convert boolean columns created by get_dummies to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)

    # Get the list of feature names after preprocessing
    features = X.columns.tolist()

    print("Data preprocessing complete.")
    return X, y, features

def train_model(X_train, y_train):
    """
    Trains a RandomForestClassifier model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        sklearn.ensemble.RandomForestClassifier: The trained model.
    """
    print("Training the RandomForestClassifier model...")
    # Initialize RandomForestClassifier
    # n_estimators: number of trees in the forest
    # random_state: for reproducibility
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): The trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    """
    print("Evaluating the model...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("Model evaluation complete.")

def main():
    """
    Main function to run the Titanic survival prediction pipeline.
    """
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Load Data
    df = load_data(DATA_DIR, TRAIN_FILE)
    if df is None:
        return # Exit if data loading failed

    print("\nOriginal Data Head:")
    print(df.head())
    print("\nOriginal Data Info:")
    df.info()
    print("\nMissing values before preprocessing:")
    print(df.isnull().sum())

    # 2. Preprocess Data
    X, y, features = preprocess_data(df.copy()) # Use a copy to avoid modifying original df

    print("\nPreprocessed Features (X) Head:")
    print(X.head())
    print("\nPreprocessed Target (y) Head:")
    print(y.head())
    print(f"\nFeatures used: {features}")

    # 3. Split Data into Training and Testing Sets
    # test_size=0.2 means 20% of data will be used for testing
    # stratify=y ensures that the proportion of target variable (Survived) is same in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")

    # 4. Train Model
    model = train_model(X_train, y_train)

    # 5. Evaluate Model
    evaluate_model(model, X_test, y_test)

    print("\n--- Program Finished Successfully ---")
    print("Next steps:")
    print("1. Ensure 'train.csv' is in the 'data' folder.")
    print("2. Run this script: python titanic_survival_prediction.py")
    print("3. Explore the output and model performance.")

if __name__ == "__main__":
    main()
