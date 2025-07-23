Titanic Survival Prediction

This project demonstrates a complete machine learning pipeline for predicting survival on the Titanic, using a classic dataset. It covers data loading, preprocessing (handling missing values, encoding categorical features), model training (Random Forest Classifier), and evaluation.

This project is suitable for showcasing fundamental machine learning skills on your GitHub profile.

Table of Contents

•	Project Overview

•	Dataset

•	Features Used

•	Installation

•	Usage

•	Results

•	Dependencies

•	Contributing

•	License

Project Overview

The goal of this project is to predict whether a passenger survived the Titanic disaster based on various features such as their age, gender, passenger class, and more. A Random Forest Classifier is used for this binary classification task.

The pipeline includes:

1.	Data Loading: Reading the train.csv dataset.

2.	Data Preprocessing:

o	Handling missing values (median imputation for 'Age', mode imputation for 'Embarked').

o	One-hot encoding for categorical features ('Sex', 'Embarked').

o	Dropping irrelevant columns ('PassengerId', 'Name', 'Ticket', 'Cabin').

3.	Model Training: Training a RandomForestClassifier on the preprocessed data.

4.	Model Evaluation: Assessing the model's performance using accuracy score and a classification report on a held-out test set.

Dataset

The dataset used is the famous Titanic - Machine Learning from Disaster dataset from Kaggle.

You need to download the train.csv file yourself.

1.	Go to the Kaggle competition page: Titanic - Machine Learning from Disaster

2.	Download the train.csv file.

3.	Create a folder named data in the root directory of this project.

4.	Place the downloaded train.csv file inside the data folder.

Your project structure should look like this:

titanic-survival-prediction/

├── data/

│   └── train.csv  <-- Place the downloaded file here

├── titanic_survival_prediction.py

├── README.md

└── requirements.txt



Features Used

The following features from the dataset are used for prediction after preprocessing:

•	Pclass: Passenger Class (1st, 2nd, 3rd)

•	Age: Age in years

•	SibSp: Number of siblings/spouses aboard the Titanic

•	Parch: Number of parents/children aboard the Titanic

•	Fare: Passenger fare

•	Sex_male: Binary indicator (1 if male, 0 if female)

•	Embarked_Q: Binary indicator (1 if embarked at Queenstown, 0 otherwise)

•	Embarked_S: Binary indicator (1 if embarked at Southampton, 0 otherwise)

The target variable is Survived (0 = No, 1 = Yes).

Installation

1.	Clone the repository (or create the files manually as described above).

2.	git clone <your-repo-url>

3.	cd titanic-survival-prediction



4.	Download the dataset as described in the Dataset section.

5.	Create a virtual environment (recommended):

6.	python -m venv venv

7.	# On Windows:

8.	.\venv\Scripts\activate

9.	# On macOS/Linux:

10.	source venv/bin/activate



11.	Install the required libraries:

12.	pip install -r requirements.txt



Usage

To run the prediction script:

python titanic_survival_prediction.py



The script will:

•	Load the train.csv dataset.

•	Preprocess the data.

•	Split the data into training and testing sets (80% train, 20% test).

•	Train a Random Forest Classifier.

•	Evaluate the model's performance on the test set and print the accuracy and classification report to the console.

Results

The script will output the model's accuracy and a detailed classification report, which includes precision, recall, and f1-score for both 'Survived' (1) and 'Not Survived' (0) classes.

An example of the output you might see (actual values may vary slightly based on data splits and model training):

...

Model Accuracy: 0.8212



Classification Report:

              precision    recall  f1-score   support



           0       0.84      0.88      0.86       110

           1       0.79      0.73      0.76        69



    accuracy                           0.82       179

   macro avg       0.82      0.81      0.81       179

weighted avg       0.82      0.82      0.82       179



--- Program Finished Successfully ---



Dependencies

The project relies on the following Python libraries:

•	pandas

•	scikit-learn

•	numpy

These are listed in requirements.txt.

Contributing

Feel free to fork this repository, open issues, or submit pull requests to improve the code or add new features (e.g., hyperparameter tuning, different models, more advanced feature engineering).

License

This project is open-sourced under the MIT License. See the LICENSE file (if you choose to add one) for more details.



