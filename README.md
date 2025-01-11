# **Human Disease Prediction based on Symptoms using Machine Learning**

## Project Overview

The **Disease Prediction Based on Symptoms** project leverages multiple machine learning algorithms to predict the most likely disease based on user-provided symptoms. The system uses five different classifiers: **Random Forest**, **Support Vector Machine (SVM)**, **Naive Bayes**, **Decision Tree**, and **K-Nearest Neighbors (KNN)** to independently predict a disease. The final prediction is determined by a **majority-vote mechanism**, where the most frequent disease predicted by the models is selected as the final output.

## Features

- **Symptom Input**: Users can input their symptoms via a user-friendly interface.
- **Multiple Model Predictions**: The disease prediction is made by five machine learning models:
  - Random Forest
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- **Majority-Vote Approach**: The system uses a majority-vote approach to select the final disease prediction based on the output from the five classifiers.
- **Prediction Accuracy**: The system aims to improve prediction accuracy by combining the results of multiple models.

## Dataset

The system relies on publicly available medical datasets to train the machine learning models. The dataset contains diseases labeled alongside their associated symptoms. An example of the dataset format:

| Disease      | Symptom 1  | Symptom 2  | Symptom 3  | Symptom 4 | ... |
|--------------|------------|------------|------------|-----------|-----|
| Flu          | Fever      | Cough      | Fatigue    | Chills    | ... |
| Malaria      | Fever      | Sweating   | Chills     | Fatigue   | ... |
| COVID-19     | Fever      | Cough      | Shortness of breath | Fatigue   | ... |
| ...          | ...        | ...        | ...        | ...       | ... |

### Data Preprocessing

- **Handling Missing Values**: Missing data is handled by imputation or row removal.
- **Encoding Symptoms**: Symptoms are converted into numerical values (binary encoding).
- **Feature Scaling**: Applied to standardize data for models like KNN and SVM.
- **Data Splitting**: Data is split into training (80%) and testing (20%) sets.

## Methodology

![Train Data](https://github.com/user-attachments/assets/34505955-645b-49f0-87cf-10b3377c6067)
<div align="center">
<strong><h1>Human Disease Prediction based on Symptoms</h1><strong>
</div>


### 1. Data Preprocessing

- **Cleaning and Transformation**: Handling missing data, converting categorical symptoms to numerical data, and scaling features for model compatibility.

### 2. Machine Learning Models

Each model is trained independently using the preprocessed dataset:

- **Random Forest**: An ensemble method that builds multiple decision trees and combines their results for prediction.
- **Support Vector Machine (SVM)**: A classifier that finds the optimal hyperplane to separate different diseases in the feature space.
- **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem.
- **Decision Tree**: A classifier that splits data at each node based on feature values.
- **K-Nearest Neighbors (KNN)**: A non-parametric classifier that predicts based on the majority vote of the k-nearest data points.

### 3. Majority-Vote Prediction

- Each model provides a disease prediction based on the user-input symptoms.
- The final disease prediction is selected based on the disease predicted by the majority of models (i.e., the disease predicted by most classifiers).

### 4. Model Evaluation

- **Accuracy**: Evaluated on the test dataset.
- **Precision, Recall, F1-Score**: Used for classification performance evaluation.
- **Confusion Matrix**: To analyze true positives, false positives, true negatives, and false negatives.

## System Architecture

- **User Interface (UI)**: Users enter symptoms via a form (e.g., checkbox selection or text input).
- **Backend Processing**: The backend processes the user input and passes it through all five trained models.
- **Majority-Vote Mechanism**: The predictions from each model are compared to select the most frequent disease.
- **Result Display**: The system displays the predicted disease to the user.

## Requirements

- **Python 3.x**
- Libraries: 
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib` (for visualizations)
  - `seaborn` (for visualizations)
  - `joblib` (for model serialization)

## Installation

To run this project on your local machine:

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/disease-prediction-symptoms.git
2. Navigate to the project directory:
   ```bash
   cd disease-prediction-symptoms
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

Once the system is up and running, you can input your symptoms (fever, cough, fatigue, etc.), and the model will predict the most likely disease based on the majority-vote prediction approach. The final result will display the disease that has been predicted by most models.

## Future Enhancements

- **Continuous Learning:** Implement a feedback loop where the models are retrained with new data for improved predictions.
- **Mobile Application:** Create a mobile app for easier access to the disease prediction system.
- **Incorporate User History:** Consider incorporating user-specific data (e.g., medical history, age, etc.) to provide more personalized predictions.

## Contributing

Feel free to fork this repository, open issues, and submit pull requests to improve the project. Contributions are welcome!

## Contact Information

- **Author: Subho Hazra**
- **Email: subho.hazra2003@gmail.com**
- **GitHub: https://github.com/SubhoHazra07**
