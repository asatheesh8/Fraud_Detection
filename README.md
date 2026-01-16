# Fraud Detection

## Introduction
This project aims to analyse credit card transactions to detect fraudulent transactions using Machine Learning. The pipeline includes data visualisation, model training, and evaluation using **XGBoost Classifier**.

## Prerequisites
- Docker installed on your machine.

## Setup & Usage

### Cloning the Repository:
```
git clone https://github.com/asatheesh8/Fraud_Detection.git
```
### Change to the project directory
```
cd Fraud_Detection
```
## Running the Application:
- **Build Docker Image**
```
docker build -t fraud_detection .
```
- **Run the Docker Image**
```
docker run fraud_detection
```
- **In order to see the result plot and text file use the command below**
```
docker run --rm -v "${PWD}/results:/fraud_detection/results" fraud_detection
```

## Dataset Information
- **Name:** Fraud Detection Dataset 
- **Source:** [Kaggle - "mlg-ulb/creditcardfraud"](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **File Used:** `creditcard.csv`  
- **Location:** `data/`  
- **Description:**
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have **492 frauds** out of **284,807 transactions**. The dataset is highly unbalanced, the positive class (frauds) account for **0.172%** of all transactions.

It contains only numerical input variables which are the result of a **PCA transformation**. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are **'Time'** and **'Amount'**. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature **'Class'** is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## What Happens When You Run the Code?
- The script **loads the dataset** from the `data/` folder. 
(! due to high file size, creditcard.csv was not uploaded to GitHub. Kindly download the data from the [Kaggle source](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud))
- Creates **plots to visualize patterns and trends** in the fraudulent cases and are displayed in the console.
- The **data** is then prepared for the ML model by being **split** into **training and test sets**.
- An **XGBClassifier model is trained** to classify transactions to determine whether it is fraudulent or non-fraudulent.
- The model's performance is **evaluated** using **F1 score**, **ROC_AUC** and **PR_AUC**.
- Results are displayed in the **console** and saved results folder as a text file.
