# ML Model Training and Evaluation Lab
 
This project demonstrates how to build and evaluate a beginner machine learning classification model using Python and scikit-learn.
 
## Project Goal
 
The goal is to train a model that predicts whether a customer will purchase a product based on simple customer behavior and profile features.
 
## Dataset
 
The dataset is located at:
 
```text
data/customer_purchase_data.csv
The dataset includes:

age

income

website_visits

time_on_site

previous_purchase

clicked_ad

purchased

The target column is:

purchased
Where:

0 = customer did not purchase
1 = customer purchased

Project Structure
ml-model-evaluation-lab/
│
├── data/
│   └── customer_purchase_data.csv
│
├── models/
│   └── customer_purchase_model.pkl
│
├── outputs/
│   ├── model_metrics.txt
│   └── figures/
│       ├── confusion_matrix.png
│       └── feature_importance.png
│
├── train_model.py
├── README.md
├── requirements.txt
└── .gitignore
ML Workflow
The project follows this workflow:

Load the CSV dataset

Inspect the data

Separate features and target

Split data into training and testing sets

Train a Logistic Regression model

Generate predictions

Evaluate using:

Accuracy

Precision

Recall

F1-score

Classification report

Confusion matrix

Visualize results

Save the trained model

How to Install
Install dependencies:

pip install -r requirements.txt
How to Run
Run the training script:

python train_model.py
Outputs
The script generates:

outputs/model_metrics.txt
outputs/figures/confusion_matrix.png
outputs/figures/feature_importance.png
models/customer_purchase_model.pkl
Example Metrics
The model outputs metrics such as:

Accuracy

Precision

Recall

F1-score

These metrics help evaluate classification performance.

Visualizations
The project includes:

Confusion matrix chart

Feature importance chart

Skills Practiced
Loading CSV data with Pandas

Preparing features and target labels

Splitting train/test data

Training a classification model

Evaluating model performance

Visualizing model results

Saving metrics and trained models

Organizing an ML project for GitHub

Future Improvements
Use a larger real-world dataset

Try different models

Add cross-validation

Add hyperparameter tuning

Add feature scaling

Add model comparison

Build a FastAPI prediction endpoint

Deploy the model as a web app

