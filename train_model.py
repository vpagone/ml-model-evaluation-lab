import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
 
 
DATA_PATH = "data/customer_purchase_data.csv"
MODEL_PATH = "models/customer_purchase_model.pkl"
METRICS_PATH = "outputs/model_metrics.txt"
CONFUSION_MATRIX_PATH = "outputs/figures/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "outputs/figures/feature_importance.png"
 
 
def load_data(file_path):
    """
    Load the customer purchase dataset.
    """
    print("Step 1: Loading dataset...")
 
    df = pd.read_csv(file_path)
 
    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
 
    return df
 
 
def inspect_data(df):
    """
    Display basic dataset information.
    """
    print("\n--- Dataset Preview ---")
    print(df.head())
 
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
 
    print("\n--- Data Types ---")
    print(df.dtypes)
 
    print("\n--- Target Distribution ---")
    print(df["purchased"].value_counts())
 
 
def prepare_features_and_target(df):
    """
    Separate input features and target label.
    """
    print("\nStep 2: Preparing features and target...")
 
    feature_columns = [
        "age",
        "income",
        "website_visits",
        "time_on_site",
        "previous_purchase",
        "clicked_ad",
    ]
 
    target_column = "purchased"
 
    X = df[feature_columns]
    y = df[target_column]
 
    print("Features and target prepared.")
    print(f"Feature columns: {feature_columns}")
    print(f"Target column: {target_column}")
 
    return X, y, feature_columns
 
 
def split_data(X, y):
    """
    Split data into training and testing sets.
    """
    print("\nStep 3: Splitting data into train and test sets...")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
 
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
 
    return X_train, X_test, y_train, y_test
 
 
def train_model(X_train, y_train):
    """
    Train a Logistic Regression classification model.
    """
    print("\nStep 4: Training model...")
 
    model = LogisticRegression(max_iter=1000)
 
    model.fit(X_train, y_train)
 
    print("Model training completed.")
 
    return model
 
 
def make_predictions(model, X_test):
    """
    Generate predictions on test data.
    """
    print("\nStep 5: Making predictions...")
 
    y_pred = model.predict(X_test)
 
    print("Predictions generated.")
 
    return y_pred
 
 
def evaluate_model(y_test, y_pred):
    """
    Evaluate the classification model using common metrics.
    """
    print("\nStep 6: Evaluating model...")
 
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
 
    report = classification_report(y_test, y_pred)
 
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }
 
    print("\n--- Model Metrics ---")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
 
    print("\n--- Classification Report ---")
    print(report)
 
    return metrics
 
 
def save_metrics(metrics, output_path):
    """
    Save model evaluation metrics to a text file.
    """
    print("\nStep 7: Saving metrics...")
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    with open(output_path, "w") as file:
        file.write("Customer Purchase Model Metrics\n")
        file.write("===============================\n\n")
        file.write(f"Accuracy: {metrics['accuracy']:.2f}\n")
        file.write(f"Precision: {metrics['precision']:.2f}\n")
        file.write(f"Recall: {metrics['recall']:.2f}\n")
        file.write(f"F1 Score: {metrics['f1_score']:.2f}\n\n")
        file.write("Classification Report:\n")
        file.write(metrics["classification_report"])
 
    print(f"Metrics saved to: {output_path}")
 
 
def plot_confusion_matrix(y_test, y_pred, output_path):
    """
    Generate and save a confusion matrix visualization.
    """
    print("\nStep 8: Creating confusion matrix...")
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    cm = confusion_matrix(y_test, y_pred)
 
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Purchased", "Purchased"],
    )
 
    display.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
 
    print(f"Confusion matrix saved to: {output_path}")
 
 
def plot_feature_importance(model, feature_columns, output_path):
    """
    Generate and save a feature importance chart based on model coefficients.
    """
    print("\nStep 9: Creating feature importance chart...")
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    importance = model.coef_[0]
 
    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
        }
    )
 
    importance_df["absolute_importance"] = importance_df["importance"].abs()
 
    importance_df = importance_df.sort_values(
        by="absolute_importance",
        ascending=False,
    )
 
    plt.figure(figsize=(8, 5))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Model Coefficient")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
 
    print(f"Feature importance chart saved to: {output_path}")
 
 
def save_model(model, output_path):
    """
    Save the trained model to a file.
    """
    print("\nStep 10: Saving trained model...")
 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
    joblib.dump(model, output_path)
 
    print(f"Model saved to: {output_path}")
 
 
def main():
    """
    Run the full machine learning training and evaluation workflow.
    """
    print("Starting ML Model Training and Evaluation Lab")
    print("============================================")
 
    df = load_data(DATA_PATH)
 
    inspect_data(df)
 
    X, y, feature_columns = prepare_features_and_target(df)
 
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    model = train_model(X_train, y_train)
 
    y_pred = make_predictions(model, X_test)
 
    metrics = evaluate_model(y_test, y_pred)
 
    save_metrics(metrics, METRICS_PATH)
 
    plot_confusion_matrix(y_test, y_pred, CONFUSION_MATRIX_PATH)
 
    plot_feature_importance(model, feature_columns, FEATURE_IMPORTANCE_PATH)
 
    save_model(model, MODEL_PATH)
 
    print("\nML workflow completed successfully!")
 
 
if __name__ == "__main__":
    main()