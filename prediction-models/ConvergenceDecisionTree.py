import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data from a CSV file
def load_data(data : str):
    
    df = pd.read_csv(data)

    print("Shape: ", df.shape)
    return df

# Drop unnecessary columns and prepare features X and target y for prediction
def select_columns(df: pd.DataFrame, target_col: str = "time_to_convergence"):
    
    drop_columns = [
        "run_id",
        "machine_id",
        "run_log_file",
        "train_duration_s",
        "steps_to_convergence"
    ]

    df_clean = df.drop(columns=drop_columns, errors="ignore")

    # Target: coerce to numeric, drop rows where target is missing
    df_clean[target_col] = pd.to_numeric(df_clean[target_col], errors="coerce")
    df_clean[target_col] = df_clean[target_col].fillna(-1)  # keep non-converged as -1

    X = df_clean.drop(columns=[target_col], errors="ignore")
    y = (df_clean[target_col] > 0).astype(int) #binary target

    # Keep NA features by filling missing values with 0 after encoding.
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)

    print("X shape: ", X.shape, "| y shape: ", y.shape)
    print("Target distribution:\n", y.value_counts())

    return X, y

# Split the dataset into training and testing sets
def split_data(X, y, test_size: float = 0.2, seed: int = 42):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    
    print("X_train shape: ", X_train.shape, "| y_train shape: ", y_train.shape)
    print("X_test shape: ", X_test.shape, "| y_test shape: ", y_test.shape)

    return X_train, X_test, y_train, y_test

# Build and train a Decision Tree model
def build_and_train_model(X_train, y_train, max_depth=6, seed: int = 42):

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    model.fit(X_train, y_train)

    print("Classifier trained successfully.")
    return model

# Evaluation for classification
def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    return preds, acc

# Display which features were most important to the model
def feature_importance(model, X_train):

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("Feature Importances:\n", importance.head(10))

    return importance


data_path = Path(__file__).resolve().parents[1] / "training_manager" / "experiments" / "results" / "main.csv"
df = load_data(str(data_path))
X, y = select_columns(df)


# Inspect/save the encoded dataset
encoded = X.copy()
encoded["converged"] = y   # y is already 0/1

out_path = Path(__file__).resolve().parent / "encoded_dataset.csv"
encoded.to_csv(out_path, index=False)

print(encoded.head())
print(encoded.shape)

X_train, X_test, y_train, y_test = split_data(X, y)
model = build_and_train_model(X_train, y_train)
preds, acc = evaluate_model(model, X_test, y_test)
#importance = feature_importance(model, X_train)
