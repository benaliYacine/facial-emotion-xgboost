import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")

# Set CUDA device for GPU acceleration
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the NVIDIA GPU

# Load the dataset
df = pd.read_csv("facial_features22.csv")

# Display info about the dataset
print(f"Dataset shape: {df.shape}")
print(f"Emotions in dataset: {df['emotion'].unique()}")
print(df["emotion"].value_counts())

# Keep only the specified features
selected_features = [
    "vertical_mouth_openness",
    "inner_outer_lip_dist",
    "mouth_nose_dist",
    "mouth_aspect_ratio",
    "left_eye_openness",
    "eye_height",
    "forehead_height",
    "eye_aspect_ratio",
    "eyebrow_arch",
    "eye_width",
    "mouth_area",
    "top_lip_curvature",
    "right_eye_openness",
    "left_brow_slope",
    "mouth_corner_angle",
]

# Make sure all selected features are in the dataset
available_features = [feat for feat in selected_features if feat in df.columns]
missing_features = [feat for feat in selected_features if feat not in df.columns]

if missing_features:
    print(f"Warning: The following features are not in the dataset: {missing_features}")

print(f"\nUsing {len(available_features)} features for training: {available_features}")

# Prepare features and target
X = df[available_features]
y = df["emotion"]

# Verify the training data
print("\nTraining data preview:")
print(X.head())
print(f"\nTraining data shape: {X.shape}")

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(
    f"Encoded emotions: {list(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}"
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create DMatrix for XGBoost (with GPU acceleration)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Initial XGBoost parameters with GPU acceleration
params = {
    "objective": "multi:softmax",
    "num_class": len(label_encoder.classes_),
    "tree_method": "gpu_hist",  # Use GPU acceleration
    "gpu_id": 0,
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "seed": 42,
}

# Train initial model
num_rounds = 100
watchlist = [(dtrain, "train"), (dtest, "test")]
model = xgb.train(
    params, dtrain, num_rounds, watchlist, early_stopping_rounds=20, verbose_eval=10
)


# Perform hyperparameter tuning
def objective(hyperparams):
    results = xgb.cv(
        hyperparams,
        dtrain,
        num_boost_round=500,
        nfold=5,
        metrics="mlogloss",
        early_stopping_rounds=50,
        stratified=True,
        seed=42,
    )
    return results["test-mlogloss-mean"].iloc[-1]


# Grid search for hyperparameter tuning
param_grid = {
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "eta": [0.01, 0.1, 0.3],
}

# Create XGBClassifier (with GPU acceleration)
xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    tree_method="gpu_hist",  # Use GPU acceleration
    gpu_id=0,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
)

# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    verbose=1,
    n_jobs=1,  # Using GPU, so keep n_jobs at 1
)

grid_search.fit(X_train_scaled, y_train)

# Best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Train final model with best parameters
best_params = grid_search.best_params_.copy()
best_params.update(
    {
        "objective": "multi:softmax",
        "num_class": len(label_encoder.classes_),
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "eval_metric": "mlogloss",
    }
)

final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=200,
    evals=watchlist,
    early_stopping_rounds=20,
    verbose_eval=20,
)

# Make predictions
y_pred = final_model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print(f"Final model accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
cm_percentage = (
    cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
)  # Convert to percentage
sns.heatmap(
    cm_percentage,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted Label2")
plt.ylabel("True Label2")
plt.title("Confusion Matrix2 (%)")
plt.tight_layout()
plt.savefig("confusion_matrix2_percent.png")
plt.show()

# Feature importance
plt.figure(figsize=(12, 6))
xgb.plot_importance(final_model, max_num_features=12)
plt.title("Feature Importance2")
plt.tight_layout()
plt.savefig("feature_importance2.png")
plt.show()

# Save the model
final_model.save_model("facial_expression_model2.json")


# Function to predict emotion from new facial features
def predict_emotion(features, scaler, model, label_encoder):
    """
    Predict emotion from facial features

    Parameters:
    features (list/array): List of facial features in the same order as training data
    scaler (StandardScaler): Fitted scaler object
    model (xgb.Booster): Trained XGBoost model
    label_encoder (LabelEncoder): Fitted label encoder

    Returns:
    str: Predicted emotion
    """
    features_scaled = scaler.transform([features])
    dfeatures = xgb.DMatrix(features_scaled)
    prediction = model.predict(dfeatures)
    emotion = label_encoder.inverse_transform([int(prediction[0])])[0]
    return emotion


# Example usage of prediction function
sample_features = X_test.iloc[0].values
predicted_emotion = predict_emotion(sample_features, scaler, final_model, label_encoder)
actual_emotion = label_encoder.inverse_transform([y_test[0]])[0]
print(f"\nExample prediction:")
print(f"Actual emotion: {actual_emotion}")
print(f"Predicted emotion: {predicted_emotion}")

# Save components for later use
import joblib

joblib.dump(scaler, "scaler2.pkl")
joblib.dump(label_encoder, "label_encoder2.pkl")

print("\nModel and components saved successfully.")
