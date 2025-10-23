# ==========================================
# 1️⃣ Importing Libraries
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# ==========================================
# 2️⃣ Importing Dataset
# ==========================================
dataset = pd.read_csv('main.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# ==========================================
# 3️⃣ Handling Missing Values
# ==========================================
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

# ==========================================
# 4️⃣ Split and Scale Data
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==========================================
# 5️⃣ Define Models
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    print("⚠️ XGBoost not installed. Run 'pip install xgboost' to include it.")
    xgb_available = False

models = {
    "Logistic Regression": LogisticRegression(random_state=0, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )

# ==========================================
# 6️⃣ Train and Evaluate Each Model
# ==========================================
results = {}

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ {name} Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    results[name] = acc

# ==========================================
# 7️⃣ Compare Models
# ==========================================
print("\n📊 Model Comparison:")
for name, acc in results.items():
    print(f"{name:<20}: {acc*100:.2f}%")

best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} ({results[best_model_name]*100:.2f}% accuracy)")






# ==========================================
# 8️⃣ Predict on Submission Data
# ==========================================

# 🚨 1. UPDATE THIS FILENAME
# This must be the CSV file that has the 52 features for testing.
TEST_FEATURES_FILE = "test.csv" 

# 🚨 2. UPDATE THIS FILENAME
# This is your sample submission file.
SUBMISSION_FILE = "submission.csv"

print(f"--- 🚀 Starting Prediction (Section 8) ---")

try:
    # 3. Load the ACTUAL test data (which has the 52 features)
    test_features_df = pd.read_csv(TEST_FEATURES_FILE)
    print(f"Successfully loaded {TEST_FEATURES_FILE}.")
    print(f"Shape of test_features_df: {test_features_df.shape}")
    print("Columns:", test_features_df.columns.to_list())

    # 4. Load the sample submission file (which has the IDs in the correct order)
    submission_df = pd.read_csv(SUBMISSION_FILE)
    ids = submission_df['sha256']
    print(f"Successfully loaded {SUBMISSION_FILE}. Ready to predict for {len(ids)} IDs.")

except FileNotFoundError as e:
    print(f"\n🚨 FILE NOT FOUND ERROR: {e}")
    print(f"Please check your filenames in Section 8.")
    print(f"TEST_FEATURES_FILE = '{TEST_FEATURES_FILE}'")
    print(f"SUBMISSION_FILE = '{SUBMISSION_FILE}'")
    exit() # Stop the script

# 5. Align the features with the submission IDs using a merge.
# This ensures your rows are in the correct order.
merged_df = pd.merge(submission_df[['sha256']], test_features_df, on='sha256', how='left')
print(f"Shape after merging: {merged_df.shape}")

# 6. Create X_submission from the merged data.
# This slicing (:, 1:) assumes 'sha256' is the first column.
X_submission = merged_df.iloc[:, 1:].values
print(f"Shape of final X_submission array (to be imputed): {X_submission.shape}")

# 7. Check for the error condition BEFORE it happens
if X_submission.shape[1] != 52:
    print(f"\n🚨 SHAPE MISMATCH! 🚨")
    print(f"The imputer expects 52 features, but your X_submission has {X_submission.shape[1]}.")
    print("This almost always means the file '{TEST_FEATURES_FILE}' does not contain the 52 features.")
    print("Please double-check your file and your filenames.")
    exit() # Stop the script

# ==========================================
# The rest of the code is the same
# ==========================================

print("Imputing missing values...")
X_submission = imputer.transform(X_submission)

print("Scaling data...")
X_submission = sc.transform(X_submission)

print(f"Predicting with best model: {best_model_name}...")
predictions = best_model.predict(X_submission)

sample_submission = pd.DataFrame({
    'sha256': ids,
    'label': predictions
})

sample_submission.to_csv("ID.csv", index=False)
print("\n✅💾 Submission file 'ID.csv' saved using best model:", best_model_name)

# ==========================================
# 9️⃣ Optional: Visualize Accuracy Comparison
# ==========================================
plt.figure(figsize=(7, 5))
model_colors = ['skyblue', 'lightgreen', 'orange']
plt.bar(results.keys(), 
        [v*100 for v in results.values()], 
        color=model_colors[:len(results)])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=15)
plt.show()



