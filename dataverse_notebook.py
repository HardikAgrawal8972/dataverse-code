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
submission_df = pd.read_csv("submission.csv")
ids = submission_df['sha256']
X_submission = submission_df.iloc[:, 1:].values

X_submission = imputer.transform(X_submission)
X_submission = sc.transform(X_submission)

predictions = best_model.predict(X_submission)

sample_submission = pd.DataFrame({
    'sha256': ids,
    'label': predictions
})

sample_submission.to_csv("ID.csv", index=False)
print("\n💾 Submission file 'ID.csv' saved using best model:", best_model_name)

# ==========================================
# 9️⃣ Optional: Visualize Accuracy Comparison
# ==========================================
plt.figure(figsize=(7, 5))
plt.bar(results.keys(), [v*100 for v in results.values()], color=['skyblue', 'lightgreen', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=15)
plt.show()
