import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('main.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.impute import SimpleImputer

# Create imputer to fill NaN values with the mean of each column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer on the entire feature matrix X
imputer.fit(X)

# Apply transformation (replace NaN with column means)
X = imputer.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training logistic regression model on the training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#making confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


submission_df = pd.read_csv("submission.csv")
ids = submission_df['sha256']
X_submission = submission_df.iloc[:,1:].values
predictions = classifier.predict(X_submission)

sample_submission = pd.DataFrame({
    'sha256': ids,
    'label': predictions
})

# Save to CSV
sample_submission.to_csv("ID.csv", index=False)