import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("C:/Users/Usuario/Desktop/patients_dataset.csv")
df = df[['Age', 'Gender', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells', 'Background']]  # Select relevant columns
df.head()

# Drop duplicates
df = df.drop_duplicates()

# Handle missing values by imputing with the mean for numerical columns
for col in ['Gender', 'Background']:
    df[col].fillna(df[col].mean(), inplace=True)

# Encode categorical variables using one-hot encoding
categorical_cols = ['Age', 'Lymphocytes T', 'Lymphocytes B', 'Red Cells']
df = pd.get_dummies(df, columns=categorical_cols)

# Define features and target variable
X = df.drop('Background', axis=1)
y = df['Background']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)

# Define values to explore for n_estimators (from 100 to 1000, step 100)
n_values_estimators = range(100, 1001, 100)

# Define values to explore for max_depth
max_depth_values = range(1, 11)

# Create a subplot for visualizing graphs with different accuracy values
fig, axes = plt.subplots(5, 2, figsize=(15, 30))

for i, n_estimators in enumerate(n_values_estimators):
    train_accuracy = []
    test_accuracy = []

    for max_depth in max_depth_values:
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        rfc.fit(X_train, y_train)
        train_pred = rfc.predict(X_train)
        test_pred = rfc.predict(X_test)

        train_accuracy.append(accuracy_score(y_train, train_pred))
        test_accuracy.append(accuracy_score(y_test, test_pred))
        
    # Plot for each n_estimators
    ax = axes[i // 2, i % 2]
    ax.plot(max_depth_values, train_accuracy, label=f'Train Accuracy, n_estimators={n_estimators}', color='purple')
    ax.plot(max_depth_values, test_accuracy, label=f'Test Accuracy, n_estimators={n_estimators}', color='green')

    ax.set_title(f'n_estimators={n_estimators}', fontweight='bold')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    ax.set_xticks(range(1, 11, 1))

# Save the accuracy vs. hyperparameters figure
plt.tight_layout()
plt.savefig('./Images/accuracy_vs_hyperparameters.png')
plt.show()

# Train the Random Forest model with optimized parameters
rf = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)  # Adjusted for dataset size and complexity
rf.fit(X_train, y_train)

# Evaluate accuracy with cross-validation
accuracy = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-validation accuracy: ", accuracy)
print("\nMean accuracy: ", accuracy.mean())
print("\nStandard deviation of accuracy: ", accuracy.std())

# Evaluate the model on the test set
y_predict_rf = rf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_predict_rf)
print("Accuracy on test set: ", accuracy_test)

print("Classification report: ")
print(classification_report(y_test, y_predict_rf))

# Create the confusion matrix
plt.figure(figsize=(8, 5))
sns.heatmap(confusion_matrix(y_test, y_predict_rf), annot=True, fmt='d', cmap='plasma')
plt.title('CONFUSION MATRIX', fontweight='bold')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

# Save the confusion matrix figure
plt.savefig('./Images/confusion_matrix.png')
plt.show()
