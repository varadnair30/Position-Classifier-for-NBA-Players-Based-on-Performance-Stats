

# © Varad Nair

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Load NBA Stats Dataset data
data = pd.read_csv('nba_stats.csv')

#Filtering out players played less than 5 mins
data=data[data['MP']>=5]

# Selecting features and target
X = data[['MP','FG%','3P%','2P%','FT%','TOV','PF','PTS', 'AST', 'TRB', 'STL', 'BLK']]
y = data['Pos']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=0, stratify=y)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_train_pred_knn = knn.predict(X_train)
y_val_pred_knn = knn.predict(X_val)

# Setup 10-fold cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Perform cross-validation
scores = cross_val_score(knn, X_scaled, y, cv=cv)

# Accuracy and Confusion Matrix
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
val_accuracy_knn = accuracy_score(y_val, y_val_pred_knn)
train_conf_matrix_knn = confusion_matrix(y_train, y_train_pred_knn)
val_conf_matrix_knn = confusion_matrix(y_val, y_val_pred_knn)
print("---Analysis on NBA Dataset----")
print("KNN Training Accuracy:", train_accuracy_knn)
print("KNN Validation Accuracy:", val_accuracy_knn)
print("KNN Training Confusion Matrix:\n", train_conf_matrix_knn)
print("KNN Validation Confusion Matrix:\n", val_conf_matrix_knn)
# Print out the accuracy of each fold
print("Cross-validation scores:", scores)

# Print out the average accuracy across all the folds
print("Average cross-validation score: {:.2f}".format(scores.mean()))

# Load Dummy dataset
data = pd.read_csv('dummy_test.csv')

#Filtering out players played less than 5 mins
data=data[data['MP']>=5]

# Selecting features and target
X = data[['MP','FG%','3P%','2P%','FT%','TOV','PF','PTS', 'AST', 'TRB', 'STL', 'BLK']]
y = data['Pos']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=0, stratify=y)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_train_pred_knn = knn.predict(X_train)
y_val_pred_knn = knn.predict(X_val)

# Accuracy and Confusion Matrix
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
val_accuracy_knn = accuracy_score(y_val, y_val_pred_knn)
train_conf_matrix_knn = confusion_matrix(y_train, y_train_pred_knn)
val_conf_matrix_knn = confusion_matrix(y_val, y_val_pred_knn)
print("\n---Analysis on Dummy Dataset----")
print("KNN Training Accuracy:", train_accuracy_knn)
print("KNN Validation Accuracy:", val_accuracy_knn)
print("KNN Training Confusion Matrix:\n", train_conf_matrix_knn)
print("KNN Validation Confusion Matrix:\n", val_conf_matrix_knn)

