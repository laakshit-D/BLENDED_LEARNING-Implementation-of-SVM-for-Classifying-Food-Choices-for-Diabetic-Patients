# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load Data**: Import the dataset and separate features (X) and target (y).

2. **Split Data**: Divide into training (80%) and testing (20%) sets.

3. **Scale Features**: Standardize the features using `StandardScaler`.

4. **Define SVM Model**: Initialize a Support Vector Machine (SVM) classifier.

5. **Hyperparameter Grid**: Define a range of values for `C`, `kernel`, and `gamma` for tuning.

6. **Grid Search**: Perform Grid Search with Cross-Validation to find the best hyperparameters.

7. **Results Visualization**: Create a heatmap to show the mean accuracy for different combinations of hyperparameters.

8. **Best Model**: Extract the best model with optimal hyperparameters.

9. **Make Predictions**: Use the best model to predict on the test set.

10. **Evaluate Model**: Calculate accuracy and print the classification report.

## Program:
```py
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: LAAKSHIT D
RegisterNumber: 212222230071
*/
```
```py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("food_items_binary.csv")

features = ['Calories', 'Total Fat', 'Saturated Fat', 'Sugars', 'Dietary Fiber', 'Protein']
target = 'class'

X = df[features]
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

svm = SVC()

param_grid = {
    'C': [0.1,1,10,100],
    'kernel': ['linear','rbf'],
    'gamma': ['scale','auto']
}

grid_search = GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Name: LAAKSHIT D")
print("Register Number: 212222230071")
print("Best Parameters:",grid_search.best_params_)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
## Output:

<img width="653" height="776" alt="Screenshot 2025-10-22 113346" src="https://github.com/user-attachments/assets/374e3895-b0fc-46d3-bdb9-a49c274af97c" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
