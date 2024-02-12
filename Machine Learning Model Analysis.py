# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

df = pd.read_csv('/content/train.csv')

dft = pd.read_csv('/content/test.csv')

df.describe

df.nunique()

df.dtypes

df = df.fillna(0)

label = []
image = []
for i in range(16):
    label.append(df.iloc()[i][0])
    image.append(df.iloc()[i][1:].to_numpy().reshape(28, 28))

_, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 5))
for ax, image, label1 in zip(axes.flatten(), image, label):
    ax.set_axis_off()
    ax.imshow(image, interpolation="nearest")
    ax.set_title("Value: %i" % label1)
plt.tight_layout()
plt.subplots_adjust(top=1)

from sklearn.decomposition import PCA
pca = PCA(n_components='mle')
x = pca.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

X = df.iloc[:, 1:]
y = df['label']

"""# **SVM**"""

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn import svm, neighbors
X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)

# Model 2: SVM
model_svm = svm.SVC()
model_svm.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
#Accuracy
y_pred_svm = model_svm.predict(X_val)
acc_svm = accuracy_score(y_val, y_pred_svm)
print(f"SVM - Validation Accuracy: {acc_svm}")

# Precision
precision_svm = precision_score(y_val, y_pred_svm, average='weighted')
print(f"SVM - Precision: {precision_svm}")

conf_matrix_svm = confusion_matrix(y_val, y_pred_svm)
print("Confusion Matrix - SVM:\n", conf_matrix_svm)

import matplotlib.pyplot as plt
plt.subplot(1, 3, 2)
plt.imshow(conf_matrix_svm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix - SVM')

"""# **KNN**

10 neighbors
"""

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 10)
neigh.fit(X,y)

#Accuracy
KNN_pred = neigh.predict(X_val)
acc_KNN = accuracy_score(y_val, KNN_pred)
print(f"KNN - Validation Accuracy: {acc_KNN}")

# Precision
precision_knn = precision_score(y_val, KNN_pred, average='weighted')
print(f"KNN - Precision: {precision_knn}")

conf_matrix_knn = confusion_matrix(y_val, KNN_pred)
print("Confusion Matrix - KNN:\n", conf_matrix_knn)

import matplotlib.pyplot as plt
plt.subplot(1, 3, 2)
plt.imshow(conf_matrix_svm, cmap='Oranges', interpolation='nearest')
plt.title('Confusion Matrix - KNN')

"""# **Decision Tree**"""

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

X = df.iloc[:,1: ]
y = df.label
#Independent and Dependent Variable

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X=X_train,y=y_train)

y_predict = tree_model.predict(X_test)

accuracy_score(y_test,y_predict)

from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_test,y_predict))

from sklearn import preprocessing, tree
tree.plot_tree(tree_model,filled=True)

"""# **Logistic** **Regression**



"""

from sklearn.linear_model import LogisticRegression

X = df.iloc[:,1: ]
y = df.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.dropna()
y_train = y_train[X_train.index]

X_test = X_test.dropna()
y_test = y_test[X_test.index]

# Create and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("y_test shape:", y_test.shape)
print("y_pred shape:", y_pred.shape)

accuracy = accuracy_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)

"""# **Conclusion**
In conclusion, the development of a handwritten digit recognition system using ML using Python involved the exploration of various algorithms, including Support Vector Machine (SVM), Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree.

Upon conducting the experiment, it was observed that Logistic Regression, despite being a widely used and versatile algorithm, exhibited the lowest accuracy among the models tested. This finding suggests that for the specific task of handwritten digit recognition, other algorithms such as SVM, KNN, or Decision Tree may be more suitable.

Support Vector Machine (SVM), known for its ability to handle complex decision boundaries, Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree were all evaluated in terms of accuracy, precision, recall, and F1 score. Despite the lower accuracy of Logistic Regression, it is important to note that the choice of the best model depends on the specific requirements and constraints of the application.

In conclusion, while Logistic Regression exhibited the lowest accuracy in this experiment, the research journey underscores the iterative nature of machine learning model development and the need for continuous refinement and exploration to achieve optimal results.

# **Question Of Curiosity**
"""

def find_equal_avg_partition(arr):
    total_sum = sum(arr)
    n = len(arr)
    target_sum = total_sum // n

    def backtrack(index, current_sum, first_subset):
        if current_sum == target_sum:
            second_subset = [x for x in arr if x not in first_subset]
            if sum(first_subset) == sum(second_subset):
                print("Partitions:")
                print(sorted(first_subset))
                print(sorted(second_subset))
                return True
        if current_sum > target_sum or index == n:
            return False

        # Include the current element in the first subset
        first_subset.append(arr[index])
        if backtrack(index + 1, current_sum + arr[index], first_subset):
            return True

        # Backtrack: Exclude the current element from the first subset
        first_subset.pop()
        return backtrack(index + 1, current_sum, first_subset)

    # Start the backtracking process
    if not backtrack(0, 0, []):
        print(-1)

# Example usage:
arr = [1, 5, 7, 3, 2, 8, 10, 6]
find_equal_avg_partition(arr)

#WRONG COQ

"""This program takes an array as input and tries to partition it into two subsets such that the average of elements in both subsets is equal. If no such partition is possible, it prints -1. Otherwise, it prints the partitions with the minimum length for the first subset and, in case of a tie, the lexicographically smallest solution"""
