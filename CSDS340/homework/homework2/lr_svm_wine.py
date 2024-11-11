# Problem 4
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_path = "/Users/efang/Desktop/coding/Intro-to-ML/CSDS340/data/wine.data.csv"

wine_data = pd.read_csv(file_path, header=None)

wine_data.head()

X = wine_data.iloc[:, 1:].values
y = wine_data.iloc[:, 0].values

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_standardized, y, test_size = 0.5, random_state = 1)
#X_train.shape, X_test.shape, y_train.shape, y_test.shape

logreg = LogisticRegression(random_state=1, max_iter=1000)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)

logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {logreg_accuracy}")

from sklearn.svm import SVC

svm_linear = SVC(kernel='linear', random_state=1)
svm_rbf = SVC(kernel='rbf', gamma=0.1, random_state=1)
svm_poly = SVC(kernel='poly', degree=3, random_state=1) 

svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
svm_linear_accuracy = accuracy_score(y_test, y_pred_svm_linear)

svm_rbf.fit(X_train, y_train)
y_pred_svm_rbf = svm_rbf.predict(X_test)
svm_rbf_accuracy = accuracy_score(y_test, y_pred_svm_rbf)

svm_poly.fit(X_train, y_train)
y_pred_svm_poly = svm_poly.predict(X_test)
svm_poly_accuracy = accuracy_score(y_test, y_pred_svm_poly)

print(f"SVM Polynomial: {svm_poly_accuracy}")
print(f"SVM RBF: {svm_rbf_accuracy}")
print(f"SVM Linear: {svm_linear_accuracy}")
