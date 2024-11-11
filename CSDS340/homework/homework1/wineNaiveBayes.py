# Problem 4
import pandas as pd 
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import Binarizer

data = pd.read_csv("/Users/efang/Desktop/coding/Intro-to-ML/CSDS340/data/wine.data.csv", header=None)

attribute_df = data.iloc[:, 1:]

normalized_data = (attribute_df - attribute_df.mean())/attribute_df.std()

normalized_data = pd.concat([data.iloc[:, [0]], normalized_data], axis=1)

# Part a
x_train, x_test, y_train, y_test = train_test_split(normalized_data.iloc[:, 1:], normalized_data.iloc[:, 0], test_size=0.5, random_state=1)

gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

gnb_accuracy = accuracy_score(y_test, y_pred)

print(f"Gaussian naive Bayes test set accuracy: {gnb_accuracy}")

# Part b

best_accuracy = 0
best_threshold = None

for threshold in np.arange(-0.0, -1.6, -0.001):
    binarizer = Binarizer(threshold=threshold)
    binarized_data = binarizer.fit_transform(normalized_data.iloc[:, 1:])
    
    x_train, x_test, y_train, y_test = train_test_split(binarized_data, normalized_data.iloc[:, 0], test_size=0.5, random_state=1)
    
    bnl = BernoulliNB()
    bnl.fit(x_train, y_train)
    
    y_pred = bnl.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best threshold: {best_threshold}")
print(f"Best accuracy: {best_accuracy}")


"""
    The highest accuracy that I was able to obtain was 0.977 for the Bernoulli naive bayes
    classifier, versus the 0.97752 that I obtained with the Gaussian naive bayes classifier.

    I found the most optimal threshold by running a loop with increments of 0.001 until
    I reached the most accurate threshold. 
"""
