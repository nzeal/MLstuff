# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

path = '../../0-DATA/weather.csv'

# data preparation
def dataprep():
    dataset = pd.read_csv(path, parse_dates=True, index_col=0)
    header = ['Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'RainTomorrow']
    df = dataset[header]
    dataset_clean = df.dropna()
    X = dataset_clean[header[:3]]
    y = dataset_clean[header[3]]
    y = np.array([0 if value == 'No' else 1 for value in y])
    return X, y

dataset = dataprep()
X = dataset[0]
y = dataset[1]

def visualize_krange():
    k_range = range(1,20)
    scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors = k)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))

    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.scatter(k_range, scores)
    plt.show()

visualize_krange()

rfig = plt.figure(figsize=(4,4),dpi=160)
knn = KNeighborsClassifier(n_neighbors = 5)
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
def visualize_classifier(X, y, t, rfig, knn):
    for s in t:
        scores = []
        for i in range(1,1000):
            print(i)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
            knn.fit(X_train, y_train)
            scores.append(knn.score(X_test, y_test))
        plt.plot(s, np.mean(scores), 'bo')

    plt.xlabel('Training set proportion (%)')
    plt.ylabel('accuracy')
    plt.show()

visualize_classifier(X, y, t, rfig, knn)