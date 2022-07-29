import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse


class Weather():
    def __init__(self, path: str):
        self._path = path
        self._dataset = pd.read_csv(self._path, parse_dates=True, index_col=0)
        self._header = ['Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'RainTomorrow']
        self._rfig = plt.figure(figsize=(8, 4), dpi=160)
        self._knn = KNeighborsClassifier(n_neighbors=5)
        self._t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        self._epoch = 100

    def dataprep(self):
        df = self._dataset[self._header]
        dataset_clean = df.dropna()
        self.X = dataset_clean[self._header[:3]]
        self.y = dataset_clean[self._header[3]]
        self.y = np.array([0 if value == 'No' else 1 for value in self.y])

    def training_accu_test(self):
        training_accuracy = []
        test_accuracy = []

        neighbors_settings = range(1, 20)

        for n_neighbors in neighbors_settings:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)
            training_accuracy.append(clf.score(X_train, y_train))
            test_accuracy.append(clf.score(X_test, y_test))

        plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.savefig('Training_test_accuracy.png')


def main(args):
    weather = Weather(args.path)
    weather.dataprep()
    weather.training_accu_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store_true',
                        default='../../../0-DATA/weather.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)
