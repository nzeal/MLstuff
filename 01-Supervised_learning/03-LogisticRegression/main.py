import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class anaylsis():
    def __init__(self, path: str):
        self._path = path
        self._dataset = pd.read_csv(self._path)

    def findData(self):
        df = self._dataset
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def splitdata(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,
                                                                                self.y,
                                                                                test_size=1/3,
                                                                                random_state=0)

    def featureScaling(self):
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
        self.regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_test)

    def confusionMatrics(self):
        confusion_matrix(self.y_test, self.y_pred)
        accuracy_score(self.y_test, self.y_pred)

    def visualSet(self):
        plt.figure(figsize=(8, 4), dpi=160)
        X_set, y_set = self.sc.inverse_transform(self.X_train), self.y_train

        X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 10, stop=X_set[:, 0].max() + 10, step=0.25),
                             np.arange(start=X_set[:, 1].min() - 1000, stop=X_set[:, 1].max() + 1000, step=0.25))
        plt.contourf(X1, X2,
                     self.regressor.predict(self.sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                     alpha=0.75, cmap=ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
        plt.title('Logistic Regression (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()


def main(args):
    ana = anaylsis(args.path)
    ana.findData()
    ana.splitdata()
    ana.featureScaling()
    ana.visualSet()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='Social_Network_Ads.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)
