import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linearRegression import LinearRegression
import argparse

# mean_squared_error = lambda y_true, y_pred: np.mean((y_true - y_pred) ** 2)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


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
        self.regressor = LinearRegression(learning_rate=0.01, niters=1000)
        self.regressor.fit(self.X_train, self.y_train)
        self.y_pred = self.regressor.predict(self.X_test)  # Predicting the Test set results
        self.y_pred_line = self.regressor.predict(self.X)
        print("Mean squared error:", mean_squared_error(self.y_test, self.y_pred), )
        print("Accuracy:", r2_score(self.y_test, self.y_pred))

    def visualSet(self):
        plt.figure(figsize=(8, 4), dpi=200)
        cmap = plt.get_cmap("viridis")
        font2 = {'family': 'serif', 'color': 'black', 'size': 15}
        m1 = plt.scatter(self.X_train, self.y_train, s=10)
        m2 = plt.scatter(self.X_test, self.y_test, s=10)
        m3 = plt.plot(self.X, self.y_pred_line, c="forestgreen", linewidth=1, label="Prediction")
        # Axis, legend
        plt.xlabel('Years of Experience', fontdict=font2)
        plt.ylabel('Salary', fontdict=font2)
        plt.legend(['Training set', 'Test set', 'Best fit'])
        plt.tight_layout()
        plt.savefig('LinearRegression.png')
        # plt.show()


def main(args):
    ana = anaylsis(args.path)
    ana.findData()
    ana.splitdata()
    ana.visualSet()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='../0-DATA/Salary_Data.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)
