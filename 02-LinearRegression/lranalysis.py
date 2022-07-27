import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linearRegression import LinearRegression
import argparse

class anaylsis():
    def __init__(self, path: str):
        self._path = path
        self._dataset = pd.read_csv(self._path)
        self._rfig = plt.figure(figsize=(4, 4), dpi=160)
        self._epoch = 100
    
    def findData(self):
        df = self._dataset
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def splitdata(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 1/3, random_state = 0)

    def mean_sqr_err(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def visul(self):
        regressor = LinearRegression(learning_rate=0.01, niters=1000)
        regressor.fit(self.X_train, self.y_train)
        # Predicting the Test set results
        y_pred = regressor.predict(self.X_test)
        
        cmap = plt.get_cmap("viridis")
        fig  = plt.figure(figsize=(8, 6))
        m1   = plt.scatter(self.X_train, self.y_train, color=cmap(0.9), s=10)
        m2   = plt.plot(self.X_train, regressor.predict(self.X_train), color = 'blue')
        plt.title('Salary vs Experience (Training set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()

def main(args):
    ana = anaylsis(args.path)
    ana.findData()
    ana.splitdata()
    ana.visul()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='../0-DATA/Salary_Data.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)