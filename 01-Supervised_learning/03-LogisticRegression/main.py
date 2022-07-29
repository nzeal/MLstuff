import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1/3, random_state=0)
                                                                                
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
        return confusion_matrix, accuracy_score


def main(args):
    ana = anaylsis(args.path)
    ana.findData()
    ana.splitdata()
    ana.featureScaling()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='Social_Network_Ads.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)