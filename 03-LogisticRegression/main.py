import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


class analysisLR(object):
    def __init__(self, path: str):
        self._path = path
        self._dataset = pd.read_csv('Social_Network_Ads.csv')

    def dataprep(self):
        self.X = self._dataset.iloc[:, :-1].values
        self.y = self._dataset.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0)


def main(args):
    ana = analysisLR(args.path)
    ana.dataprep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='Social_Network_Ads.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)
