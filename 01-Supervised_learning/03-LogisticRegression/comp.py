"""
This is other analysis
"""
#import mglearn
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


class anaylsis():
    def __init__(self, path: str):
        self._path = path
        self._dataset = pd.read_csv(self._path)

    def findData(self):
        df = self._dataset
        self.X = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values

    def visReg(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3))
        for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
            clf = model.fit(self.X, self.y)
            mglearn.plots.plot_2d_separator(clf, self.X, fill=False, eps=0.5, ax=ax, alpha=.7)
            mglearn.discrete_scatter(self.X[:, 0], self.X[:, 1], self.y, ax=ax)
            ax.set_title("{}".format(clf.__class__.__name__))
            ax.set_xlabel("Feature 0")
            ax.set_ylabel("Feature 1")
            axes[0].legend()
            plt.savefig('CompReg.png')


def main(args):
    ana = anaylsis(args.path)
    ana.findData()
    ana.visReg()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        default='Social_Network_Ads.csv',
                        help='Path. Default: %(default)s')
    args = parser.parse_args()
    main(args)
