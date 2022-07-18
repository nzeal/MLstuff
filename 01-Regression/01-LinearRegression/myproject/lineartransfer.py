import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


def slr(file):
    dataset = pd.read_csv(file)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return X, y
