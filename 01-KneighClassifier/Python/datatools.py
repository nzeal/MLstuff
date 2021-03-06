import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

path = 'Data.csv'


def dataprep():
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    return X
