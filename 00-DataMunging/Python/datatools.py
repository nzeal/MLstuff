# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

path = 'Data.csv'

# Importing and missing data 
def dataprep():
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    return X

