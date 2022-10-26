import pandas as pd

class data_cleaning: 
    def __init__(self) -> None:
        pass

    def list_null_values(self, df):
        df = df.dropna()
        df = df.dropna(axis=1)
        df = df.dropna(axis=0)
        df = df.dropna(axis=0, how='any')
        return df
    
    def clean_data(self, df):
        df = df.dropna()
        df = df.dropna(axis=1)
        df = df.dropna(axis=0)
    
class feature_engineer:
    def __init__(self) -> None:
        pass

    def bucketing(self, df, columns, num_bins):
        df[col + 'binned'] = pd.cut(df[col], num_bins)
        return df

class DataFractory:
    def get_formatter(self, format):
        if format == 'Cleaning':
            return data_cleaning()
        elif format == 'Features':
            return feature_engineer()
        else:
            raise ValueError(format)

df = pd.read_csv(filename)
data_cleaning = DataFractory('Cleaning')
data_cleaning.list_null_values(df)

data_features = DataFractory('Features')
df = data_features.bucketing()
            