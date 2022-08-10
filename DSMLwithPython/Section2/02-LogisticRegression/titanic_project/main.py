import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

def run():
	train = pd.read_csv('titanic_train.csv')
	train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
	train.drop('Cabin',axis=1,inplace=True)
	sex = pd.get_dummies(train['Sex'],drop_first=True)
	embark = pd.get_dummies(train['Embarked'],drop_first=True)
	train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
	train = pd.concat([train,sex,embark],axis=1)

	X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)
	logmodel = LogisticRegression()
	logmodel.fit(X_train,y_train)
	predictions = logmodel.predict(X_test)
	print(classification_report(y_test,predictions))


if __name__ == '__main__':
    run()