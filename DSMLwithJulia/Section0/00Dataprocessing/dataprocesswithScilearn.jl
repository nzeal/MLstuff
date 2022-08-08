# ==============================================================================================
# IMPORTS
using DataFrames, CSV, Statistics

# ==============================================================================================
# Loading data ..
dataset = CSV.read("Data.csv", DataFrame);
X = dataset[:,1:3];
y= dataset[:, :4];

# ==============================================================================================
# Fixxing missing data
## Encoding categorical data
using ScikitLearn
using DataFrames: DataFrame, missing
@sk_import preprocessing: (LabelBinarizer, StandardScaler)

mapper = DataFrameMapper([
     (:Country, LabelBinarizer()),
     (:Age, nothing),
     (:Salary, nothing)])


 X = fit_transform!(mapper, copy(dataset))

 mapper2 = DataFrameMapper([
    (:Purchased, LabelBinarizer())])

Y = fit_transform!(mapper2, copy(dataset))

## Splitting the dataset into the Training set and Test set
using ScikitLearn.CrossValidation: train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

## Feature Scaling

@sk_import preprocessing: (LabelBinarizer, StandardScaler)

sc = StandardScaler()
X_train[:, 4:5] = sc.fit_transform(X_train[:, 4:5]);
X_test[:, 4:5] = sc.transform(X_test[:, 4:5]);

print(X_train, X_test)

