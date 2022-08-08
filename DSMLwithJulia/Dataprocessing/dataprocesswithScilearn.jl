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