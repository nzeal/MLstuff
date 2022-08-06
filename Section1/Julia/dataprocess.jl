# ==============================================================================================
# IMPORTS

using DataFrames, CSV, Statistics


dataset = CSV.read("Data.csv", DataFrame);
X = dataset[:,1:3];
y= dataset[:, :4];

q = describe(dataset)

print(q[:,2])