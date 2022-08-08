### Importing the libraries

begin
    using DataFrames
    using Statistics
    using CSV
    using ScikitLearn
    using Plots
    using BenchmarkTools
    using PyCall
end

## Importing the dataset

dataset = CSV.read("Salary_Data.csv", DataFrame);
X = dataset[:, :1];
y = dataset[:, :2];

p1 = scatter(X, y, color=:black, leg=false, xaxis="Years of Experience", yaxis="Salary")
title!("Salary vs Experience")

savefig(p1,"plot1.png")

## Splitting the dataset into the Training set and Test set
using ScikitLearn.CrossValidation: train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Linear Regression with single variable (i.e salary)
### Training the Simple Linear Regression model on the Training set

using GLM, DataFrames

dataTrain = DataFrame(X_train=X_train, y_train=y_train);
linearRegressor = lm(@formula(y_train ~ X_train), dataTrain)
y_pred = GLM.predict(linearRegressor);

## Visualising the Training set results
p2=scatter(X_train, y_train, legend = :topleft)
plot!(X_train, y_pred , legend = :topleft)
title!("Salary vs Experience (Training set)")
xaxis!("Years of Experience", minorgrid = true)
yaxis!("Salary", minorgrid = true)

savefig(p2,"plot2.png")

## Visualising the Test set results
p3=scatter(X_test, y_test, color=:red, legend = :topleft)
plot!(X_train,  y_pred, color=:blue, legend = :topleft)
title!("Salary vs Experience (Training set)")
xaxis!("Years of Experience", minorgrid = true)
yaxis!("Salary", minorgrid = true)

savefig(p3,"plot3.png")

## Predicting the Test set results

function find_best_fit(xvals,yvals)
    meanx = mean(xvals)
    meany = mean(yvals)
    stdx = std(xvals)
    stdy = std(yvals)
    r = cor(xvals,yvals)
    a = r*stdy/stdx
    b = meany - a*meanx
    return a,b
end

a,b = find_best_fit(X,y)
ynew = a .* X .+ b;

np = pyimport("numpy");

@time myfit = np.polyfit(X, y, 1);
ynew2 = collect(X) .* myfit[1] .+ myfit[2];

p4=scatter(X_train, y_train, legend = :topleft)
plot!(X, ynew2, legend = :topleft)
plot!(X, ynew, linestyle = :dash)
title!("Salary vs Experience (Training set)")
xaxis!("Years of Experience", minorgrid = true)
yaxis!("Salary", minorgrid = true)
savefig(p3,"plot4.png")

p5=scatter(X_test, y_test, color=:red, legend = :topleft)
plot!(X, ynew, color=:blue, legend = :topleft)
title!("Salary vs Experience (Training set)")
xaxis!("Years of Experience", minorgrid = true)
yaxis!("Salary", minorgrid = true)
savefig(p5,"plot5.png")