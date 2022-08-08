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
q = describe(dataset)
print(q)
function secondol()
    secondcol = dataset[:,2]
    for i in eachindex(secondcol)
        if (ismissing.(secondcol))[i] == 1
            secondcol[i] = convert(Int64, round(q[:,2][2], digits=0))
        end
    end
    return secondcol
end 

function thirdcol()
    thirdcol = dataset[:,3]
    for i in eachindex(thirdcol)
        if (ismissing.(thirdcol))[i] == 1
            thirdcol[i] = convert(Int64, round(63777.8, digits=0))
        end
    end
        return thirdcol
end 

print(thirdcol())

dataset[:,2] = push!(secondol());
dataset[:,3] = push!(thirdcol());

print("Clean data set\n")
print("==============\n")
print(dataset)