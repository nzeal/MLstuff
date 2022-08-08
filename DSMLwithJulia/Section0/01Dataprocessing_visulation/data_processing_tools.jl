# ==============================================================================================
# IMPORTS
using DataFrames, CSV, Statistics

# ==============================================================================================
# Loading data ..
fd = CSV.read("weather_by_cities.csv", DataFrame);

gdf = groupby(fd, :city)

function city()
    for city in gdf
        print(city)
    end
end


city_name = String[]
for gr in gdf 
   push!(city_name, gr[1, :city])
end

get_group(gdf, keys...) = gdf[(keys...,)]

g = get_group(gdf, "paris")

combine(gdf, :temperature => mean)
combine(gdf, :temperature => maximum)
combine(gdf, :temperature => (x -> [extrema(x)]) => [:min, :max])

combine(gdf) do fd
    m = mean(fd.temperature)
end

combine(gdf) do fd
    n = var(fd.temperature)
end


# Visualization

using StatsPlots
p1 = @df fd plot(:city, [:temperature :windspeed], colour = [:red :blue])
p2 = @df fd plot(:city, cols(3:4), colour = [:red :blue])
p3 = @df fd scatter(
    :temperature,
    :windspeed,
    group = :city,
    m = (0.5, [:+ :h :star7], 12),
    bg = RGB(0.2, 0.2, 0.2)
)

p4 = @df fd density([:temperature, :windspeed], group = (:city), legend = :topright)


savefig(p1,"plt1.png")
savefig(p2,"plt2.png")
savefig(p3,"plt3.png")
savefig(p4,"plt4.png")