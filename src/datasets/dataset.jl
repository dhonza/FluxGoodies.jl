using CSV
using DataFrames

import Base: show

const Tabular = Union{AbstractArray,AbstractDataFrame}

deps(path...) = joinpath(@__DIR__, "deps", path...)

#TODO make it tabular
struct RawDataset
    df::DataFrame
    targets::Vector{Symbol}
    desc::AbstractString
end

inputs(d::RawDataset) = view(d.df, setdiff(names(d.df), d.targets))
targets(d::RawDataset) = view(d.df, d.targets)

function show(io::IO, mime::MIME"text/html", x::RawDataset)
    targets = join(x.targets, ", ")
    write(io, "<b>Target attributes: $targets</b>")
    show(io, mime, x.df)
end

function _load_UCI(uciname, cols, targets; cvsopts...)
    dir = deps(uciname)
    mkpath(dir)
    for fname in ["$uciname.data", "$uciname.names"]
        if !isfile(joinpath(dir, fname))
            download("https://archive.ics.uci.edu/ml/machine-learning-databases/$uciname/$fname", joinpath(dir, fname))
        end
    end
    df = CSV.read(joinpath(dir, "$uciname.data"); header = cols, copycols = true, cvsopts...)
    desc = read(joinpath(dir, "$uciname.names"), String)
    RawDataset(df, targets, desc)
end

function load_abalone()
    # 3133 training, final 1044 testing
    cols = [:sex, :length, :diameter, :height, :whole_weight, :shucked_weight, :viscera_weight, :shell_weight, :rings]
    targets = [:rings]
    _load_UCI("abalone", cols, targets)
end

function load_ecoli()
    cols = [:sequence_name, :mcg, :gvh, :lip, :chg, :aac, :alm1, :alm2, :class]
    targets = [:class]
    _load_UCI("ecoli", cols, targets; delim = " ", ignorerepeated = true)
end

function load_housing()
    cols = [:crim, :zn, :indus, :chas, :nox, :rm, :age, :dis, :rad, :tax, :ptratio, :b, :lstat, :medv]
    targets = [:medv]
    _load_UCI("housing", cols, targets; delim = " ", ignorerepeated = true)
end

function load_iris()
    cols = [:sepal_length, :sepal_width,  :petal_length, :petal_width, :class]
    targets = [:class]
    _load_UCI("iris", cols, targets; limit = 150)
end