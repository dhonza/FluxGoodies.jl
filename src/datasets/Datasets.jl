module Datasets

export Dataset, loaddataset, obsparts, colparts, colpartcols, nobs, ncols, tonumerical, buildtransform, transform, invert

using CSV
using DataFrames
using DataStructures: counter, DefaultDict, OrderedDict
import MLDataUtils: shuffleobs, splitobs, stratifiedobs
import MLDataPattern
import MLDataPattern: nobs, getobs
using Random
using RDatasets

import Base: show

import ..Transforms: tonumerical, ColTrans, toposort, transform, invert, outids
import ..FluxGoodies: dview, ncols

deps(path...) = joinpath(@__DIR__, "deps", path...)

"""
    Dataset

`Dataset` encapsulates any data partitioned along both observations (e.g. train, validation, test) and 
columns/features (e.g. inputs, targets).

Each column is identified by a `Symbol` and a description can be included.
"""
struct Dataset
    data
    cols::AbstractVector{Symbol}
    desc::AbstractString
    obspartitions::OrderedDict{Symbol,Any}
    colpartitions::OrderedDict{Symbol,Any}
    function Dataset(data, cols, desc, obspartitions, colpartitions)
        #TOOO check no intersections of indices
        op = OrderedDict{Symbol,Any}(obspartitions)
        cp = OrderedDict{Symbol,Any}(colpartitions)
        new(data, cols, desc, op, cp)
    end
end

"""
    Dataset(data::AbstractDataFrame, desc, obspartitions, colpartitions = nothing)

Create a dataset from `data` described by `desc`. Both `obspartitions` and `colpartitions` must be in 
a format convertible to `OrderedDict{Symbol,Any}`. If no `colpartitions` are given, two partitions are created 
by default: `:X` which contain all but last column and `:T` containing only the last column.
"""
function Dataset(data::AbstractDataFrame, desc, obspartitions, colpartitions = nothing)
    if isnothing(colpartitions)
        colpartitions = [:X => 1:size(data, 2) - 1, :T => [size(data, 2)]]
    end
    Dataset(data, names(data), desc, obspartitions, colpartitions)
end

"""
    Base.getindex(dataset::Dataset, obspartition = Colon(), colpartition = Colon())

Address the `Dataset` by observation and column partition ids (`Symbol`s). Use `:` to include all partitions.
"""
function Base.getindex(dataset::Dataset, obspartition = Colon(), colpartition = Colon())
    o = obspartition isa Colon ? obspartition : dataset.obspartitions[obspartition]
    c = colpartition isa Colon ? colpartition : dataset.colpartitions[colpartition]
    dview(dataset.data, o, c)
end

"""
    obsparts(d::Dataset)

Return all observation partition ids.
"""
obsparts(d::Dataset) = keys(d.obspartitions)

"""
    colparts(d::Dataset)

Return all column, i.e., feature partition ids.
"""
colparts(d::Dataset) = keys(d.colpartitions)

"""
    colpartcols(d::Dataset, p::Symbol)

Give all column names associated to the the column partition `p`. 
"""
colpartcols(d::Dataset, p::Symbol) = d.cols[d.colpartitions[p]]

"""
    MLDataPattern.nobs(d::Dataset, p::Symbol)

Number of observations in partition `p`.
"""
MLDataPattern.nobs(d::Dataset, p::Symbol) = length(d.obspartitions[p])

"""
    MLDataPattern.nobs(d::Dataset)

A sum observation numbers in all partitions.
"""
MLDataPattern.nobs(d::Dataset) = sum(nobs(d, p) for p in obsparts(d))

"""
    MLDataPattern.ncols(d::Dataset, p::Symbol)

Number of columns in partition `p`.
"""
ncols(d::Dataset, p::Symbol) = length(d.colpartitions[p])

"""
    MLDataPattern.ncols(d::Dataset, p::Symbol)

A sum of column numbers in all partitions.
"""
ncols(d::Dataset) = sum(ncols(d, p) for p in colparts(d))

MLDataPattern.nobs(g::GroupedDataFrame{DataFrame}) = length(g)
MLDataPattern.getobs(g::GroupedDataFrame{DataFrame}, r) = g[r]

function Base.show(io::IO, mime::MIME"text/html", d::Dataset)
    write(io, "<b>Observations:</b><br/>")
    write(io, """<table class="data-frame"><thead><tr><th>partition</th><th>size</th></tr></thead>""")
    for (k, v) in d.obspartitions
        write(io, """<tr><td>$(string(k))</td><td>$(length(v))</td></tr>""")
    end
    write(io, """</table>""")
    
    write(io, "<b>Columns:</b><br/>")
    write(io, """<table class="data-frame"><thead><tr><th>partition</th><th>columns</th></tr></thead>""")
    for (k, v) in d.colpartitions
        cols = join(["<td>$(string(c))</td>" for c in d.cols[d.colpartitions[k]]])
        write(io, """<tr><th>$(string(k))</th>$cols</tr>""")
    end
    write(io, """</table>""")

    # """<table class="data-frame"><thead><tr><th></th><th>sex</th><th>length</th><th>diameter</th><th>height</th><th>whole_weight</th><th>shucked_weight</th><th>viscera_weight</th><th>shell_weight</th><th>rings</th></tr><tr><th></th><th>String</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Float64</th><th>Int64</th></tr></thead><tbody><p>2 rows × 9 columns</p><tr><th>1</th><td>M</td><td>0.455</td><td>0.365</td><td>0.095</td><td>0.514</td><td>0.2245</td><td>0.101</td><td>0.15</td><td>15</td></tr><tr><th>2</th><td>M</td><td>0.35</td><td>0.265</td><td>0.09</td><td>0.2255</td><td>0.0995</td><td>0.0485</td><td>0.07</td><td>7</td></tr></tbody></table>"""
    show(io, mime, d.data)
end

function _load_UCI(uciname, cols, obspartitions, colpartitions; cvsopts...)
    dir = deps(uciname)
    mkpath(dir)
    for fname in ["$uciname.data", "$uciname.names"]
        if !isfile(joinpath(dir, fname))
            download("https://archive.ics.uci.edu/ml/machine-learning-databases/$uciname/$fname", joinpath(dir, fname))
        end
    end
    df = CSV.read(joinpath(dir, "$uciname.data"); header = cols, copycols = true, cvsopts...)
    if obspartitions isa Function
        idxs = obspartitions(df)
        df = df[vcat(last.(idxs)...),:] #TODO make memory efficient
        off = 1
        obs = Pair{Symbol,UnitRange{Int}}[]
        for (p, is) in idxs
            push!(obs,  p => off:off + length(is) - 1)
            off += length(is)
        end
        obspartitions = obs
    end
    desc = read(joinpath(dir, "$uciname.names"), String)
    Dataset(df, desc, obspartitions, colpartitions)
end

"""
    loaddataset(name::Symbol)

Load a dataset identified by the `name`.

Available datasets are: `abalone`, `ecoli`, `housing` and `iris`.
"""
function loaddataset(name::Symbol)
    eval(Symbol("_load_$(string(name))"))()
end

function _shufflesplit(df, obsparts::Vector{Symbol}, fractions; seed = 1)
    Random.seed!(seed)
    Pair.(obsparts, first.(parentindices.(splitobs(shuffleobs(df), fractions))))
end

function _shufflestratifiedsplit(f, df, obsparts::Vector{Symbol}, fractions; seed = 1)
    Random.seed!(seed)
    Pair.(obsparts, first.(parentindices.(stratifiedobs(f, df, fractions))))
end

function _load_abalone()
    cols = [:sex, :length, :diameter, :height, :whole_weight, :shucked_weight, :viscera_weight, :shell_weight, :rings]
    _load_UCI("abalone", cols, df->_shufflestratifiedsplit(row->row[:rings], df, [:train, :test], 0.75), 
                                    [:X => 1:length(cols) - 1, :T => [length(cols)]])
end

function _load_ecoli()
    cols = [:sequence_name, :mcg, :gvh, :lip, :chg, :aac, :alm1, :alm2, :class]
    targets = [:class]
    _load_UCI("ecoli", cols, df->_shufflestratifiedsplit(row->row[:class], df, [:train, :test], 0.5), 
            [:X => 1:length(cols) - 1, :T => [length(cols)]]; 
            delim = " ", ignorerepeated = true)
end

function _load_housing()
    cols = [:crim, :zn, :indus, :chas, :nox, :rm, :age, :dis, :rad, :tax, :ptratio, :b, :lstat, :medv]
    _load_UCI("housing", cols, df->_shufflesplit(df, [:train, :test], 0.8), [:X => 1:length(cols) - 1, :T => [length(cols)]]; 
        delim = " ", ignorerepeated = true)
end

function _load_iris()
    cols = [:sepal_length, :sepal_width,  :petal_length, :petal_width, :class]
    targets = [:class]
    _load_UCI("iris", cols, df->_shufflestratifiedsplit(row->row[:class], df, [:train, :test], 0.8), 
            [:X => 1:length(cols) - 1, :T => [length(cols)]]; limit = 150)
end


# ----- Transforms --------
# -------------------------

struct DatasetTransform
    trns::OrderedDict{Symbol,ColTrans}
end

function tonumerical(dataset::Dataset,  obspart::Union{Symbol,Colon}, colpartopts::Pair{Symbol,<:AbstractDict}...)
    #TODO remove use buildtransform instead?
    cpsetup = DefaultDict{Symbol,Any,Any}(Dict(:nominal => nothing, :standardize => true))
    foreach(e->(cpsetup[first(e)] = last(e)), colpartopts)
    DatasetTransform(OrderedDict(cp => tonumerical(dataset[obspart,cp]; cpsetup[cp]...) for cp in colparts(dataset)))
end

function buildtransform(dataset::Dataset, obspart::Union{Symbol,Colon}, colpartfuncs::Pair{Symbol,<:Function}...)
    cpfs = OrderedDict(colpartfuncs)
    DatasetTransform(OrderedDict(cp => cpfs[cp](dataset[obspart,cp]) for cp in keys(cpfs)))
end

function invert(dt::DatasetTransform, dataset::Dataset)
    trns = OrderedDict(cp => invert(trn, ) for (cp, trn) in dt.trns)
end

function _transformhelper(dt::DatasetTransform)
    colpartitions = OrderedDict{Symbol,UnitRange{Int}}()
    cols = Symbol[]
    off = 1
    for cp in keys(dt.trns)
        cnames = outids(dt.trns[cp])
        colpartitions[cp] = off:off + length(cnames) - 1
        off += length(cnames)
        cols = [cols..., cnames...]
    end
    cols, colpartitions
end

function transform(d::Dataset, dt::DatasetTransform, type::Type{<:AbstractDataFrame}; sid = :default)
    data = hcat([transform(dt.trns[col], Dict(sid => d[:,col]), type) for col in keys(dt.trns)]...)
    cols, colpartitions = _transformhelper(dt)
    Dataset(data, d.desc, copy(d.obspartitions), colpartitions)
end

function transform(d::Dataset, dt::DatasetTransform, type::Type{<:Matrix}; sid = :default)
    data = vcat([transform(dt.trns[col], Dict(sid => d[:,col]), type) for col in keys(dt.trns)]...)
    cols, colpartitions = _transformhelper(dt)
    Dataset(data, cols, d.desc, copy(d.obspartitions), colpartitions)
end


Base.getindex(dataset::DatasetTransform, idx) = dataset.trns[idx]

function Base.show(io::IO, dtrn::DatasetTransform)
    for (i, (cp, trn)) in enumerate(dtrn.trns)
        ts = toposort(trn)
        println(io, "Column partition: $cp")
        for e in ts
            println(io, e)
        end
        i < length(dtrn.trns) && println(io)
    end
end

end