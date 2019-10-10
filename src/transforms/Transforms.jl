module Transforms
export ColTrans, datasource, mergecols, onehot, onecold, replacevals, standardize, fit!, transform, invert, collecttrans, 
    deserialize, tonumerical, copytransform, inids, outids, intype, outtype, getdatasource, invtransform

using DataFrames
using DataStructures: counter, OrderedDict, OrderedSet
using JSON
using MLDataUtils
using Statistics

import ..FluxGoodies: toposort, dview, ncols

# TODO different meta and IO options?
customshow(io, e) = show(io, e)
customshow(io, e::AbstractString) = print(io, "\"$e\"")
customshow(io, p::Pair) = begin customshow(io, first(p)); print(io, " => "); customshow(io, last(p)) end

function customshow(io, v::AbstractVector, brackets = true)
    brackets && print(io, "[")
    for (i, e) in enumerate(v)
        i > 1 && print(io, ", ")
        customshow(io, e)
    end
    brackets && print(io, "]")
end

customshow(io, v::AbstractDict, brackets = true) = customshow(io, collect(v), brackets)

_smap_process(smap) = smap isa AbstractDict ? smap : Dict(:default => smap)

# ----- ColTrans -----
# --------------------
"Abstract Column transform."
abstract type ColTrans end

# ----- Connections -------
# -------------------------

"""
Describes input connection by its source transform `trn` and possibly its `type`.
"""
mutable struct Inp
    trn::ColTrans
    type::Type
    Inp(trn, type = Nothing) = new(trn, type)
end


"""
Describes output connection assigning it a `type`.
"""
mutable struct Out
    type::Type
    Out(type = Nothing) = new(type)
end

"""
Maps id to a transformation.
"""
const IdColTrans = Pair{Symbol,<:ColTrans}

"""
Maps input id to the input connection (`Inp`).
"""
const IdInp = Pair{Symbol,Inp}

IdInp(i::IdInp) = i
IdInp(i::IdColTrans) = first(i) => Inp(last(i))

"""
Maps output id to the output connection (`Out`).
"""
const IdOut = Pair{Symbol,Out}

IdOut(o::IdOut) = o
IdOut(o::IdColTrans) = first(o) => Out()

"""
Manage transform node input and output connections.
"""
mutable struct Connections
    ins::OrderedDict{Symbol,Inp}
    outs::OrderedDict{Symbol,Out}
    maxins::Int
    maxouts::Int
    function Connections(ins::OrderedDict{Symbol,Inp}, outs::OrderedDict{Symbol,Out}; maxins = typemax(Int), maxouts = typemax(Int))
        length(ins) > maxins && throw(ArgumentError("too many inputs $(length(ins)) > $(maxins)!"))
        length(outs) > maxouts && throw(ArgumentError("too many outputs $(length(outs)) > $(maxouts)!"))
        new(ins, outs, maxins, maxouts)
    end
end

function Connections(ins = IdInp[], outs = IdOut[]; kwargs...) 
    length(ins) == length(unique(map(first, ins))) || error("duplicate input ids!")
    length(outs) == length(unique(map(first, outs))) || error("duplicate output ids!")
    Connections(OrderedDict(IdInp[IdInp(i) for i in ins]), OrderedDict(IdOut[IdOut(o) for o in outs]); kwargs...)
end

function Base.push!(c::Connections, id::Symbol, inp::Inp)
    length(c.ins) >= c.maxins && throw(ArgumentError("more than $(c.maxins) inputs not allowed!"))
    id ∈ keys(c.ins) && throw(ArgumentError("input with id $(id) already exists!"))
    push!(c.ins, id => inp)
end

function Base.push!(c::Connections, id::Symbol, out::Out)
    length(c.outs) >= c.maxouts && throw(ArgumentError("more than $(c.maxouts) outputs not allowed!"))
    id ∈ keys(c.outs) && throw(ArgumentError("output with id $(id) already exists!"))
    push!(c.outs, id => out)
end

inids(c::Connections) = collect(keys(c.ins))
intrns(c::Connections) = [v.trn for v in values(c.ins)]
outids(c::Connections) = collect(keys(c.outs))

function Base.show(io::IO, c::Connections)
    inrule = (r)->"$(first(r))(#$(last(r).trn.id))"
    if length(c.ins) > 0
        print(io, join(inrule.(collect(c.ins)), ", "))    
    else
        print(io, "empty")
    end
end

function Base.getindex(c::Connections, e::Symbol)
    for (col, tran) in c.in
        col == e && return tran
    end
    throw(KeyError(e))
end

# ----- ColTrans -----
# --------------------
coltranscounter = Threads.Atomic{Int}(1)
_newid() = Threads.atomic_add!(coltranscounter, 1)

inids(c::ColTrans) = inids(c.con)
intrns(c::ColTrans) = intrns(c.con)
intrn(c::ColTrans, id::Symbol) = c.con.ins[id].trn
intype(c::ColTrans, id::Symbol) = c.con.ins[id].type
outids(c::ColTrans) = outids(c.con)
outtype(c::ColTrans, id::Symbol) = c.con.outs[id].type

idstr(t::ColTrans) = "#$(t.id)"
instancestr(t::ColTrans) = string(typeof(t).name.name) * "($(idstr(t)))"

"""
    isfit(t::ColTrans)

Find if the transform `t` is fit.
"""
isfit(t::ColTrans) = true

"""
    isfitrec(t::ColTrans)

Find recursively if the transform `t` is fit.
"""
isfitrec(t::ColTrans) = all((isfit(t), isfitrec.(intrns(t))...))

"""
     checkfit(t::ColTrans)

Throw error if the transform `t` is not fit.
"""
checkfit(t::ColTrans) = isfit(t) || error("$(t) not initialized, use fit!")

"""
     checkfitrec(t::ColTrans)

Throw error if the transform `t` or any of its predecessors is not fit.
"""
checkfitrec(t::ColTrans) = isfitrec(t) || error("$(t) not initialized, use fit!")


"""
     check(t::ColTrans)

Check transform `t` validity and throw error if needed.

Currently checking for more than one `DataSource`. More to come...
"""
function check(t::ColTrans)
    ts = typeof.(collecttrans(t))
    cnts = counter(ts)
    if cnts[DataSource] > 1
        error("More than one DataSource!")
    end
end

function _fit_helper!(c::Connections, t::ColTrans, smap)
    smap = _smap_process(smap)
    #TODO optimize! Some calls are now performed multiple times!
    data = OrderedDict() # maps predecessor transform to it output
    cache = Dict{ColTrans,Any}()
    for (id, input) in c.ins # fit all inputs recursively storing results to the cache
        if input.trn ∉ keys(cache)
            cache[input.trn] = fit!(input.trn, smap)
        end
        data[id] = cache[input.trn][id]
    end
    ret = fit!(t, data, smap) # compute output of this transform
    for (i, d) in zip(values(c.ins), values(data)) # fix input types based on output of predecessors
        if i.type == Nothing
            i.type = eltype(d)
        end
    end
    for (o, r) in zip(values(c.outs), values(ret)) # set ouput types
        if o.type == Nothing
            o.type = eltype(r)
        end
    end
    ret
end

"""
    fit!(t::ColTrans, smap)

Fit all transforms rooted at `t` using the source map `smap`.
"""
fit!(t::ColTrans, smap) = _fit_helper!(t.con, t, smap)

function _checktransform(t::ColTrans, smap)
    checkfit(t)
    length(smap) > 0 || error("no source!")
    ns = [nobs(s) for s in values(smap)]
    minimum(ns) != maximum(ns) && throw(ArgumentError("source sizes not same; $ns"))
end

"""
    transform(t::ColTrans, smap, ::Type{<:AbstractDataFrame})

Apply transform `t` to source map `smap` to create a new DataFrame.

# Example
```jldoctest
julia> using DataFrames;

julia> df = DataFrame(name = ["car", "bus", "cat"], category = [1, 1, 2], weight = [1000, 10000, 5.0])
3×3 DataFrames.DataFrame
│ Row │ name   │ category │ weight  │
│     │ String │ Int64    │ Float64 │
├─────┼────────┼──────────┼─────────┤
│ 1   │ car    │ 1        │ 1000.0  │
│ 2   │ bus    │ 1        │ 10000.0 │
│ 3   │ cat    │ 2        │ 5.0     │

julia> transform(tonumerical(df), df, DataFrame)
3×5 DataFrames.DataFrame
│ Row │ name_bus │ name_car │ name_cat │ category │ weight    │
│     │ Float64  │ Float64  │ Float64  │ Float64  │ Float64   │
├─────┼──────────┼──────────┼──────────┼──────────┼───────────┤
│ 1   │ -0.57735 │ 1.1547   │ -0.57735 │ -0.57735 │ -0.484631 │
│ 2   │ 1.1547   │ -0.57735 │ -0.57735 │ -0.57735 │ 1.14998   │
│ 3   │ -0.57735 │ -0.57735 │ 1.1547   │ 1.1547   │ -0.665346 │
```
"""
function transform(t::ColTrans, smap, ::Type{<:AbstractDataFrame})
    smap = _smap_process(smap)
    _checktransform(t, smap)
    os = outids(t)
    data = [t[o](smap) for o in os]
    DataFrame(data, os)
end

"""
    transform(t::ColTrans, smap, type::Type{<:AbstractMatrix})

Apply transform `t` to source map `smap` to create a new Matrix.

# Example
```jldoctest
julia> using DataFrames;

julia> df = DataFrame(name = ["car", "bus", "cat"], category = [1, 1, 2], weight = [1000, 10000, 5.0])
3×3 DataFrames.DataFrame
│ Row │ name   │ category │ weight  │
│     │ String │ Int64    │ Float64 │
├─────┼────────┼──────────┼─────────┤
│ 1   │ car    │ 1        │ 1000.0  │
│ 2   │ bus    │ 1        │ 10000.0 │
│ 3   │ cat    │ 2        │ 5.0     │

julia> transform(tonumerical(df), df, Matrix{Float32})
5×3 Array{Float32,2}:
 -0.57735    1.1547   -0.57735
  1.1547    -0.57735  -0.57735
 -0.57735   -0.57735   1.1547
 -0.57735   -0.57735   1.1547
 -0.484631   1.14998  -0.665346
```
"""
function transform(t::ColTrans, smap, type::Type{<:AbstractMatrix})
    smap = _smap_process(smap)
    _checktransform(t, smap)
    n = nobs(first(values(smap)))
    os = outids(t)
    M = type(undef, length(os), n)
    for (i, o) in enumerate(os)
        M[i,:] .= t[o](smap)
    end
    M
end

# ----- DataSource -----
# ----------------------
struct DataSource <: ColTrans
    id::Int
    con::Connections # no inputs here
    cols::OrderedDict{Symbol,Function} # column id to function which reads the data based on source map
    sids::OrderedDict{Symbol,Symbol} # column id to sid
    function DataSource(con::Connections, cols::OrderedDict{Symbol,Function}, sids::OrderedDict{Symbol,Symbol})
        con.maxins != 0 && throw(ArgumentError("maxins != 0!"))
        new(_newid(), con, cols, sids)
    end
end

Base.getindex(t::DataSource, col::Symbol) = t.cols[col]

function Base.show(io::IO, t::DataSource) 
    print(io, instancestr(t) * "[" * join(["$col($(t.sids[col]))" for col in keys(t.cols)], ", ") * "]")
end

"""
    datasource(s::AbstractDataFrame; sid::Symbol=:default)

Create a `DataSource` with source id `sid` based on a DataFrame `s`.
"""
datasource(s::AbstractDataFrame; sid::Symbol=:default) = datasource(s, names(s); sid=sid)

"""
    datasource(s, names::Vector{Symbol}; sid::Symbol=:default)

Create a `DataSource` with source id `sid` based on a source object `s`.

The column names are given by `names`.  Method `ncols(s)` must be defined for the object.
"""
function datasource(s, names::Vector{Symbol}; sid::Symbol=:default)
    ncols(s) != length(names) && error("source $(typeof(s)) size $(size(s)) does not match the number of col names ($(length(names)))")
    colsid = OrderedDict{Symbol,Symbol}(col => sid for col in names)
    datasource(colsid)
end

"""
    datasource(colsid::OrderedDict)

Create a `DataSource` based on ordered dictionary of column id => source id pairs. 

Data container must support `dview` method and access by column indices.
"""
function datasource(colsid::OrderedDict)
    con = Connections(IdInp[], [col => Out() for col in keys(colsid)]; maxins = 0)
    colsid = OrderedDict{Symbol,Symbol}(Symbol(col) => Symbol(sid) for (col, sid) in colsid)
    colfunc = OrderedDict{Symbol,Function}(col => ((i)->dview(i[sid], :, colidx)) for (colidx, (col, sid)) in enumerate(colsid))
    DataSource(con, colfunc, colsid)
end

function fit!(t::DataSource, data, smap)
    # nothing to fit here just give column id => column data dict
    OrderedDict(col => t[col](smap) for col in outids(t))
end

"""
    getdatasource(t::ColTrans)

Finds the only DataSource in the transformation DAG rooted in `t`.
"""
function getdatasource(t::ColTrans)
    filter(trn->typeof(trn) == DataSource, collecttrans(t)) |> first
end

# ----- MergeCols -----
# ---------------------
struct MergeCols <: ColTrans
    id::Int
    con::Connections
    function MergeCols(con::Connections)
        new(_newid(), con)
    end
end

"""
     mergecols(ins...)

Merges all input id to input connection mappings in `ins`. 

No id duplicates are allowed.
"""
function mergecols(ins...)
    outs = IdOut[first(i) => Out() for i in ins]
    MergeCols(Connections(ins, outs))
end

"""
     mergecols(cs::ColTrans...)

Merges outputs of all transforms `cs`. 

All transforms must be fitted or error is thrown. Also no output id duplicates are allowed.
"""
function mergecols(cs::ColTrans...)
    ins = IdInp[]
    cols = Set{Symbol}()
    for c in cs
        checkfitrec(c)
        for outid in outids(c)
            outid ∈ cols && error("duplicate symbols: $(outid)")
            push!(ins, outid => Inp(c))
            push!(cols, outid)
        end
    end
    mergecols(ins...)
end

function Base.show(io::IO, t::MergeCols)
    print(io, instancestr(t) * "[" * repr(t.con) * "]")
end

"""
    Base.getindex(t::MergeCols, col)

Give a column evaluation function for `col`.

The column evaluation function (CEF) expects a source map parameter (`smap`) to give actual results.
"""
function Base.getindex(t::MergeCols, col)
    fsrc = intrn(t, col)[col]
    (i)->func(t, col, OrderedDict(col => fsrc(i)))
end

"""
    func(t::MergeCols, col, data)

Get column evaluation function for `col` based on input `data`.
"""
func(t::MergeCols, col, data) = data[col]

"""
    fit!(t::MergeCols, data, smap)

Nothing to fit for `mergecols`. Just return a dict of column evaluation functions.
"""
function fit!(t::MergeCols, data, smap)
    OrderedDict(col => func(t, col, data) for col in outids(t))
end

# ----- One Hot -----
# -------------------
struct OneHot <: ColTrans
    id::Int
    con::Connections
    tbl::OrderedDict{Symbol,Any} # output column id to value
    function OneHot(con::Connections, tbl::OrderedDict{Symbol,Any})
        con.maxins > 1 && throw(ArgumentError("OneHot: only single input allowed!"))
        new(_newid(), con, tbl)
    end
end

"""
    onehot(in, tblpairs::Pair...)

Define one hot encoding transformation.

Input is given as: `in = icol => itrans`, where `icol` is the output column of the input 
transform `itrans`. Transform table `tblpairs` can be either given or automatically generated later using by `fit!()`. 
If generated, the output columns are denoted `icol_v1`, `icol_v2`, ..., where `v`i denotes possible values of `icol` 
(sorted when fitting). Default output type is `Bool`.
"""
function onehot(in, tblpairs::Pair...)
    OneHot(Connections([in], []; maxins = 1), OrderedDict{Symbol,Any}(tblpairs...))
end

isfit(t::OneHot) = length(t.tbl) > 0

function Base.show(io::IO, t::OneHot) 
    print(io, instancestr(t) * "[" * repr(t.con) * "|")
    if length(t.tbl) == 0
        print(io, "not initialized!]")
    else
        customshow(io, t.tbl, false)
    end
    print(io, "]")
end

function Base.getindex(t::OneHot, col)
    checkfit(t)
    icol = first(inids(t)) # only possible input id
    fsrc = intrn(t, icol)[icol] # get CEF for this column
    (i)->func(t, col, fsrc(i)) # compute output, if generated the the col is one of icol_VAL1, icol_VAL2, ...
end

function func(t::OneHot, col, data)
    col ∉ keys(t.tbl) && ArgumentError("unknown output column $col")
    T = outtype(t, col)
    T = T == Nothing ? Bool : T
    T.(data .== t.tbl[col])
end

function fit!(t::OneHot, data, smap)
    icol = first(inids(t))
    if !isfit(t)
        unq = sort(unique(data[icol]))
        for val in unq
            newcol = Symbol(string(icol) * "_" * string(val))
            t.tbl[newcol] = val
            push!(t.con, newcol, Out())
        end
    end
    OrderedDict(col => func(t, col, data[icol]) for col in outids(t))
end

# ----- One Cold ----
# -------------------
struct OneCold <: ColTrans
    id::Int
    con::Connections
    tbl::OrderedDict{Symbol,Any} # input column id to output value
    function OneCold(con::Connections, tbl::OrderedDict{Symbol,Any})
        con.maxouts > 1 && throw(ArgumentError("OneCold: only single output allowed!"))
        new(_newid(), con, tbl)
    end
end

"""
    onecold(ins, outid::Symbol, tbl::OrderedDict{Symbol,Any})

Define one cold transform.

# Examples
The following transform finds maximum in `itrans` columns `:I1` and `:I2` and outputs
either `"a"` (for maximum in `:I1`) or `2` (for maximum in `:I2`) to column `:O`.

```julia
onecold([:I1 => itrans, :I2 => itrans], :O, OrderedDict(:I1 => "a", :I2 => 2))
```
"""
function onecold(ins, outid::Symbol, tbl::OrderedDict{Symbol,Any})
    OneCold(Connections(ins, [outid => Out()]; maxouts = 1), tbl)
end

function Base.show(io::IO, t::OneCold)
    print(io, instancestr(t) * "[" * repr(t.con) * "|$(outids(t)[1]): ")
    customshow(io, t.tbl)
    print(io, "]")
end

function Base.getindex(t::OneCold, col)
    col != first(outids(t)) && error("OneCold: unknown output column $(col) != $(first(outids(t)))")
    fsrcs = [intrn(t, icol)[icol] for icol in inids(t)]
    (i)->func(t, [fsrc(i) for fsrc in fsrcs])
end

function func(t::OneCold, data)
    vals = [t.tbl[icol] for icol in inids(t)]
    [vals[argmax(obs)] for obs in zip(data...)]
end

function fit!(t::OneCold, data, smap)
    OrderedDict(col => func(t, data) for col in outids(t))
end

# ----- ReplaceVals -----
# -----------------------
struct ReplaceVals <: ColTrans
    id::Int
    con::Connections
    tbl::OrderedDict
    function ReplaceVals(con::Connections, tbl::OrderedDict)
        con.maxins > 1 && throw(ArgumentError("ReplaceVals: only single input allowed!"))
        con.maxouts > 1 && throw(ArgumentError("ReplaceVals: only single output allowed!"))
        new(_newid(), con, tbl)
    end
end

"""
    replacevals(in, tblpairs::Pair...)

Replace values of a column.

Input `in` is given as: `in = icol => itrans`, where `icol` is the output column of the input 
transform `itrans`. Rewriting rules are given as `Pair`s. Nonlisted values are kept untouched.

# Examples

The following transforms `itrans` column `:C` values `X` and `Z` to the respective lowercase letters.
```julia
replacevals(:C => itrans, "X" => "x", "Z" => "z")
```
"""
function replacevals(in, tblpairs::Pair...)
    ReplaceVals(Connections([in], [first(in) => Out()]; maxins = 1, maxouts = 1), OrderedDict(tblpairs...))
end

function Base.show(io::IO, t::ReplaceVals)
    print(io, instancestr(t) * "[" * repr(t.con) * "|")
    customshow(io, t.tbl, false)
    print(io, "]")
end

function func(t::ReplaceVals, data)
    map(e->get(t.tbl, e, e), data)
end

function Base.getindex(t::ReplaceVals, col)
    fsrc = intrn(t, col)[col] # same name of icol and col
    (i)->func(t, fsrc(i))
end

function fit!(t::ReplaceVals, data, smap)
    OrderedDict(col => func(t, data[col]) for col in outids(t))
end

# ----- Standardize ----
# ----------------------
mutable struct Standardize <: ColTrans
    id::Int
    con::Connections
    μ::AbstractFloat
    σ::AbstractFloat
    function Standardize(con::Connections, μ::AbstractFloat, σ::AbstractFloat)
        con.maxins > 1 && throw(ArgumentError("Standardize: only single input allowed!"))
        con.maxouts > 1 && throw(ArgumentError("Standardize: only single output allowed!"))
        isnan(σ)  || σ > 0 || throw(ArgumentError("Standardize: σ ≦ 0!: $(σ)"))
        new(_newid(), con, μ, σ)
    end
end

"""
    standardize(in, μ::AbstractFloat = NaN, σ::AbstractFloat = NaN)

Standardize a column.

Default output type is `Float64`.

The following standardizes `itrans` column `:B`.
```julia
standardize(:B => itrans)
```
"""
function standardize(in, μ::AbstractFloat = NaN, σ::AbstractFloat = NaN)
    Standardize(Connections([in], [first(in) => Out()]; maxins = 1, maxouts = 1), μ, σ)
end

isfit(t::Standardize) = !(isnan(t.μ) || isnan(t.σ))

Base.show(io::IO, t::Standardize) = print(io, instancestr(t) * "[" * repr(t.con) * "::{$(intype(t, inids(t)[1]))}|μ = $(t.μ) , σ = $(t.σ)::{$(outtype(t, outids(t)[1]))}]")

function func(t::Standardize, col, data)
    T = outtype(t, col)
    T = T == Nothing ? Float64 : T
    try
        return T.((data .- t.μ) ./ t.σ)
    catch e
        e isa InexactError && return T.(round.((data .- t.μ) ./ t.σ))
        rethrow(e)
    end
end

function Base.getindex(t::Standardize, col)
    checkfit(t)
    fsrc = intrn(t, col)[col] # col is same as icol
    (i)->func(t, col, fsrc(i))
end

function fit!(t::Standardize, data, smap)
    icol = first(inids(t))
    din = data[icol]
    if !isfit(t)
        t.μ = mean(din)      
        t.σ = std(din)
        if t.σ <= 0
            t.σ = 1
        end
    end
    OrderedDict(col => func(t, col, data[icol]) for col in outids(t))
end

# ----- INVERSE -----
# -------------------
_inverse(o::DataSource, outs::Vector{IdOut}) = MergeCols(Connections(IdInp[], outs))

_inverse(o::OneHot, outs::Vector{IdOut}) = OneCold(Connections(IdInp[], outs; maxouts = 1), copy(o.tbl))

_inverse(o::OneCold, outs::Vector{IdOut}) = OneHot(Connections(IdInp[], outs; maxins = 1), copy(o.tbl))

function _inverse(o::ReplaceVals, outs::Vector{IdOut})
    length(values(o.tbl)) > length(unique(values(o.tbl))) && 
        throw(ArgumentError("cannot invert ReplaceVals $(idstr(o)), rules: $(o.tbl)"))
    revtbl = OrderedDict(v => k for (k, v) in o.tbl)
    ReplaceVals(Connections(IdInp[], outs; maxins = 1, maxouts = 1), revtbl)
end

_inverse(o::Standardize, outs::Vector{IdOut}) = Standardize(Connections(IdInp[], outs; maxins = 1, maxouts = 1), -o.μ / o.σ, 1 / o.σ)

function _invhelper(o::ColTrans, oi::OrderedDict{ColTrans,ColTrans})
    if o ∉ keys(oi)
        outs = IdOut[id => Out(intype(o, id)) for id in inids(o)]
        oi[o] = _inverse(o, outs)
    end
    for pid in inids(o)
        ptrn = intrn(o, pid)
        _invhelper(ptrn, oi)
        icon = oi[ptrn].con
        otype = ptrn.con.outs[pid].type
        pid ∈ keys(icon.ins) || push!(icon, pid, Inp(oi[o], otype)) 
    end
end

"""
    invert(odst::ColTrans, isrc::ColTrans)

Invert original transformation `odst`, the source for the inverted transform will be `isrc`.
"""
function invert(odst::ColTrans, isrc::ColTrans)
    if !(odst isa MergeCols)
        odst = mergecols(odst)
    end
    check(odst)
    checkfit(odst)
    osrc = getdatasource(odst)

    isrc.con.outs = OrderedDict(id => Out(intype(odst, id)) for id in outids(odst))
    oi = OrderedDict{ColTrans,ColTrans}(odst => isrc)
    _invhelper(odst, oi)
    idst = oi[osrc]
    idst.con.outs = OrderedDict(id => Out(intype(idst, id)) for id in inids(idst))
    idst
end

"""
    invert(odst::ColTrans; sid=:default)

Invert original transformation `odst`, feed it with a new `datasource` with given source id `sid` 
based on `odst outputs`.
"""
function invert(odst::ColTrans; sid=:default)
    invert(odst, datasource(OrderedDict{Symbol,Symbol}(col => sid for col in outids(odst))))
end


"""
    invtransform(t::ColTrans, smap, type::Type)

Invert original transformation `odst` and immediatelly run `transform`.

Shortcut for `transform(invert(t), smap, type)`.
"""
invtransform(t::ColTrans, smap, type::Type) = transform(invert(t), smap, type)


# ----- COLLECTING/SORTING -----
# ------------------------------
function collecttrans(root::ColTrans)
    set = OrderedSet{ColTrans}()
    function helper(t::ColTrans)
        t ∈ set && return
        push!(set, t)
        for i in intrns(t)
            helper(i)
        end
    end
    helper(root)
    set
end

function toposort(root::ColTrans)
    nodes = collect(collecttrans(root))
    toposort(nodes, v->OrderedSet(intrns(v)))
end

# ----- SERIALIZATION -----
# -------------------------

function JSON.lower(t::Connections) 
    ins = OrderedDict(id => OrderedDict("from" => string(v.trn.id), "type" => v.type) for (id, v) in t.ins)
    outs = OrderedDict(id => v.type for (id, v) in t.outs)
    OrderedDict("ins" => ins, "outs" => outs, "maxins" => t.maxins, "maxouts" => t.maxouts)
end
JSON.lower(t::DataSource) = OrderedDict("type" => "DataSource", "con" => t.con, "colsid" => t.sids)
JSON.lower(t::OneHot) = OrderedDict("type" => "OneHot", "con" => t.con, "tbl" => t.tbl, "tbltype" => typeof(t.tbl))
JSON.lower(t::OneCold) = OrderedDict("type" => "OneCold", "con" => t.con, "tbl" => t.tbl, "tbltype" => typeof(t.tbl))
JSON.lower(t::ReplaceVals) = OrderedDict("type" => "ReplaceVals", "con" => t.con, "tbl" => t.tbl, "tbltype" => typeof(t.tbl))
JSON.lower(t::Standardize) = OrderedDict("type" => "Standardize", "con" => t.con, "μ" => t.μ, "σ" => t.σ)

function JSON.lower(ddst::MergeCols)
    ts = toposort(ddst)
    tdict = OrderedDict()
    for t in ts
        if t == ddst
            tdict[t.id] = OrderedDict("type" => "MergeCols", "con" => t.con)
        else
            tdict[t.id] = JSON.lower(t)
        end
    end
    tdict
end

Base.parse(::Type{String}, s::String) = s
Base.parse(::Type{Symbol}, s::String) = Symbol(s)

function _converttable(tbl, tbltype)
    T = eval(Meta.parse(tbltype))
    T(parse(T.parameters[1], k) => v for (k, v) in tbl)
end

deserializetype(t::Type{DataSource}, p::OrderedDict, newnodes) = datasource(OrderedDict(Symbol(k) => v for (k, v) in p["colsid"]))
deserializetype(t::Type{OneHot}, p::OrderedDict, newnodes) = OneHot(deserializecons(p["con"], newnodes), 
    _converttable(p["tbl"], p["tbltype"]))
deserializetype(t::Type{OneCold}, p::OrderedDict, newnodes) = OneCold(deserializecons(p["con"], newnodes),
    OrderedDict{Symbol,Any}(Symbol(k) => v for (k, v) in p["tbl"]))
deserializetype(t::Type{ReplaceVals}, p::OrderedDict, newnodes) = ReplaceVals(deserializecons(p["con"], newnodes), 
    _converttable(p["tbl"], p["tbltype"]))
deserializetype(t::Type{Standardize}, p::OrderedDict, newnodes) = Standardize(deserializecons(p["con"], newnodes), 
    p["μ"], p["σ"])
deserializetype(t::Type{MergeCols}, p::OrderedDict, newnodes) = MergeCols(deserializecons(p["con"], newnodes))

function deserializecons(d, newnodes)
    ins = OrderedDict{Symbol,Inp}(Symbol(id) => Inp(newnodes[info["from"]], 
        eval(Meta.parse(info["type"]))) for (id, info) in d["ins"])
    outs = OrderedDict{Symbol,Out}(Symbol(id) => Out(eval(Meta.parse(type))) for (id, type) in d["outs"])
    #TODO fix maxins/maxouts in corresponding ColTrans
    Connections(ins, outs; maxins = d["maxins"], maxouts = d["maxouts"]) 
end

function deserializehelper(dict, newnodes)
    (dict isa OrderedDict && "type" ∈ keys(dict)) || return dict
    t = eval(Meta.parse(dict["type"]))
    deserializetype(t, dict, newnodes)
end

function deserialize(dict)
    function predf(v)
        "in" ∉ keys(last(v)) && return []
        string.(unique(values(last(v)["in"]))) # JSON keys as strings
    end
    
    orgnodes = toposort(dict, predf, first)
    newnodes = OrderedDict{String,ColTrans}()
    
    dst = nothing
    for n in orgnodes
        newnodes[n] = deserializehelper(dict[n], newnodes)
        if newnodes[n] isa MergeCols
            dst !== nothing && error("multiple MergeCols!")
            dst = newnodes[n]
        end
    end
    dst === nothing && error("no MergeCols in serialized data!")
    dst
end


# ----- BUILDERS -----
# --------------------
#TODO make these work for more than DataFrames!

"""
    nominal(df::AbstractDataFrame; maxunique::Union{Int,Nothing} = nothing)

Select nominal attributes from DataFrame.

Returns DataFrame column ids. These are ids corresponding to other than Real column types. Even Real types can be included 
if less then or equal to `maxunique` values are present in data.     
"""
function nominal(df::AbstractDataFrame; maxunique::Union{Int,Nothing} = nothing)
    ret = OrderedSet{Symbol}()
    for (name, type) in zip(names(df), eltypes(df))
        if type <: Real
            if maxunique != nothing && length(unique(df[name])) <= maxunique
                push!(ret, name)
            end
        else
            push!(ret, name) 
        end
    end
    ret
end

"""
    tonumerical(df::AbstractDataFrame; nominal = nothing, standardize = true, sid = :default)

Automaticaly create a DataFrame transform to preprocess data as expected by many ML models such as neural networks. 

All columns are transformed to numerical types. One may select columns (the `nominal` parameter) which should be 
transformed using one hot encoding or `nominal()` is used to determine them automatically. All outputs are standardized 
by default - this behavior can be altered using the `standardize` parameter. 
"""
function tonumerical(df::AbstractDataFrame; nominal = nothing, standardize = true, sid = :default)
    if nominal === nothing
        nominal = Transforms.nominal(df)
    end
    nominal = OrderedSet(nominal)
    sid = Symbol(sid)
    smap = Dict(sid => df)

    src = datasource(df; sid=sid)
    transforms = Pair{Symbol,ColTrans}[]
    for id in outids(src)
        if id ∈ nominal
            oh = onehot(id => src)
            fit!(oh, smap)
            for outid in outids(oh)
                push!(transforms, outid => oh)
            end
        else
            push!(transforms, id => src)
        end
    end
    if standardize
        stransforms = Pair{Symbol,ColTrans}[]
        for (id, trn) in transforms
            st = Transforms.standardize(id => trn)
            fit!(st, smap)
            push!(stransforms, id => st)
        end
        transforms = stransforms
    end
    me = mergecols(transforms...)
end

"""
    copytransform(df::AbstractDataFrame, sid = :default)

Create identity transform for a DataFrame.
"""
function copytransform(df::AbstractDataFrame, sid = :default)
    sid = Symbol(sid)
    src = datasource(df; sid=sid)
    mergecols(src)
end

end