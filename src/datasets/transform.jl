module Transforms
export ColTrans, datasource, mergecols, onehot, onecold, replacevals, standardize, fit!, transform, invert, collecttrans, 
    deserialize, tonumerical, copytransform, inids, outids, intype, outtype, getdatasource, invtransform

using DataFrames
using DataStructures: counter, OrderedDict, OrderedSet
using JSON
using MLDataUtils
using Statistics

import ..FluxGoodies: toposort, dview, coldim, obsdim, ncols, nobs

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
abstract type ColTrans end

# ----- Connections -------
# -------------------------

mutable struct Inp
    trn::ColTrans
    type::Type
    Inp(trn, type = Nothing) = new(trn, type)
end

mutable struct Out
    type::Type
    Out(type = Nothing) = new(type)
end

const IdInp = Pair{Symbol,Inp}
const IdOut = Pair{Symbol,Out}

mutable struct Connections
    ins::OrderedDict{Symbol,Inp}
    outs::OrderedDict{Symbol,Out}
    maxins::Int
    maxouts::Int
    function Connections(ins, outs, maxins, maxouts)
        length(ins) > maxins && throw(ArgumentError("too many inputs $(length(ins)) > $(maxins)!"))
        length(outs) > maxouts && throw(ArgumentError("too many outputs $(length(outs)) > $(maxouts)!"))
        new(ins, outs, maxins, maxouts)
    end
end

Connections(ins::Vector{IdInp} = IdInp[], outs::Vector{IdOut} = IdOut[]; maxins = typemax(Int), maxouts = typemax(Int)) = 
    Connections(OrderedDict(ins), OrderedDict(outs), maxins, maxouts)

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

fcounter = counter(Int) 

function fit!(c::Connections, t::ColTrans, smap)
    smap = _smap_process(smap)
    #TODO optimize! Now performs for multiple times!
    data = OrderedDict()
    cache = Dict{ColTrans,Any}()
    for (id, input) in c.ins
        if input.trn ∉ keys(cache)
            cache[input.trn] = fit!(input.trn, smap)
        end
        data[id] = cache[input.trn][id]
    end
    ret = fit!(t, data, smap)
    for (i, d) in zip(values(c.ins), values(data))
        if i.type == Nothing
            i.type = eltype(d)
        end
    end
    for (o, r) in zip(values(c.outs), values(ret))
        if o.type == Nothing
            o.type = eltype(r)
        end
    end
    ret
end

inids(c::Connections) = collect(keys(c.ins))
intrns(c::Connections) = [v.trn for v in values(c.ins)]

outids(c::Connections) = collect(keys(c.outs))

# Base.collect(c::Connections) = collect(c.in)
# Base.iterate(c::Connections) = iterate(c.in)
# Base.iterate(c::Connections, state) = iterate(c.in, state)
# Base.length(c::Connections) = length(c.in)

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

isfit(t::ColTrans) = true

isfitrec(t::ColTrans) = all((isfit(t), isfitrec.(intrns(t))...))

checkfit(t::ColTrans) = isfit(t) || error("$(t) not initialized, use fit!")
checkfitrec(t::ColTrans) = isfitrec(t) || error("$(t) not initialized, use fit!")

function check(t::ColTrans)
    ts = typeof.(collecttrans(t))
    cnts = counter(ts)
    if cnts[DataSource] > 1
        error("More than one DataSource!")
    end
end

fit!(t::ColTrans, smap) = fit!(t.con, t, smap)

function _checktransform(t::ColTrans, smap)
    checkfit(t)
    length(smap) > 0 || error("no source!")
    ns = [nobs(s) for s in values(smap)]
    minimum(ns) != maximum(ns) && throw(ArgumentError("source sizes not same; $ns"))
end

function transform(t::ColTrans, smap, ::Type{DataFrame})
    smap = _smap_process(smap)
    _checktransform(t, smap)
    os = outids(t)
    data = [t[o](smap) for o in os]
    DataFrame(data, os)
end

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
    con::Connections
    cols::OrderedDict{Symbol,Function}
    sids::OrderedDict{Symbol,Symbol}
    function DataSource(con::Connections, cols::OrderedDict{Symbol,Function}, sids::OrderedDict{Symbol,Symbol})
        con.maxins != 0 && throw(ArgumentError("maxins != 0!"))
        new(_newid(), con, cols, sids)
    end
end

Base.getindex(t::DataSource, col::Symbol) = t.cols[col]

function Base.show(io::IO, t::DataSource) 
    print(io, instancestr(t) * "[" * join(["$col($(t.sids[col]))" for col in keys(t.cols)], ", ") * "]")
end

datasource(s::AbstractDataFrame; sid::Symbol=:default) = datasource(s, names(s); sid=sid)

function datasource(s, names::Vector{Symbol}; sid::Symbol=:default)
    ncols(s) != length(names) && error("source $(typeof(s)) size $(size(s)) does not match the number of col names ($(length(names)))")
    colsid = OrderedDict{Symbol,Symbol}(col => sid for col in names)
    datasource(colsid)
end

function datasource(colsid::AbstractDict)
    con = Connections(IdInp[], [col => Out() for col in keys(colsid)]; maxins = 0)
    colsid = OrderedDict{Symbol,Symbol}(Symbol(col) => Symbol(sid) for (col, sid) in colsid)
    colfunc = OrderedDict{Symbol,Function}(col => ((i)->dview(i[sid], :, colidx)) for (colidx, (col, sid)) in enumerate(colsid))
    DataSource(con, colfunc, colsid)
end

function fit!(t::DataSource, data, smap)
    OrderedDict(col => t[col](smap) for col in keys(t.con.outs))
end

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

function mergecols(ins::IdInp...)
    outs = IdOut[first(i) => Out() for i in ins]
    MergeCols(Connections(IdInp[i for i in ins], outs))
end

function mergecols(ins::Pair{Symbol,<:ColTrans}...)
    mergecols([first(i) => Inp(last(i)) for i in ins]...)
end

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

function Base.getindex(t::MergeCols, col)
    fsrc = intrn(t, col)[col]
    (i)->func(t, col, OrderedDict(col => fsrc(i)))
end

function func(t::MergeCols, col, data)
    push!(fcounter, t.id)
    data[col]
end

function fit!(t::MergeCols, data, smap)
    OrderedDict(col => func(t, col, data) for col in outids(t))
end

# ----- One Hot -----
# -------------------
struct OneHot <: ColTrans
    id::Int
    con::Connections
    tbl::OrderedDict{Symbol,Any} # column id to value
    function OneHot(con::Connections, tbl::OrderedDict{Symbol,Any})
        con.maxins > 1 && throw(ArgumentError("OneHot: only single input allowed!"))
        new(_newid(), con, tbl)
    end
end

function onehot(tbl::OrderedDict{Symbol,Any} = OrderedDict{Symbol,Any}())
    OneHot(Connections(IdInp[], IdOut[]; maxins = 1), tbl)
end

function onehot(in::Pair{Symbol,<:ColTrans}, ps::Pair...)
    onehot(first(in) => Inp(last(in)))
end

function onehot(in::IdInp, ps::Pair...)
    OneHot(Connections([in], IdOut[]; maxins = 1), OrderedDict{Symbol,Any}(ps...))
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
    inid = inids(t)[1]
    fsrc = intrn(t, inid)[inid]
    (i)->func(t, col, fsrc(i))
end

function func(t::OneHot, col, data)
    col ∉ keys(t.tbl) && ArgumentError("unknown column $col")
    push!(fcounter, t.id)
    T = outtype(t, col)
    T = T == Nothing ? Bool : T
    T.(data .== t.tbl[col])
end

function fit!(t::OneHot, data, smap)
    inid = inids(t)[1]
    if !isfit(t)
        unq = sort(unique(data[inid]))
        for val in unq
            newcol = Symbol(string(inid) * "_" * string(val))
            t.tbl[newcol] = val
            push!(t.con, newcol, Out())
        end
    end
    OrderedDict(col => func(t, col, data[inid]) for col in outids(t))
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

function onecold(outid::Symbol, tbl::OrderedDict{Symbol,Any})
    onecold(IdInp[], outid, tbl)
end

function onecold(ins::Vector{IdInp}, outid::Symbol, tbl::OrderedDict{Symbol,Any})
    OneCold(Connections(ins, [outid => Out()]; maxouts = 1), tbl)
end

function onecold(ins::Vector{<:Pair{Symbol,<:ColTrans}}, outid::Symbol, tbl::OrderedDict{Symbol,Any})
    onecold([first(i) => Inp(last(i)) for i in ins], outid, tbl)
end

function Base.show(io::IO, t::OneCold)
    print(io, instancestr(t) * "[" * repr(t.con) * "|$(outids(t)[1]): ")
    customshow(io, t.tbl)
    print(io, "]")
end

function Base.getindex(t::OneCold, col)
    col != outids(t)[1] && error("OneCold: unknown output column $(col) != $(outids(t)[1])")
    fsrcs = [intrn(t, inid)[inid] for inid in inids(t)]
    (i)->func(t, [fsrc(i) for fsrc in fsrcs])
end

function func(t::OneCold, data)
    push!(fcounter, t.id)
    vals = [t.tbl[i] for i in inids(t)]
    [vals[argmax(obs)] for obs in zip(data...)]
end

function fit!(t::OneCold, data, smap)
    OrderedDict(col => func(t, data) for col in keys(t.con.outs))
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

function replacevals(in::IdInp, ps::Pair...)
    ReplaceVals(Connections([in], [first(in) => Out()]; maxins = 1, maxouts = 1), OrderedDict(ps...))
end

function replacevals(in::Pair{Symbol,<:ColTrans}, ps::Pair...)
    replacevals(first(in) => Inp(last(in)), ps...)
end

function Base.show(io::IO, t::ReplaceVals)
    print(io, instancestr(t) * "[" * repr(t.con) * "|")
    customshow(io, t.tbl, false)
    print(io, "]")
end

function func(t::ReplaceVals, data)
    push!(fcounter, t.id)
    map(e->get(t.tbl, e, e), data)
end

function Base.getindex(t::ReplaceVals, col)
    fsrc = intrn(t, col)[col]
    (i)->func(t, fsrc(i))
end

function fit!(t::ReplaceVals, data, smap)
    OrderedDict(col => func(t, data[col]) for col in keys(t.con.outs))
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

function standardize(in::IdInp, μ::AbstractFloat = NaN, σ::AbstractFloat = NaN)
    Standardize(Connections([in], [first(in) => Out()]; maxins = 1, maxouts = 1), μ, σ)
end

function standardize(in::Pair{Symbol,<:ColTrans}, μ::AbstractFloat = NaN, σ::AbstractFloat = NaN)
    standardize(first(in) => Inp(last(in)), μ, σ)
end

isfit(t::Standardize) = !(isnan(t.μ) || isnan(t.σ))

Base.show(io::IO, t::Standardize) = print(io, instancestr(t) * "[" * repr(t.con) * "::{$(intype(t, inids(t)[1]))}|μ = $(t.μ) , σ = $(t.σ)::{$(outtype(t, outids(t)[1]))}]")

function func(t::Standardize, col, data)
    push!(fcounter, t.id)
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
    fsrc = intrn(t, col)[col]
    (i)->func(t, col, fsrc(i))
end

function fit!(t::Standardize, data, smap)
    inid = inids(t)[1]
    din = data[inid]
    if !isfit(t)
        t.μ = mean(din)      
        t.σ = std(din)
        if t.σ <= 0
            t.σ = 1
        end
    end
    OrderedDict(col => func(t, col, data[inid]) for col in outids(t))
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

function invert(odst::ColTrans; sid=:default)
    invert(odst, datasource(OrderedDict{Symbol,Symbol}(col => sid for col in outids(odst))))
end

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
    toposort(nodes, identity, v->OrderedSet(intrns(v)))
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
    Connections(ins, outs, d["maxins"], d["maxouts"]) 
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
    
    orgnodes = toposort(dict, first, predf)
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

function copytransform(df::AbstractDataFrame, sid = :default)
    sid = Symbol(sid)
    src = datasource(df; sid=sid)
    mergecols(src)
end

end