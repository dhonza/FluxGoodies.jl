using DataFrames
using DataStructures: OrderedDict, OrderedSet
using JSON
using Statistics


#TODO should be typed?
struct Column{T}
    name::Symbol
    # ctype::T
end

function columns(df::AbstractDataFrame)
    [Column{ctype}(name) for (name, ctype) in zip(names(df), eltypes(df))]
end

name(c::Column) = c.name
ctype(c::Column{T}) where T = T

abstract type ColumnTransform end

inplace(t::ColumnTransform) = false

struct CopyColumnTransform <: ColumnTransform
    col::Column
end

function transform!(t::CopyColumnTransform, src::Tabular, dst::Tabular; opts...)
    view(dst, :, 1) .= view(src, :, 1)
end

function invtransform!(t::CopyColumnTransform, src::Tabular, dst::Tabular; opts...)
    view(src, :, 1) .= view(dst, :, 1)
end

srccols(t::CopyColumnTransform) = [t.col]

dstcols(t::CopyColumnTransform) = [t.col]

inplace(t::CopyColumnTransform) = true

struct OneHotColumnTransform{T} <: ColumnTransform
    col::Column{T}
    vals::Vector{T}

    function OneHotColumnTransform{T}(col::Column{T}, data::AbstractVector{T}) where T
        new{T}(col, Vector{T}(sort(unique(data))))
    end
end

srccols(t::OneHotColumnTransform) = [t.col]

dstcols(t::OneHotColumnTransform) = [Column{Int}(String(t.col.name) * "_" * String(v) |> Symbol) for v in t.vals]

# TODO remove S and D types, src and dst must be Matrices/Tables i.e., 2D
function transform!(t::OneHotColumnTransform, src::Tabular, dst::Tabular; opts...)
    d = Dict(v => i for (i, v) in enumerate(t.vals))
    for i in 1:size(dst, 2)
        view(dst, :, i) .= 0
    end
    for i in 1:size(src, 1)
        j = d[src[i, 1]]
        dst[i, j] = 1
    end
end

function invtransform!(t::OneHotColumnTransform, src::Tabular, dst::Tabular; opts...)
    for i in 1:size(src, 1)
        view(src, i, 1) .= t.vals[argmax(Vector(dst[i, :]))]
    end
end

mutable struct StandardizeColumnTransform{T <: Real} <: ColumnTransform
    col::Column{T}
    μ::Union{T, Nothing}
    σ::Union{T, Nothing}
end

function StandardizeColumnTransform(col::Column{T}) where T <: Real
    StandardizeColumnTransform(col, nothing, nothing)   
end

function StandardizeColumnTransform(col::Column{T}, data::AbstractVector{U}) where T <: Real where U <: Real
    μ, σ = mean(data), std(data)
    if σ == 0
        σ = 1
    end
    StandardizeColumnTransform(col, μ, σ)
end

srccols(t::StandardizeColumnTransform) = [t.col]

dstcols(t::StandardizeColumnTransform) = [t.col]

inplace(t::StandardizeColumnTransform) = true

function transform!(t::StandardizeColumnTransform, src::Tabular, dst::Tabular; opts...)
    if get(opts, :fit, false)
        temp = StandardizeColumnTransform(t.col, src[:, 1])
        t.μ, t.σ = temp.μ, temp.σ
    end
    (t.μ == nothing || t.σ == nothing) && error("StandardizeColumnTransform not fit!")
    view(dst, :, 1) .= (view(src, :, 1) .- t.μ) ./ t.σ
end

function invtransform!(t::StandardizeColumnTransform, src::Tabular, dst::Tabular; opts...)
    view(src, :, 1) .= (t.σ .* view(dst, :, 1)) .+ t.μ
end

struct ParallelColumnTransform <: ColumnTransform
    transforms::Vector{ColumnTransform}
end

srccols(t::ParallelColumnTransform) = vcat([srccols(t) for t in t.transforms]...)

dstcols(t::ParallelColumnTransform) = vcat([dstcols(t) for t in t.transforms]...)

inplace(t::ParallelColumnTransform) = all(inplace.(t.transforms))

function transform!(t::ParallelColumnTransform, 
        src::Tabular, 
        dst::Tabular; opts...)
    offset = 1
    for (i, trn) in enumerate(t.transforms)
        ncols = length(dstcols(trn))
        vsrc = view(src, :, i:i)
        vdst = view(dst, :, offset:offset + ncols - 1)
        transform!(trn, vsrc, vdst; opts...)
        offset += ncols
    end
end

function invtransform!(t::ParallelColumnTransform, src::Tabular, dst::Tabular; opts...)
    offset = 1
    for (i, trn) in enumerate(t.transforms)
        ncols = length(dstcols(trn))
        vsrc = view(src, :, i:i)
        vdst = view(dst, :, offset:offset + ncols - 1)
        invtransform!(trn, vsrc, vdst; opts...)
        offset += ncols
    end
end

struct ChainColumnTransform <: ColumnTransform
    transforms::Vector{ColumnTransform}
end

srccols(t::ChainColumnTransform) = srccols(t.transforms[1])

dstcols(t::ChainColumnTransform) = srccols(t.transforms[end])

inplace(t::ChainColumnTransform) = all(inplace.(t.transforms))

function transform!(t::ChainColumnTransform, src::Tabular, dst::Tabular; opts...)
    #TODO no allocation for inplace = false transforms
    for trans in t.transforms[1:end-1]
        src = transform(trans, src; opts...)
    end
    transform!(t.transforms[end], src, dst; opts...)
end

function invtransform!(t::ChainColumnTransform, src::Tabular, dst::Tabular; opts...)
    #TODO better memory management (use inplace)
    for i in length(t.transforms):-1:2
        trans = t.transforms[i]
        dst = invtransform(trans, dst; opts...)
    end
    invtransform!(t.transforms[1], src, dst; opts...)
end

# ------ COMMON

function allocate(tp::Type{Matrix{T}}, t::ColumnTransform, len::Int, fcols=dstcols) where T
    tp(undef, length(fcols(t)), len)' # generate transposed matrix
end

function allocate(::Type{<:AbstractDataFrame}, t::ColumnTransform, len::Int, fcols=dstcols) where T
#     tp(undef, length(dstcols(t)), len)
    cols = fcols(t)
    DataFrame(ctype.(cols), name.(cols), len)
end

function ann_transform(src::AbstractDataFrame)
    cols = columns(src)
    transforms = ColumnTransform[]
    for col in cols
        if ctype(col) <: Real
            push!(transforms, CopyColumnTransform(col))
        elseif ctype(col) <: AbstractString
            t = OneHotColumnTransform{ctype(col)}(col, src[name(col)])
            push!(transforms, t)
        else
            error("unknown type: $(ctype(col))")
        end
    end
    ParallelColumnTransform(transforms)
end

function transform(t::ColumnTransform, 
    src::Tabular, 
    dsttype::Type; opts...)
    dst = allocate(dsttype, t, size(src, 1))
    transform!(t, src, dst; opts...)
    dst
end

transform(t::ColumnTransform, src::Tabular; opts...) = transform(t, src, typeof(src); opts...)

function invtransform(t::ColumnTransform, 
    dst::Tabular, 
    srctype::Type; opts...)
    src = allocate(srctype, t, size(dst, 1), srccols)
    invtransform!(t, src, dst; opts...)
    src
end

invtransform(t::ColumnTransform, dst::Tabular; opts...) = invtransform(t, dst, typeof(dst); opts...)

function fit!(t::ColumnTransform, src::Tabular)
    #TODO better memory management (use inplace)
    transform(t, src; fit=true)
end

# ------ SERIALIZATION
JSON.lower(t::Column{T}) where T = OrderedDict("type" => "Column{$T}", "name" => t.name)
JSON.lower(t::CopyColumnTransform) = OrderedDict("type" => "CopyColumnTransform", "col" => t.col)
JSON.lower(t::OneHotColumnTransform{T}) where T = OrderedDict("type" => "OneHotColumnTransform{$T}", 
        "col" => t.col, "vals" => t.vals)
JSON.lower(t::StandardizeColumnTransform) = OrderedDict("type" => "StandardizeColumnTransform", 
        "col" => t.col, "mu" => t.μ, "sigma" => t.σ)
JSON.lower(t::ParallelColumnTransform) = OrderedDict("type" => "ParallelColumnTransform", "transforms" => t.transforms)

function deserializetype(t::Type, params::OrderedDict)
    t(deserialize.(values(params))...)
end

deserializetype(t::Type{Column{T}}, params::OrderedDict) where T = t(Symbol(params["name"]))

deserializetype(t::Type{OneHotColumnTransform{T}}, params::OrderedDict) where T =  t(deserialize(params["col"]), String.(params["vals"]))

deserializetype(t::Type{ParallelColumnTransform}, params::OrderedDict) = t(deserialize.(values(params["transforms"])))

function deserialize(d)
    (d isa OrderedDict && "type" ∈ keys(d)) || return d
    T = eval(Meta.parse(d["type"]))
    d = copy(d)
    delete!(d, "type")
    deserializetype(T, d)
end

function dump_transform(file, transform::ColumnTransform)
    open(f->write(f, JSON.json(transform, 3)), file, "w")
end

function load_transform(file)::ColumnTransform
    d = open(f->JSON.parse(f, dicttype = OrderedDict), file, "r")
    deserialize(d)
end