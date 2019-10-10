module Utils

using DataFrames
using DataStructures: OrderedDict

export dview, toposort, coldim, ncols

# obsdim(::AbstractDataFrame) = 1
# obsdim(::AbstractMatrix) = 2

coldim(::AbstractDataFrame) = 2
coldim(::AbstractMatrix) = 1

# nobs(d) = size(d, obsdim(d))

"""
    ncols(d)

Return the number of columns, i.e., features of `d`.
"""
ncols(d) = size(d, coldim(d))

dview(d::AbstractDataFrame, obs, cols) = view(d, obs, cols)
dview(d::AbstractMatrix, obs, cols) = view(d, cols, obs)

"""
    isapprox(dfa::AbstractDataFrame, dfb::AbstractDataFrame)

Check if two `DataFrame` instances are approximatelly equal.
"""
function Base.isapprox(dfa::AbstractDataFrame, dfb::AbstractDataFrame)
    function isapproxifexists(a, b)
        try
            return a ≈ b
        catch e
            e isa MethodError && return a == b
            rethrow(e)
        end
    end
    names(dfa) == names(dfb) && eltypes(dfa) == eltypes(dfb) && size(dfa) == size(dfb) && 
        all(isapproxifexists(dfa[!,n], dfb[!,n]) for n in names(dfa))
end

"""
    toposort(nodes, predf, nodef = identity)

Sort graph nodes topologically.

Provide `node` an iterator over node objects, `predf` a function giving predecessor of a node and 
a `nodef` function optionally applied to the node in the result. [See](https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search).
   
# Examples
```@meta
DocTestSetup = quote
    using DataStructures: OrderedDict
    using FluxGoodies: toposort
end
```

```jldoctest
julia> nodes = OrderedDict(2 => [1], 5 => [1, 3, 4], 4 => [1, 2], 6 => [5], 1 => [], 3 => [1], 20 => [5, 10], 10 => [1]);

julia> toposort(nodes, v -> last(v), v -> first(v))
8-element Array{Any,1}:
  1
  2
  3
  4
  5
  6
 10
 20
```
"""
function toposort(nodes, predf, nodef = identity)
    ndict = OrderedDict(nodef(n) => predf(n) for n in nodes)
    
    tmarks = Set()
    pmarks = Set()
    l = []
    function visit(n)
        n in pmarks && return
        n in tmarks && error("not a DAG!")
        push!(tmarks, n)
        for p in ndict[n]
            visit(p)
        end
        delete!(tmarks, n)
        push!(l, n)
        push!(pmarks, n)
    end

    while length(pmarks) < length(nodes)
        for n in keys(ndict)
            n ∉ pmarks && visit(n)
        end
    end
    l
end

end