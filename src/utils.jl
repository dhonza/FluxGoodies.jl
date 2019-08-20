using DataFrames
using DataStructures: OrderedDict

obsdim(::AbstractDataFrame) = 1
obsdim(::AbstractMatrix) = 2

coldim(::AbstractDataFrame) = 2
coldim(::AbstractMatrix) = 1

nobs(d) = size(d, obsdim(d))
ncols(d) = size(d, coldim(d))

dview(d::AbstractDataFrame, obs, cols) = view(d, obs, cols)
dview(d::AbstractMatrix, obs, cols) = view(d, cols, obs)

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
        all(isapproxifexists(dfa[n], dfb[n]) for n in names(dfa))
end

function toposort(nodes, nodef, predf)
    # nodes: list of node objects
    # function nodef: node object -> node
    # function predf: node object -> node predecessors
    # see https://en.wikipedia.org/wiki/Topological_sorting#Depth-first_search
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