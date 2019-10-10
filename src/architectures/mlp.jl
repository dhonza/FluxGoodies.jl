using Flux

struct MLP
    layers
end
  
@Flux.treelike MLP

"""
    MLP(inopts::Int, layeropts::Union{Int,Pair{Int}}...; activation = relu, outactivation = nothing)

Build a MultiLayer Perceptron.

# Examples
```jldoctest
julia> model = MLP(3, 5, 10 => [:act => relu, :dropout => 0.2], 5 => [:act => relu, :batchnorm],  2; activation = σ, outactivation = identity)
MLP(Chain(Dense(3, 5, σ), Dense(5, 10, relu), Dropout{Float64}(0.2, Colon(), true), Dense(10, 5, relu), BatchNorm(5), Dense(5, 2)))
```
"""
function MLP(inopts::Int, layeropts::Union{Int,Pair{Int}}...; activation = relu, outactivation = nothing)
    outactivation = outactivation === nothing ? activation : outactivation
    function extract_layeropts(l, i)
        act = i < length(layeropts) ? activation : outactivation
        drop = 0.0
        bn = false
        if l isa Pair{Int}
            opts = last(l)
            opts = opts isa Union{AbstractArray,Tuple} ? opts : [opts]
            for o in opts
                o = o isa Symbol ? o => true : o
                if first(o) == :act
                    act = last(o)
                elseif first(o) == :dropout
                    drop = last(o)
                elseif first(o) == :batchnorm
                    bn = true
                else
                    error("unknown layer option: $(first(o))")
                end
            end
        end
        (size = first(l), activation = act, batchnorm = bn, dropout = drop)
    end
    
    layers = []
    nprev = inopts
    for (i, l) in enumerate(layeropts)
        opts = extract_layeropts(l, i)
        push!(layers, Dense(nprev, opts.size, opts.activation))
        opts.batchnorm && push!(layers, BatchNorm(opts.size))
        opts.dropout > 0.0 && push!(layers, Dropout(opts.dropout))
        nprev = first(l)
    end
    MLP(Chain(layers...))
end


"""
    (m::MLP)(x::AbstractArray)

Evaluate MLP.

# Examples
```jldoctest
julia> using Random; Random.seed!(1);

julia> X = randn(Float32, 3, 4)
3×4 Array{Float32,2}:
  0.297288  -0.0104452   2.29509   0.431422
  0.382396  -0.839027   -2.26709   0.583708
 -0.597634   0.311111    0.529966  0.963272

julia> model = MLP(3, 5, 2)
MLP(Chain(Dense(3, 5, relu), Dense(5, 2, relu)))

julia> model(X)
Tracked 2×4 Array{Float32,2}:
 0.0  0.0  0.0  0.377914
 0.0  0.0  0.0  0.687224
```
"""
(m::MLP)(X::AbstractArray) = m.layers(X)