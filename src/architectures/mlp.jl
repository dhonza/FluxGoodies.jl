using Flux

struct MLP
    layers
end
  
@Flux.treelike MLP
  
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

                              
(m::MLP)(x::AbstractArray) = m.layers(x)