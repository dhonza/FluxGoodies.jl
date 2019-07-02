using Flux

struct MLP
    layers
end
  
@Flux.treelike MLP
  
function MLP(ls::Vector{Int}; dropout = 0.0, batchnorm = false)
    layers = []
    for i in 2:length(ls) - 1
        push!(layers, Dense(ls[i - 1], ls[i], relu))
        batchnorm && push!(layers, BatchNorm(ls[i]))
        dropout > 0.0 && push!(layers, Dropout(dropout))
    end
  #     push!(layers, Dense(ls[end-1], ls[end], σ))
    push!(layers, Dense(ls[end - 1], ls[end]))
    MLP(Chain(layers...))
end
                              
(m::MLP)(x::AbstractArray) = m.layers(x)