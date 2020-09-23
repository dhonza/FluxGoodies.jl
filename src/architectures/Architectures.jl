module Architectures
using Base.Iterators
using MLDataUtils

export MLP, batcheval

include("mlp.jl")

"""
    batcheval(model, bsize, X...)

Evaluate `model` on inputs `X` using batches of size `bsize` allocating the output automatically.
"""
function batcheval(model, bsize, X...)
    n = nobs(X[1])
    ranges = partition(1:n, bsize)
    idxs, state = iterate(ranges)
    Y1 = model((X_[:, idxs] for X_ in X)...).data # old Flux
    # Y1 = model((X_[:, idxs] for X_ in X)...) # Zygote Flux
    Y = similar(Y1, size(Y1)[1:end-1]..., n)
    Y[:, idxs] .= Y1
    while true
        res = iterate(ranges, state)
        isnothing(res) && break
        idxs, state = res
        Y[:,idxs] .= model((X_[:, idxs] for X_ in X)...).data # old Flux
        # Y[:,idxs] .= model((X_[:, idxs] for X_ in X)...) # Zygote Flux
    end
    Y
end

end