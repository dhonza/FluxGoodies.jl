module Evaluation

using MLDataPattern: nobs, getobs
using Statistics

using ..Transforms: ColTrans

# binary for single output
prob2class(data::AbstractMatrix{<:AbstractFloat}) = reshape(Int.(getobs(data)), :)

probs2class(data::AbstractMatrix{<:AbstractFloat}) = reshape(getindex.(argmax(data; dims=1), 1), :)

ids2classes(is::Symbol...) = begin pairs = split.(string.(is), "_"); Symbol(pairs[1][1]) => [string(v[2]) for v in pairs] end
ids2classes(i::ColTrans) = ids2classes(outids(i)...)

function accuracy(Y::AbstractVector{<:Integer}, T::AbstractVector{<:Integer}) 
    size(Y) == size(T) || error("different dimensions size(Y)=$(size(Y)), size(T)=$(size(T))")
    mean(Y .== T)
end

accuracy(Y::AbstractMatrix{<:AbstractFloat}, T::AbstractMatrix{<:AbstractFloat}) = accuracy(probs2class(Y), probs2class(T))

end