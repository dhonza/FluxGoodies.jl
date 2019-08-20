using Flux
using Statistics

# Note: original Flux.logitbinarycrossentropy broadcasting calls fail on GPU, this is a batch version
batchlogitbinarycrossentropy(logY, T) = mean((1 .- T).*logY .- logσ.(logY))