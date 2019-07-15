using Flux
using Statistics

batchlogitbinarycrossentropy(logY, T) = mean((1 .- T).*logY .- logσ.(logY))