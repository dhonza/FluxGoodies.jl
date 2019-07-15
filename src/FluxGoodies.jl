module FluxGoodies

include("losses/basic.jl")
export batchlogitbinarycrossentropy

include("training/training.jl")
export trainepochs!

include("architectures/mlp.jl")
export MLP

include("datasets/dataset.jl")

end # module
