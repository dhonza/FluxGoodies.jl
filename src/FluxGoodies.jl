module FluxGoodies
using Reexport

include("utils/Utils.jl")
@reexport using .Utils

include("losses/Losses.jl")
@reexport using .Losses

include("training/Training.jl")
@reexport using .Training

include("architectures/Architectures.jl")
@reexport using .Architectures

include("transforms/Transforms.jl")
@reexport using .Transforms

include("datasets/Datasets.jl")
@reexport using .Datasets

include("evaluation/Evaluation.jl")
@reexport using .Evaluation

end # module
