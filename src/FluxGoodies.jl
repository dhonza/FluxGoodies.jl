module FluxGoodies
using Reexport

include("losses/Losses.jl")
@reexport using .Losses

include("training/Training.jl")
@reexport using .Training

include("architectures/Architectures.jl")
@reexport using .Architectures

end # module
