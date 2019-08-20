module FluxGoodies

include("utils.jl")
export dview, toposort, obsdim, coldim, nobs, ncols

include("losses/basic.jl")
export batchlogitbinarycrossentropy

include("training/training.jl")
export trainepochs!, StopException

include("architectures/mlp.jl")
export MLP

include("datasets/transform.jl")
using .Transforms
export ColTrans, datasource, mergecols, onehot, onecold, replacevals, standardize, fit!, transform, invert, collecttrans, toposort, 
    deserialize, tonumerical, copytransform, invert, inids, outids, intype, outtype, getdatasource, invtransform

include("datasets/dataset.jl")
using .Datasets
export Dataset, loaddataset, obsparts, colparts, obspartcols, colpartcols, tonumerical, buildtransform

end # module
