module FluxGoodies

include("losses/basic.jl")
export batchlogitbinarycrossentropy

include("training/training.jl")
export trainepochs!

include("architectures/mlp.jl")
export MLP

include("datasets/dataset.jl")
export Tabular, RawDataset, inputs, targets, partition, load_abalone, load_ecoli, load_housing, load_iris
include("datasets/transform.jl")
export Column, ColumnTransform, CopyColumnTransform, 
    NominalToIntTransform, OneHotColumnTransform, StandardizeColumnTransform, 
    ParallelColumnTransform, ChainColumnTransform
export srccols, dstcols, transform!, invtransform!, transform, invtransform, allocate, fit! 
export nominal, transform_to_numerical
export dump_transform, load_transform

end # module
