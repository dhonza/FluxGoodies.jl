using DataFrames
using FluxGoodies
using Random

function build_trans()
    Random.seed!(0)
    n = 20000
    df = DataFrame([10 * (randn(n) .- 5), rand(["X", "Y", "Z"], n), rand(["M", "N", "O"], n)], [:A, :B, :C])
    trans = ChainColumnTransform([
        ParallelColumnTransform([
                CopyColumnTransform(Column{Float64}(:A)),
                NominalToIntTransform(Column{String}(:B), df.B),
                OneHotColumnTransform(Column{String}(:C), df.C)
                ]), 
        ParallelColumnTransform([
                StandardizeColumnTransform(Column{Float64}(:A)),
                CopyColumnTransform(Column{Int}(:B)),
                StandardizeColumnTransform(Column{Float64}(:C_M)),
                StandardizeColumnTransform(Column{Float64}(:C_N)),
                StandardizeColumnTransform(Column{Float64}(:C_O))
                ])
    ])
    trans, df
end

@testset "transform and invtransform" begin
    trans, df = build_trans()
    fit!(trans, df)
    df2 = transform(trans, df)
    df3 = invtransform(trans, df2)
    @test df.A ≈ df3.A
    @test df.B == df3.B
    @test df.C == df3.C
end

@testset "dump_transform and load_transform" begin
    trans, df = build_trans()
    path, io = mktemp()
    path2, io2 = mktemp()
    try
        dump_transform(io, trans)
        flush(io)
        trans2 = load_transform(path)
        dump_transform(io2, trans2)
        flush(io2)
        @test read(path, String) == read(path2, String)
    finally
        close(io)
        close(io2)
        rm(path)
        rm(path2)
    end
end

@testset "transform_to_numerical" begin
    @testset "DataFrame to DataFrame" begin
        srcdf = DataFrame([[1,2,3], ["X", "Y", "Z"]], [:A, :B])
        trans = transform_to_numerical(srcdf)
        dstdf = transform(trans, srcdf, DataFrame)
        @test dstdf == DataFrame([[1,2,3], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [:A, :B_X, :B_Y, :B_Z])
        dstdf[1, :A] = 5
        dstdf[2, :B_X] = 1.0
        dstdf[2, :B_Y] = 0.0
        invtransform!(trans, srcdf, dstdf)
        @test srcdf == DataFrame([[5,2,3], ["X", "X", "Z"]], [:A, :B])

        srcdf = DataFrame([[1,2,3], ["X", "Y", "Z"]], [:A, :B])
        trans = transform_to_numerical(srcdf, false)
        dstdf = transform(trans, srcdf, DataFrame)
        @test dstdf == DataFrame([[1,2,3], [1,2,3]], [:A, :B])
        dstdf[1, :A] = 5
        dstdf[2, :B] = 1
        invtransform!(trans, srcdf, dstdf)
        @test srcdf == DataFrame([[5,2,3], ["X", "X", "Z"]], [:A, :B])
    end

    @testset "DataFrame to Matrix" begin
        srcdf = DataFrame([[1,2,3], ["X", "Y", "Z"]], [:A, :B])
        trans = transform_to_numerical(srcdf)
        dstdf = transform(trans, srcdf, Matrix{Float64})
        @test dstdf == [1.0 1 0 0; 2 0 1 0; 3 0 0 1]
        dstdf[1, 1] = 5
        dstdf[2, 2] = 1.0
        dstdf[2, 3] = 0.0
        invtransform!(trans, srcdf, dstdf)
        @test srcdf == DataFrame([[5,2,3], ["X", "X", "Z"]], [:A, :B])

        srcdf = DataFrame([[1,2,3], ["X", "Y", "Z"]], [:A, :B])
        trans = transform_to_numerical(srcdf, false)
        dstdf = transform(trans, srcdf, Matrix{Float64})
        @test dstdf == [1.0 1.0; 2.0 2.0; 3.0 3.0]
        dstdf[1, 1] = 5
        dstdf[2, 2] = 1.0
        invtransform!(trans, srcdf, dstdf)
        @test srcdf == DataFrame([[5,2,3], ["X", "X", "Z"]], [:A, :B])
    end
end