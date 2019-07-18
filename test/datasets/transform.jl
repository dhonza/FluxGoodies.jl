using DataFrames
using FluxGoodies
using Random

function build_trans()
    Random.seed!(0)
    n = 20000
    df = DataFrame([10 * (randn(n) .- 5), rand(["X", "Y", "Z"], n)], [:A, :B])
    trans = ChainColumnTransform([
        ParallelColumnTransform([
                CopyColumnTransform(Column{Float64}(:A)),
                OneHotColumnTransform{String}(Column{String}(:B), df.B)
                ]), 
        ParallelColumnTransform([
                StandardizeColumnTransform(Column{Float64}(:A)),
                StandardizeColumnTransform(Column{Float64}(:B_X)),
                StandardizeColumnTransform(Column{Float64}(:B_Y)),
                StandardizeColumnTransform(Column{Float64}(:B_Z))
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

@testset "ann_transform" begin
    @testset "DataFrame to DataFrame" begin
        srcdf = DataFrame([[1,2,3], ["X", "Y", "Z"]], [:A, :B])
        trans = ann_transform(srcdf)
        dstdf = transform(trans, srcdf, DataFrame)
        @test dstdf == DataFrame([[1,2,3], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [:A, :B_X, :B_Y, :B_Z])
        dstdf[1, :A] = 5
        dstdf[2, :B_X] = 1.0
        dstdf[2, :B_Y] = 0.0
        invtransform!(trans, srcdf, dstdf)
        @test srcdf == DataFrame([[5,2,3], ["X", "X", "Z"]], [:A, :B])
    end

    @testset "DataFrame to Matrix" begin
        srcdf = DataFrame([[1,2,3], ["X", "Y", "Z"]], [:A, :B])
        trans = ann_transform(srcdf)
        dstdf = transform(trans, srcdf, Matrix{Float64})
        @test dstdf == [1.0 1 0 0; 2 0 1 0; 3 0 0 1]
        dstdf[1, 1] = 5
        dstdf[2, 2] = 1.0
        dstdf[2, 3] = 0.0
        invtransform!(trans, srcdf, dstdf)
        @test srcdf == DataFrame([[5,2,3], ["X", "X", "Z"]], [:A, :B])
    end
end