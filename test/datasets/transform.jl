using DataFrames
using FluxGoodies

@testset "ANN Transform" begin
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