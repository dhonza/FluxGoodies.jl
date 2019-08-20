using DataFrames
using DataStructures: OrderedDict
using JSON
using Random

function generate_data1()
    Random.seed!(0)
    n = 10
    df = DataFrame([rand(1:10, n), 0.5 .+ randn(n), 10 .+ 3randn(n), rand(["X", "Y", "Z"], n), rand(Int8[0,1], n), zeros(Bool, n)], [:A1, :A2, :B, :C, :Da, :D2])
    df.D2 .= 1 .- df.Da
    df
end

function transform_data1(X, sid=:default)
    dsrc = datasource(X; sid=sid)
    re = replacevals(:C => dsrc, "X" => "x", "Z" => "z")
    oh = onehot(:C => re)
    re2 = replacevals(:C_Y => oh, false => "no", true => "yes")
    oc = onecold([:Da => dsrc, :D2 => dsrc], :D, OrderedDict(:Da => "a", :D2 => 2))
    sd = standardize(:A1 => dsrc)
    sd2 = standardize(:B => dsrc)
    sd3 = standardize(:C_z => oh)
    ddst = mergecols(:A1 => sd, :A2 => dsrc, :B => sd2, :C_Y => re2, :C_z => sd3, :C_x => oh, :D => oc)
    dsrc, ddst
end

function generate_data2()
    Random.seed!(0)
    n = 10
    DataFrame([rand(0:100, n), rand(["M", "F"], n)], [:O, :S])
end

@testset "transform and inverse" begin
    X = generate_data1()
    dsrc, ddst = transform_data1(X, :sid1)
    smap = Dict(:sid1 => X)
    fit!(ddst, smap)

    Y = transform(ddst, smap, DataFrame)
    T = DataFrame(
        A1 = [-1.1423471897815947, -0.5565281180987257, 1.4938386327913158, 0.32220048942557783, -1.1423471897815947, -1.1423471897815947, -0.2636185822572912, 1.2009290969498814, 0.9080195611084468, 0.32220048942557783],
        A2 = [-1.1072563241277753, -1.9807927306599402, 2.7762328327845243, 0.7196934675425409, 0.3828624692670648, -0.1012535941853836, 1.6422764389413194, 0.41138368179952706, 0.7794662520898984, 0.6114216780915173],
        B = [-1.0510047730459515, 0.28255507796868556, 0.004359925255237256, -1.700136355232938, 1.8049184082894767, 0.17779513965723276, 0.46682135785382906, -0.9522129757959658, 0.6311744633370341, 0.33572973171335546],
        C_Y = ["no", "no", "no", "no", "yes", "no", "yes", "no", "no", "no"],
        C_z = [1.1618950038622249, -0.7745966692414833, -0.7745966692414833, 1.1618950038622249, -0.7745966692414833, 1.1618950038622249, -0.7745966692414833, 1.1618950038622249, -0.7745966692414833, -0.7745966692414833],
        C_x = Bool[false, true, true, false, false, false, false, false, true, true],
        D = Any["a", "a", "a", "a", 2, "a", "a", "a", "a", "a"]
    )
    @test Y ≈ T

    smap2 = Dict(:sid2 => Y)
    dsrc2 = datasource(Y; sid=:sid2)
    ddst2 = invert(ddst, dsrc2)
    Xt = transform(ddst2, smap2, DataFrame)
    @test X ≈ Xt

    @test dsrc == getdatasource(ddst)
end

@testset "tonumerical" begin
    X = generate_data1()
    ddst = tonumerical(X)
    fit!(ddst, X)
    Ym = transform(ddst, X, Matrix{Float64})
    dsrcm = datasource(Ym, outids(ddst))
    ddstm = invert(ddst, dsrcm)
    Xtm = transform(ddstm, Ym, DataFrame)
    @test X ≈ Xtm

    Xtm2 = invtransform(ddst, Ym, DataFrame)
    @test X ≈ Xtm2
end

@testset "serialization" begin
    X = generate_data1()
    dsrc, ddst = transform_data1(X, :sid1)
    smap = Dict(:sid1 => X)
    fit!(ddst, smap)

    str = JSON.json(ddst, 3)
    dict = JSON.parse(str, dicttype = OrderedDict)
    ddst2 = deserialize(dict)

    Y = transform(ddst, smap, DataFrame)
    Y2 = transform(ddst2, smap, DataFrame)

    @test Y ≈ Y2
end