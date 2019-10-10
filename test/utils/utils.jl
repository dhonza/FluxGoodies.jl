using DataStructures: OrderedDict
using FluxGoodies

@testset "toposort" begin
    nodes = OrderedDict(2 => [1], 5 => [1, 3, 4], 4 => [1, 2], 6 => [5], 1 => [], 3 => [1], 20 => [5, 10], 10 => [1])
    res = toposort(nodes, v -> last(v), v -> first(v))
    @test res == [1, 2, 3, 4, 5, 6, 10, 20]
end