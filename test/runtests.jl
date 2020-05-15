using HMM
using Test


@testset "Path Probabilities" begin
    m = HMM.Model([.6 .4; .5 .5], [.2 .4 .4; .5 .4 .1], [.8 .2])
    o = [3, 1, 3]
    α = HMM.forwardprobs(m, o)
    β = HMM.backwardprobs(m, o)
    T = size(o, 1)
    lklhd = HMM.likelihood(m, o)

    @test α[1,:] ≈ m.initial' .* m.emission[:, o[1]]
    @test α ≈ [.32 .02; .0404 .069; .023496 .005066]
    @test .028562 ≈ lklhd
    @test sum(α[T, :]) == lklhd
    for t ∈ 1:T
        @test α[t, :]' * β[t, :] ≈ lklhd
    end
end