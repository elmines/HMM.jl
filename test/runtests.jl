using HMM
using Test
using LinearAlgebra: norm
using Random: MersenneTwister


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

@testset "Sampling" begin
    m = Model([.6 .4; .5 .5], [.2 .4 .4; .5 .4 .1], [.8 .2])
    T = 3
    n = 1000
    outsize = size(m.emission, 2)

    out_samples = Array{Int, 2}(undef, n, T)
    for i ∈ 1:n
        out_samples[i, :] = sample(m, T; seed=i)
    end

    function obs_counts(seq)::Vector{Int}
        c = zeros(Int, outsize)
        for i ∈ seq
            c[i] += 1
        end
        c
    end

    exp_dists = Array{Float64, 2}(undef, T, outsize)
    for t ∈ 1:T
        exp_dists[t, :] = obs_counts(out_samples[:,t]) ./ n
    end
    theor_dists = simulate(m, T)

    @test norm(exp_dists .- theor_dists, 2) < .1
end

@testset "Learning" begin
    rng = MersenneTwister(0)
    m = Model([.9 .1; .1 .9], [.2 .4 .4; .5 .4 .1], [.8 .2])
    randInt = (a,b) -> a + abs(rand(rng,Int)) % (b-a+1)

    n = 20000
    samples::Vector{Vector{Int}} = [sample(m, randInt(5,10); seed=i) for i ∈ 1:n]

    # Trying to relearn the original true model
    hiddensize = size(m.transition, 1)
    outsize = size(m.emission, 2)
    init = deepcopy(m.initial)

    m̂ = learn(samples, outsize, hiddensize; max_epochs=15, init=init, verbose=true) 

    println(m.transition)
    println(m̂.transition)
    println()
    println(m.emission)
    println(m̂.emission)

end