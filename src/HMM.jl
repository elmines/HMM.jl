module HMM

using LinearAlgebra: norm
using Random: MersenneTwister
using Distributions: Categorical
using Printf

RowVector{T} = Array{T, 2}

mutable struct Model
	transition::Matrix{Float64}
    emission::Matrix{Float64}
    initial::RowVector{Float64}
	function Model(trans, emis, init)
		if size(trans, 2) != size(emis, 1)
			throw(ArgumentError("Inner dimensions of transition and emission must match"))
        end
        if size(init, 2) != size(trans, 1)
            throw(ArgumentError("Inner dimensions of initial and transtion must match"))
        end
		new(trans, emis, init)
	end
end

function simulate(m::Model, n::Int) :: Array{Float64, 2}
    outsize = size(m.emission, 2)
    probs = Array{Float64, 2}(undef, n, outsize)
    
    probs[1, :] = m.initial * m.emission
    accum = m.initial
	for i in 2:n
        accum = accum * m.transition
        probs[i, :] = accum * m.emission
	end
	probs
end

function sample(m::Model, T::Int; seed::Int = 0) :: Vector{Int}
    o = Vector{Int}(undef, T)
    dist = m.initial
    rng = MersenneTwister(seed)

    cat_dist = () -> Categorical(dropdims(dist * m.emission, dims=1))

    o[1] = rand(rng, cat_dist() )
    for t ∈ 2:T
        dist *= m.transition
        o[t] = rand(rng, cat_dist() )
    end
    o
end

function forwardprobs(m::Model, o::Vector{Int})::Array{Float64, 2}
    T = size(o,1)
    em = m.emission
    hiddensize = size(em, 1)
    probs = Array{Float64, 2}(undef, T, hiddensize)
    probs[1, :] = m.initial .* transpose(em[:,o[1]])
    for i in 2:T
        probs[i, :] = probs[i-1, :]' * m.transition .* em[:,o[i]]'
    end
    probs
end

function backwardprobs(m::Model, o::Vector{Int})::Array{Float64, 2}
    trans = m.transition
    em = m.emission
    T = size(o, 1)
    N = size(trans, 1)
    probs::Array{Float64, 2} = Array{Float64, 2}(undef, T, N)
    probs[T, :] .= 1
    for t ∈ (T-1):-1:1
        probs[t, :] = trans * ( probs[t+1,:] .* em[:, o[t+1]] )
    end
    probs
end


function likelihood(m::Model, o::Vector{Int})::Float64
    probs = forwardprobs(m, o)
    sum(probs[end, :])
end

"""
Has runtime Θ(TN^2) where T = |o| and N is the number of states
"""
function decode(m::Model, o::Vector{Int})::Vector{Int}
    T = size(o, 1)
    N = size(m.transition, 1)
    backpointers = Array{Int, 2}(undef, T, N)
    trans = m.transition
    em = m.emission
    joint_probs = dropdims( m.initial .* transpose(em[:, o[1]]), dims=1)
    for t = 2:T                                                   # Θ(TN^2)
        (probs, inds) = findmax(joint_probs .* trans, dims=1)       # Θ(N^2)
        probs = dropdims(probs, dims=1) .* em[:,o[t]]               # Θ(N)
        backpointers[t, :] = transpose([i[1] for i in inds])        # Θ(N)
        joint_probs = probs
    end

    decoded = Vector{Int}(undef, T)
    decoded[T] = findmax(joint_probs)[2]
    for t = T:-1:2                                                # Θ(T)
        next = backpointers[t, decoded[t]]
        decoded[t-1] = next
    end
    decoded
end


"""

Returns the "progress": (the L1 norm of the change in the model's parameters)
"""
function update(m::Model, o::Vector{Int})::Tuple{Matrix{Float64},Matrix{Float64}}
    (hiddensize, outsize) = size(m.emission)
    trans = m.transition
    emit = m.emission
    T::Int = size(o,1)
    sum_drop = (A, dims) -> dropdims(sum(A,dims=dims),dims=dims)

    nonans = (A) -> !any(isnan.(A))

    α = forwardprobs(m, o);
    β = backwardprobs(m, o)


    lklhd::Float64 = sum(α[end, :])
    γ::Array{Float64,2} = α .* β ./ lklhd

    ξ::Array{Float64,3} = zeros(T-1, hiddensize, hiddensize)
    for t = 1:T-1
        temp::Array{Float64, 2} = α[t, :] .* trans .* β[t+1, :]' .* emit[:, o[t+1]]
        ξ[t, :, :] = temp
    end
    ξ ./= lklhd

    trans_new::Array{Float64,2} = sum_drop(ξ, 1) ./ sum_drop(ξ, (1,3))

    mask::Array{Float64,2} = o .==  collect(1:outsize)'

    numer::Array{Float64, 2} = (γ' * mask)::Array{Float64,2}
    denom::Array{Float64, 1} = sum_drop(γ,1)::Array{Float64,1}

    emit_new::Array{Float64,2} = numer ./ denom

    (trans_new, emit_new)
end

"""
Train a model given a set of observations

- O: A jagged array of observations (i.e. the observation sequences can have different durations)
- outsize: Number of distinct possible observations
- hiddensize: Number of Markov states
"""
function learn(O::Vector{Vector{Int}}, outsize::Int, hiddensize::Int;
               init::RowVector{Float64}=missing, max_epochs::Int=10, verbose::Bool=false)::Model
    trans::Array{Float64,2} = ones(hiddensize,hiddensize) / hiddensize
    emit::Array{Float64,2} = ones(hiddensize, outsize) / outsize
    nsamples = size(O, 1)
    if ismissing(init)
        init = ones(1,hiddensize) / hiddensize
    elseif size(init) != (1, hiddensize)
        throw(ArgumentError("init must be a hiddensize-vector"))
    end

    m = Model(trans, emit, init)
    hasnan = (A) -> any(isnan.(A))

    for i ∈ 1:max_epochs
        new_trans = zero(m.transition)
        new_emit = zero(m.emission)
        good_incs = 0
        for (j, o) in enumerate(O)
            (trans_inc, emit_inc) = update(m, o)
            if hasnan(trans_inc) || hasnan(emit_inc); continue; end
            good_incs += 1
            new_trans += trans_inc
            new_emit += emit_inc
        end
        if good_incs == 0;
            if verbose
                println("No more good samples; terminating training.")
            end
            break;
        end
        new_trans ./= good_incs
        new_emit ./= good_incs

        prog = norm(m.transition - new_trans, 2) + norm(m.emission - new_emit, 2)
        if verbose
            @printf("Epoch %d/%d: progress=%.3f\n", i, max_epochs, prog)
        end
        m.transition = new_trans
        m.emission = new_emit
    end
    m
end

export Model, simulate, sample, likelihood, decode, update, learn

end # module
