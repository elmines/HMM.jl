module HMM

using LinearAlgebra: norm

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

function simulate(m::Model, n::Int) :: RowVector{Float64}
	accum = m.initial;
	for i in 1:n
		accum = accum * m.transition
	end
	accum * m.emission
end

function _forwardprobs(m::Model, o::Vector{Int})::Array{Float64, 2}
    T = size(o,1)
    em = m.emission
    hiddensize = size(em, 1)
    probs = Array{Float64, 2}(undef, T, hiddensize)
    probs[1, :] = m.initial .* transpose(em[:,o[1]])
    for i in 2:T
        probs[i, :] = transpose(probs[i-1, :]) * m.transition .* transpose(em[:,o[i]])
    end
    probs
end
function _backwardprobs(m::Model, o::Vector{Int})::Vector{Float64}
    T = size(o, 1)
    emit = m.emission
    trans = m.transition
    probs = Array{Float64, 2}(undef, T, size(trans, 1))
    probs[T, :] .= 1
    for t = (T-1):-1:1
        probs[t, :] = trans * probs[t+1,:] .* emit[:, o[t+1]]
    end
    probs
end

function likelihood(m::Model, o::Vector{Int})::Float64
    probs = _forwardprobs(m, o)
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
function update(m::Model, o::Vector{Int})::Float64
    (hiddensize, outsize) = size(m.emission)
    trans = m.transition
    emit = m.emission
    T = size(o)

    α = _forwardprobs(m, o)
    β = _backwardprobs(m, o)
    lklhd = sum(α[end, :])
    γ = α .* β ./ lklhd

    ξ = zeros(T-1, hiddensize, hiddensize)
    for t = 1:T-1
        ξ[t, :, :] = transpose(α[t, :]) .* trans .* β[t+1, :] .* emit[:, o[t+1]]
    end
    ξ ./= lklhd

    sum_drop = (A, dims) -> dropdims(sum(A,dims=dims),dims=dims)
    trans_new = sum_drop(ξ, 1) ./ sum_drop(ξ, (1,3))

    emit_new = similar(m.emission)
    mask = o .==  collect(1:hiddensize)'

    emit_new = sum_drop(γ .* mask, 1) ./ 
    #for j = 1:hiddensize
    #    denom::Float64 = sum(γ[:,j])
    #    for k = 1:outsize
    #    end
    #end

    prog = norm(trans_new - m.transition, 1) + norm(emit_new - m.emission, 1)
    prog
end

"""
Train a model given a set of observations

- O: A jagged array of observations (i.e. the observation sequences can have different durations)
- outsize: Number of distinct possible observations
- hiddensize: Number of Markov states
"""
function learn(O::Vector{Vector{Int}}, outsize::Int, hiddensize::Int)::Model
    # Initialize transition matrix to a uniform distribution
    trans::Array{Float64,2} = ones(hiddensize,hiddensize) / hiddensize
    emit::Array{Float64,2} = ones(hiddensize, outsize) / outsize
    T::Int = size(O)

  

    prog = 0.01

    init = ones(1, hiddenSize) / hiddensize #FIXME: Calcluate from α(1)
end

export Model, simulate, likelihood, decode

end # module
