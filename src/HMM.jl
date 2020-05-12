module HMM

RowVector{T} = Array{T, 2}

struct Model
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

function likelihood(m::Model, o::Vector{Int})::Float64
    T = size(o,1)
    em = m.emission
    joint_probs = m.initial .* transpose(em[:,o[1]])
    for i in 2:T
        joint_probs = joint_probs * m.transition
        joint_probs = joint_probs .* transpose(em[:,o[i]])
    end
    sum(joint_probs)
end

function decode(m::Model, o::Vector{Int})::Vector{Int}
    T = size(o, 1)
    N = size(m.transition, 1)
    backpointers = Array{Int, 2}(undef, T, N)
    trans = m.transition
    em = m.emission
    joint_probs = dropdims( m.initial .* transpose(em[:, o[1]]), dims=1)
    for t = 2:T
        (probs, inds) = findmax(joint_probs .* trans, dims=1)
        probs = dropdims(probs, dims=1) .* em[:,o[t]]
        backpointers[t, :] = transpose([i[1] for i in inds])
        joint_probs = probs
    end

    decoded = Vector{Int}(undef, T)
    decoded[T] = findmax(joint_probs)[2]
    for t = T:-1:2
        next = backpointers[t, decoded[t]]
        decoded[t-1] = next
    end
    decoded
end

export Model, simulate, likelihood, decode

end # module
