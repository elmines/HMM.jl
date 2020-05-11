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
    println(joint_probs)
    for i in 2:T
        joint_probs = joint_probs * m.transition
        joint_probs = joint_probs .* transpose(em[:,o[i]])
        println(joint_probs)
    end
    sum(joint_probs)
end

export Model, simulate, likelihood

end # module
