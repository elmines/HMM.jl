module HMM

RowVector{T} = Array{T, 2}

struct Model
	transition::Matrix{Float64}
	emission::Matrix{Float64}
	function Model(trans, emis)
		if size(trans, 2) != size(emis, 1)
			throw(ArgumentError("Inner dimensions of transition and emission must match"))
		end
		new(trans, emis)
	end
end

function simulate(m::Model, p::RowVector{Float64}, n::Int) :: RowVector{Float64}
	if size(p, 2) != size(m.transition, 1)
		throw(ArgumentError("p must have same number of states as m"))
	end
	accum = p;
	for i in 1:n
		accum = accum * m.transition
	end
	accum * m.emission
end

end # module
