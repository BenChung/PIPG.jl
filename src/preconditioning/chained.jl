#=
Chained preconditioning:
=#

struct Chained{PCs <: Tuple{Vararg{<:Preconditioner}}} <: Preconditioner
	preconditioners::PCs
end
struct ChainedState{T,P<:Problem{T}, STs <: Tuple{Vararg{<:PreconditionerState{T}}}} <: PreconditionerState{T}
	problems::Vector{P}
	states::STs
end

function precondition(p::Problem{T}, pcs::Chained{PCs}) where {T, PCs}
	intermediate_problems = Problem{T}[]
	states = PreconditionerState[]
	current_problem = p
	for (i, pc) in enumerate(pcs.preconditioners)
		(current_problem, state) = precondition(current_problem, pc)
		if i < length(pcs.preconditioners)
			push!(intermediate_problems, current_problem)
		end
		push!(states, state)
	end
	return (current_problem, ChainedState(intermediate_problems, (states...,)))
end
function precondition!(src::Problem, tgt::Problem, pcs::ChainedState)
	if length(pcs.states) == 0
		return # we literally didn't do anything
	end
	if length(pcs.problems) == 0
		precondition!(src, tgt, pcs.states[1])
		return
	end
	current_state = src
	current_target = first(pcs.problems)
	for (i, state) in enumerate(pcs.states)
		if i == length(pcs.states)
			current_target = tgt
		else 
			current_target = pcs.problems[i]
		end
		precondition!(current_state, current_target, state)
		current_state = current_target
	end
end

function extract(next_pc::Function, pc::ChainedState{T}, p::Union{Primal, Dual}, ind::Int)::T where T 
	# next_pc :: Func{Int, T}
	@inline function index_mapping(v::Val{IDX}) where IDX
		if IDX == length(pc.states)+1
			return next_pc
		else
			return (index) -> extract(index_mapping(Val(IDX+1)), pc.states[IDX], p, index)
		end
	end
	if length(pc.states) == 0
		return next_pc(ind)
	end
	res = extract(index_mapping(Val(2)), first(pc.states), p, ind)
	return res
end