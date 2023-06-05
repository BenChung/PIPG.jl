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

function precondition(p::Problem, pc::Chained{PCs}) where PCs
	intermediate_problems = Problem[]
	states = PreconditionerState[]
	current_problem = p
	for (i, pc) in enumerate(pc.preconditioners)
		(current_problem, state) = precondition(current_problem, pc)
		if i < length(pc.preconditioners)
			push!(intermediate_problems, current_problem)
		end
		push!(states, state)
	end
	return (current_problem, ChainedState(intermediate_problems, (states...,)))
end
function precondition!(src::Problem, tgt::Problem, pc::ChainedState)
	current_state = src
	current_target = first(pc.problems)
	for (i, pc) in enumerate(pc.states)
		if i == length(pc.problems)
			current_target = tgt
		else 
			current_target = pc.problems[i+1]
		end
		precondition!(current_state, current_target, pc)
		current_state = current_target
	end
end

function extract(next_pc::Function, pc::ChainedState{T}, p::Union{Primal, Dual}, ind::Int)::T where T 
	# next_pc :: Func{Int, T}
	@inline function index_mapping(::Val{IDX}) where IDX
		if IDX == length(pc.states)
			return next_pc
		else
			return (index) -> extract(index_mapping(Val(pc_index+1)), pc.states[IDX], p, index)
		end
	end
	return extract(index_mapping(Val(1)), first(pc.states), p, ind)
end