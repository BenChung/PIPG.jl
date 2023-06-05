abstract type Preconditioner end
abstract type PreconditionerState{T} end
struct Primal end 
struct Dual end

#=
Preconditioner protocol:
precondition(p::Problem{T}, pc::Preconditioner)::Tuple{Problem{T}, PreconditionerState{T}}
precondition!(src::Problem, tgt::Problem, pc::PreconditionerState{T})

extract(next_pc::Func{Int, T}, pc::PreconditionerState{T}, ::Union{Primal, Dual}, ind::Int)::T
=#
