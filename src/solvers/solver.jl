abstract type Solver{T} end
abstract type SolverState{T, A<:AbstractArray} end

@enum OptimizationResult INDEFINITE OPTIMAL PRIMAL_INFEASIBLE DUAL_INFEASIBLE TIMEOUT
mutable struct State{T,N,M,K,D,A,
		P <: Problem{T,N,M,K,D,A}, 
		G <: Diagnostics{T},
		Pc <: PreconditionerState{T}, 
		Pr <: Problem{T},
		Sl <: Solver{T},
		Ss <: SolverState{T, A}}
	solver::Sl
	solver_state::Ss

	preconditioner::Pc
	preconditioned::Pr

	diagnostics::G
	function State(p::P, solver::Solver{T}; diag::G = NoDiagnostics{T}(), preconditioner::Pr = Ruiz()) where {T,N,M,K,D,A,
			P<:Problem{T,N,M,K,D,A}, 
			G<:Diagnostics{T},
			Pr<:Preconditioner}
	(tp, pcs) = precondition(p, preconditioner)
	ss = initialize(solver, tp)
	return new{T,N,M,K,D,A,P,G,typeof(pcs),typeof(tp), typeof(solver), typeof(ss)}(solver, ss, pcs, tp, diag)
	end
end

function precondition!(prob::P, s::S) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}, S<:State{T,N,M,K,D,A}}
	precondition!(prob, s.preconditioned, s.preconditioner)
end 
function solve(prob::P, s::S) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}, S<:State{T,N,M,K,D,A}}
	precondition!(prob, s.preconditioned, s.preconditioner)
	return solve(s.preconditioned, s.solver, s.solver_state, s.diagnostics)
end

primal(s::State) = let v = primal(s.solver_state); extract.(idx -> v[idx], (s.preconditioner, ), (Primal(), ), eachindex(v)) end
dual(s::State) = let v = dual(s.solver_state); extract.(idx -> v[idx], (s.preconditioner, ), (Dual(), ), eachindex(v)) end

function objective_value(p::P, s::State{T,N,M,K,D,A,P}) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}}
	pr = primal(s)
	return transpose(pr) * p.P * pr * 0.5 + dot(pr, p.q)
end
function dual_objective_value(p::P, s::State{T,N,M,K,D,A,P}) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}}
	pr = primal(s)
	dl = dual(s)
	return -transpose(pr) * p.P * pr * 0.5 - dot(dl, p.g)
end
