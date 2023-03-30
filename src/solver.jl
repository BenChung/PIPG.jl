abstract type Diagnostics{T} end
struct NoDiagnostics{T} <: Diagnostics{T} end
struct LogDiagnostics{T} <: Diagnostics{T} end
record_diagnostics(d::NoDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T = nothing
function record_diagnostics(d::LogDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T 
	if i == 1
		println("iteration | w | w_delta | z | z_delta")
	end
	if i % 500 == 0
		println("$i | $w | $w_delta | $z | $z_delta")
	end
end

abstract type Scaling{T, M, N} end
struct GeoMean{T, M, N} <: Scaling{T, M, N}
	colmax::Vector{T} # N
	colmin::Vector{T} # N
	GeoMean(::Problem{T,N,M}) where {T,M,N} = new{T, M, N}(zeros(T, N), zeros(T, N))
end
struct ArithMean{T, M, N} <: Scaling{T, M, N}
	colsum::Vector{T} # N
	ArithMean(::Problem{T,N,M}) where {T,M,N} = new{T, M, N}(zeros(T, N))
end
struct Equilibration{T, M, N} <: Scaling{T, M, N}
	colmax::Vector{T} # N
	Equilibration(::Problem{T,N,M}) where {T,M,N} = new{T, M, N}(zeros(T, N))
end

function initialize_like(x::AbstractArray{T}) where T
	out = similar(x)
	out .= zero(T)
	return out
end


@enum SolverState INDEFINITE OPTIMAL PRIMAL_INFEASIBLE DUAL_INFEASIBLE TIMEOUT
mutable struct State{T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A}, G <: Diagnostics{T}, PE <: AbstractArray{Int}}
	w_i1::A # M
	ξ_i1::A # N
	z_i2::A # N
	v_i1::A # M

	w_prev::A # M
	z_prev::A # N

	w_work::A # M
	z_work::A # N

	η_work::A # M
	ξ_work::A # N
	k_scaling::A # M
	
	col_scale::A # N
	row_scale::A # M

	col_perm::PE # N
	row_perm::PE # M
	
	solver_state::SolverState
	primal::A # N
	dual::A # M
	history::Vector{A}

	diagnostics::G
	function State(p::P; col_perm::PE = 1:N, row_perm::PE = 1:M, diag::G = NoDiagnostics{T}()) where {T,N,M,K,D,A,
			P<:Problem{T,N,M,K,D,A}, 
			G<:Diagnostics{T}, PE<:AbstractArray{Int}}
	col_scale = similar(p.q)
	col_scale .= one(T)
	row_scale = similar(p.g)
	row_scale .= one(T)
	w_i1 = initialize_like(p.g)
	ξ_i1 = initialize_like(p.q)
	z_i2 = initialize_like(p.q)
	v_i1 = initialize_like(p.g)
	w_prev = initialize_like(p.g)
	z_prev = initialize_like(p.q)
	w_work = initialize_like(p.g)
	z_work = initialize_like(p.q)
	η_work = initialize_like(p.g)
	ξ_work = initialize_like(p.q)
	primal = initialize_like(p.q)
	dual = initialize_like(p.g)
	k_scaling = initialize_like(p.g)
	return new{T,N,M,K,D,A,P,G,PE}(
		w_i1, ξ_i1, z_i2, v_i1, 
		w_prev, z_prev, 
		w_work, z_work, 
		η_work, ξ_work,
		k_scaling,
		col_scale, row_scale, col_perm, row_perm,
		INDEFINITE, primal, dual, [], diag)
	end
end

function objective_value(p::P, s::State{T,N,M,K,D,A,P}) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}}
	return transpose(s.primal) * p.P * s.primal * 0.5 + dot(s.primal, p.q)
end
function dual_objective_value(p::P, s::State{T,N,M,K,D,A,P}) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}}
	return -transpose(s.primal) * p.P * s.primal * 0.5 - dot(s.dual, p.g)
end

function compute_α(p::P, γ::T, ω::T=T(2.0)) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}}
	λ = max_singular_value(p.P)
	ν = max_singular_value(p.H)
	α = (2γ)/(sqrt(λ^2 + 4*ω*ν^2) + λ)
	β = ω * α
	return α, β
end

function spmul!(res::MVector{N,T}, A::SparseMatrixCSC{T, Int}, B::StaticVector{M,T}, α::T, β::T) where {T<:Number,N,M}
	@assert N == A.m
	@assert M == A.n
	res .*= β
	for j in 1:A.n
		vv = B[j] * α
		for i in A.colptr[j]:A.colptr[j+1]-1
			@inbounds res[A.rowval[i]] += A.nzval[i] * vv
		end
	end
end
function spmul!(res::MVector{N,T}, At::Transpose{T, SparseMatrixCSC{T, Int}}, B::StaticVector{M,T}, α::T, β::T) where {T<:Number,N,M}
	A = At.parent
	@assert N == A.n 
	@assert M == A.m
	res .*= β
	for j in 1:A.n
		acc = zero(T)
		for i in A.colptr[j]:A.colptr[j+1]-1
			@inbounds acc += A.nzval[i] * B[A.rowval[i]] * α
		end
		@inbounds res[j] += acc
	end
end

function norm_err(a::Vector{T}, b::Vector{T}) where T
	@assert length(a) == length(b)
	result = zero(T)
	for i=1:length(a)
		@inbounds result += (a[i] - b[i])^2
	end
	return sqrt(result)
end
function pipg(p::P,s::State{T,N,M,K,D,A,P,G}, iters::Int, (α,β)::Tuple{T,T}, ϵ::T, ξ_init::A, η_init::A; ρ=T(1.5)) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}, G<:Diagnostics{T}}
	@assert length(ξ_init) == N
	@assert length(η_init) == M
	s.w_prev .= zero(T)
	s.z_prev .= zero(T)
	s.w_work .= zero(T)
	s.z_work .= zero(T)
	s.ξ_work .= ξ_init
	s.η_work .= η_init
	s.ξ_i1 .= zero(T)
	s.w_i1 .= zero(T)
	s.primal .= zero(T)
	s.dual .= zero(T)
	w_delta = zero(T)
	z_delta = zero(T)
	w_prev_delta = zero(T)
	z_prev_delta = zero(T)
	β_scale = 1/(β*ρ)
	α_scale = 1/(α*ρ)
	niters = 0
	for i=1:iters
		s.z_prev .= s.z_work
		# z_raw = ξ - α(H^T η + Pξ +  q)
		s.ξ_i1 .= p.q # ξ_i1 = q
		mul!(s.ξ_i1, p.P, s.ξ_work, α, α) # ξ_i1 = α * P * ξ + α * q
		mul!(s.ξ_i1, p.H, s.η_work, -α, -1.0) # ξ_i1 = - α (H^t η + P * ξ + q)
		z_raw = s.ξ_i1 # z_raw = - α (H^t η + P * ξ + q)
		z_raw .+= s.ξ_work # z_raw = ξ - α (H^t η + P * ξ + q)
		project!(s.z_work, 1, p.d, z_raw)
		z_delta = norm_err(s.z_work, s.z_prev)

		s.w_prev .= s.w_work
		# s.w_i1 = v + α*(transpose(p.H) * z - p.g) -> ξ + β(H(2z - ξ) - g)
		s.w_i1 .= p.g 
		s.ξ_i1 .= @~ 2 .* s.z_work .- s.ξ_work # w_work = 2z - ξ
		mul!(s.w_i1, transpose(p.H), s.ξ_i1, β, -β) # w_i1 = β(H(2z - ξ) - g)
		s.w_i1 .+= s.η_work # w_i1 = η + β(H(2z - ξ) - g)
		project!(s.w_work, 1, polar(p.k), s.w_i1)
		# ξ = (1-ρ)ξ + ρ z
		s.ξ_work .= @~ (1-ρ) .* s.ξ_work .+ ρ .* s.z_work
		# η = (1-ρ)η + ρ w
		s.η_work .= @~ (1-ρ) .* s.η_work .+ ρ .* s.w_work

		w_prev_delta = w_delta
		z_prev_delta = z_delta
		w_delta = norm_err(s.w_work, s.w_prev)

		record_diagnostics(s.diagnostics, i, s.w_work, s.z_work, s.ξ_work, w_delta, z_delta)
		niters = i
		if β_scale*w_delta < ϵ && α_scale*z_delta < ϵ
			s.w_work .= s.w_prev
			s.z_work .= s.z_prev
			break
		end
		if isinf(w_delta) || isinf(z_delta)
			break
		end
		if isnan(w_delta) || isnan(z_delta)
			# halt iteration with the previous values of w and z
			w_delta = w_prev_delta
			z_delta = z_prev_delta
			break
		end
	end
	if β_scale*w_delta < ϵ && α_scale*z_delta < ϵ
		s.solver_state = OPTIMAL
		s.primal .= s.z_work
		s.dual .= s.w_work
		return niters
	elseif β_scale*w_delta > ϵ || isinf(w_delta)
		s.solver_state = PRIMAL_INFEASIBLE
		s.dual .= s.w_work ./ norm(s.w_work)
		return niters
	elseif α_scale*z_delta > ϵ || isinf(z_delta)
		s.solver_state = DUAL_INFEASIBLE
		s.primal .= s.z_work ./ norm(s.z_work)
		return niters
	end
	s.solver_state = TIMEOUT
	s.primal .= s.z_work
	return niters
end

function max_singular_value(m::SparseMatrixCSC{T, Int}; iters=10, ϵ=1e-8) where {T}
	rows, cols = size(m)
	vect = rand(cols)
	temp = zeros(rows)
	for i=1:iters
		if norm(vect) <= ϵ 
			return sqrt(norm(vect))
		end
		mul!(temp, m, vect)
		mul!(vect, transpose(m), temp, 1/norm(vect), 0.0)
	end
	return sqrt(norm(vect))
end
#=
function test()
	P_ex = spzeros(2,2)
	q = MVector(-50.0, -120.0)
	H_ex = sparse([100.0 200; 10 30; 1 1; -1 0; 0 -1])
	g = MVector(10000.0, 1200, 110, -0.0, -0.0)
	prob = PIPG.Problem(PIPG.NOCone{Float64, 5}(), PIPG.Reals{Float64, 2}(), H_ex, P_ex, q, g, 0.0)
	state = PIPG.State(prob)

	λ = PIPG.max_singular_value(P_ex)
	ν = PIPG.max_singular_value(H_ex)
	γ = 0.9
	α = (8.0-4.0/γ)/(sqrt(λ^2 + 16*ν^2) + λ)
	result = @time PIPG.pipg(prob, state, 2000000, α, 0.0001, SVector(1.0, 1.0), SVector(0.0,0.0,0.0,0,0))
	println(result)
end
=#