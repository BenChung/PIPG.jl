struct xPIPG{T} <: Solver{T}
	ϵ::T
	ξ_init::T
	η_init::T
	ρ::T
	γ::T
	ω::T
	iters::Int
	xPIPG(ϵ::T, γ::T; ξ_init::T=T(0), η_init::T=T(0), ρ::T = T(1.5), ω::T = T(2.0), iters=2000000) where {T} = new{T}(ϵ, ξ_init, η_init, ρ, γ, ω, iters)
end
struct xPIPGState{T, A} <: SolverState{T, A}
	primal::A # N
	dual::A # M

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
end

primal(s::xPIPGState) = s.primal
dual(s::xPIPGState) = s.dual

function initialize_like(x::AbstractArray{T}) where T
	out = similar(x)
	out .= zero(T)
	return out
end

function initialize(solver::xPIPG{T}, p::Problem{T, N, M, K, D, A}) where {T,N,M,K,D,A}
	primal = initialize_like(p.q)
	dual = initialize_like(p.g)
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
	return xPIPGState{T,A}(
		primal, dual,
		w_i1, ξ_i1, z_i2, v_i1, 
		w_prev, z_prev, 
		w_work, z_work, 
		η_work, ξ_work)
end

function norm_err(a::Vector{T}, b::Vector{T}) where T
	@assert length(a) == length(b)
	result = zero(T)
	for i=1:length(a)
		@inbounds result += (a[i] - b[i])^2
	end
	return sqrt(result)
end

function compute_α(p::Problem{T}, γ::T, ω::T=T(2.0)) where {T}
	λ = max_singular_value(p.P)
	ν = max_singular_value(p.H)
	α = (2γ)/(sqrt(λ^2 + 4*ω*ν^2) + λ)
	β = ω * α
	return α, β
end

function solve(p::P, solver::xPIPG{T}, solver_state::xPIPGState{T, A}, d::G) where {T,N,M,K,D,A,P<:Problem{T,N,M,K,D,A}, G<:Diagnostics{T}}
	if (M == 0 && max_singular_value(p.P) < 1e-9) 
		solver_state.primal .= -p.q
		return DUAL_INFEASIBLE, 0
	end
	iters = solver.iters
	(α, β) = compute_α(p, solver.γ, solver.ω)
	ϵ = solver.ϵ
	ρ = solver.ρ
	w_prev = solver_state.w_prev; z_prev = solver_state.z_prev; 
	w_work = solver_state.w_work; z_work = solver_state.z_work
	ξ_work = solver_state.ξ_work; η_work = solver_state.η_work
	ξ_i1 = solver_state.ξ_i1; w_i1 = solver_state.w_i1
	primal = solver_state.primal; dual = solver_state.dual
	w_prev .= zero(T)
	z_prev .= zero(T)
	w_work .= zero(T)
	z_work .= zero(T)
	ξ_work .= solver.ξ_init
	η_work .= solver.η_init
	ξ_i1 .= zero(T)
	w_i1 .= zero(T)
	
	primal .= zero(T)
	dual .= zero(T)
	w_delta = zero(T)
	z_delta = zero(T)
	w_prev_delta = zero(T)
	z_prev_delta = zero(T)
	β_scale = 1/(β*ρ)
	α_scale = 1/(α*ρ)
	niters = 0
	for i=1:iters
		z_prev .= z_work
		# z_raw = ξ - α(H^T η + Pξ +  q)
		ξ_i1 .= p.q # ξ_i1 = q
		mul!(ξ_i1, p.P, ξ_work, α, α) # ξ_i1 = α * P * ξ + α * q
		mul!(ξ_i1, p.H, η_work, -α, -1.0) # ξ_i1 = - α (H^t η + P * ξ + q)
		z_raw = ξ_i1 # z_raw = - α (H^t η + P * ξ + q)
		z_raw .+= ξ_work # z_raw = ξ - α (H^t η + P * ξ + q)
		project!(z_work, 1, p.d, z_raw)
		z_delta = norm_err(z_work, z_prev)

		w_prev .= w_work
		# w_i1 = v + α*(transpose(p.H) * z - p.g) -> ξ + β(H(2z - ξ) - g)
		w_i1 .= p.g 
		ξ_i1 .= @~ 2 .* z_work .- ξ_work # w_work = 2z - ξ
		mul!(w_i1, transpose(p.H), ξ_i1, β, -β) # w_i1 = β(H(2z - ξ) - g)
		w_i1 .+= η_work # w_i1 = η + β(H(2z - ξ) - g)
		project!(w_work, 1, polar(p.k), w_i1)
		# ξ = (1-ρ)ξ + ρ z
		ξ_work .= @~ (1-ρ) .* ξ_work .+ ρ .* z_work
		# η = (1-ρ)η + ρ w
		η_work .= @~ (1-ρ) .* η_work .+ ρ .* w_work

		w_prev_delta = w_delta
		z_prev_delta = z_delta
		w_delta = norm_err(w_work, w_prev)

		record_diagnostics(d, i, w_work, z_work, ξ_work, w_delta, z_delta)
		niters = i
		if β_scale*w_delta < ϵ && α_scale*z_delta < ϵ
			w_work .= w_prev
			z_work .= z_prev
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
		primal .= z_work
		dual .= w_work
		return OPTIMAL, niters
	elseif β_scale*w_delta > ϵ || isinf(w_delta)
		println("term case 1")
		println(p)
		dual .= w_work ./ norm(w_work)
		return PRIMAL_INFEASIBLE, niters
	elseif α_scale*z_delta > ϵ || isinf(z_delta)
		println("term case 2")
		primal .= z_work ./ norm(z_work)
		return DUAL_INFEASIBLE, niters
	end
	primal .= z_work
	return TIMEOUT, niters
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