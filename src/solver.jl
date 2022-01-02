abstract type Space{T, D} end	
struct InfNorm{T, D, δ} <: Space{T, D} end
struct Product{T, D1, D2, D3, S1 <: Space{T, D1}, S2 <: Space{T, D2}} <: Space{T, D3}
	s1::S1
	s2::S2
	Product(s1::S1, s2::S2) where {T, D1, D2, S1 <: Space{T, D1}, S2 <: Space{T, D2}} = new{T, D1, D2, D1+D2, S1, S2}(s1, s2)
end
abstract type Cone{T, D} <: Space{T, D} end
dim(::Cone{T,D}) where {T,D} = D
dim(::Type{T}) where {D, Tc, T<:Cone{Tc,D}} = D
struct Reals{T, D} <: Cone{T, D} end # R^D
struct Zeros{T, D} <: Cone{T, D} end # 0^D
struct POCone{T, D} <: Cone{T, D} end # { x | forall i, x_i >= 0}
struct NOCone{T, D} <: Cone{T, D} end # { x | forall i, x_i <= 0}
struct SOCone{T, D} <: Cone{T, D} end # { [t,x] | |x| <= t }
struct NSOCone{T, D} <: Cone{T, D} end # { [t,x] | |x| <= -t }

struct PCone{T, D1, D2, D3, C1 <: Cone{T, D1}, C2 <: Cone{T, D2}} <: Cone{T, D3}
	c1::C1
	c2::C2
	PCone(c1::C1, c2::C2) where {T, D1, D2, C1 <: Cone{T, D1}, C2 <: Cone{T, D2}} = new{T, D1, D2, D1+D2, C1, C2}(c1, c2)
end
struct PTCone{T, Cs<:Tuple{Vararg{C where C<:Cone{T}}}, D} <: Cone{T, D}
	cones::Cs
	@generated PTCone{T}(cs::Cs) where {T, Cs <: Tuple{Vararg{C where C<:Cone{T}}}} = Expr(:new, PTCone{T, Cs, sum(dim.(Cs.parameters); init=0)}, :(cs))
end

# projections
project(::Reals{T, D}, x::SVector{D, T}) where {D, T} = x
project(::Zeros{T, D}, x::SVector{D, T}) where {D, T} = zeros(SVector{D})
project(::InfNorm{T, D, δ}, x::SVector{D, T}) where {D, T, δ} = clamp.(x, -δ, δ)


function project(c::POCone{T, D}, x::SVector{D, T}) where {D, T}
	return max.(x, zero(T))
end

function project(c::NOCone{T, D}, x::SVector{D, T}) where {D, T}
	return min.(x, zero(T))
end

@generated function project(c::SOCone{T, D}, x::SVector{D, T}) where {T, D}
	onev = one(T)
	two = 2*onev
	zv = zeros(SVector{D,T})
	vect_inds = SVector{D-1}(2:D)
	return quote
		xnorm = norm(x[$vect_inds])
		r = x[1]
		if xnorm <= r
			return x
		elseif xnorm <= -r
			return $zv
		else 
			scalefact = (xnorm + r)/($two)
			component_factor = (scalefact)/xnorm
			return vcat(SVector(scalefact), component_factor * x[$vect_inds])
		end
	end
end

@generated function project(c::NSOCone{T, D}, x::SVector{D, T}) where {T, D}
	vect_inds = SVector{D-1}(2:D)
	return quote 
		po_equiv = project(SOCone{T,D}(), vcat(-x[1], x[$vect_inds]))
		return vcat(-po_equiv[1], po_equiv[$vect_inds])
	end
end

@generated function project(c::PCone{T, D1, D2, D3, C1, C2}, x::SVector{D3, T}) where {D1, D2, D3, C1, C2, T}
	r1 = SVector{D1, Int64}(1:D1)
	r2 = SVector{D2, Int64}(D1+1:D3)
	return :(vcat(project(c.c1, x[$r1]), project(c.c2, x[$r2])))
end
@generated function project(c::PTCone{T, Cs, D}, x::SVector{D, T}) where {T, Cs, D}
	if length(Cs.parameters) == 0 || D == 0 return :(SVector{0,T}()) end
	idxes = prepend!(accumulate(+, dim.(Cs.parameters); init=1), 1)
	ranges = map(x->UnitRange(x...), zip(idxes, Iterators.drop(idxes, 1) .- 1))
	projections = [:(project(c.cones[$i], x[SVector{$(dim(Cs.parameters[i])), Int64}($(ranges[i]))])) for i in 1:length(Cs.parameters)]
	return :(vcat($(projections...)))
end


@generated function project(c::Product{T, D1, D2, D3, C1, C2}, x::SVector{D3, T}) where {D1, D2, D3, C1, C2, T}
	r1 = SVector{D1, Int64}(1:D1)
	r2 = SVector{D2, Int64}(D1+1:D3)
	return :(vcat(project(c.s1, x[$r1]), project(c.s2, x[$r2])))
end

# polars
polar(::Reals{T,D}) where {T,D} = Zeros{T,D}()
polar(::Zeros{T,D}) where {T,D} = Reals{T,D}()
polar(::POCone{T,D}) where {T,D} = NOCone{T,D}()
polar(::NOCone{T,D}) where {T,D} = POCone{T,D}()
polar(::SOCone{T,D}) where {T,D} = NSOCone{T,D}()
polar(::NSOCone{T,D}) where {T,D} = SOCone{T,D}()
polar(c::PCone{T,D1,D2,D3,C1,C2}) where {T,D1,D2,D3,C1,C2} = PCone(polar(c.c1), polar(c.c2)) 

@generated function polar(c::PTCone{T, Cs, D}) where {T, Cs, D} 
	tup = Expr(:tuple, (:(polar(c.cones[$i])) for i in 1:length(Cs.parameters))...)
	return :(PTCone{T}($tup))
end


# problem
# minimize 1/2 z^t P z + q^T z 
# s.t. H z - g ∈ K, z ∈ D

struct Problem{T,N,M,K<:Cone{T,M},D<:Space{T,N}}
	k::K
	d::D
	H::SparseMatrixCSC{T, Int}
	P::SparseMatrixCSC{T, Int}
	q::MVector{N, T}
	g::MVector{M, T}
	c::T
end

abstract type Diagnostics{T} end
struct NoDiagnostics{T} <: Diagnostics{T} end
struct LogDiagnostics{T} <: Diagnostics{T} end
record_diagnostics(d::NoDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T = nothing
function record_diagnostics(d::LogDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T 
	if i == 1
		println("iteration | w | w_delta | z | z_delta")
	end
	if i % 10000 == 0
		println("$i | $w | $w_delta | $z | $z_delta")
	end
end

abstract type Scaling{T, M, N} end
struct NoScaling{T, M, N} <: Scaling{T, M, N}
	NoScaling(::Problem{T,N,M}) where {T,M,N} = new{T, M, N}()
end
struct GeoMean{T, M, N} <: Scaling{T, M, N}
	rowmaxes::MVector{M, T}
	rowmins::MVector{M, T}
	colscale::MVector{N, T}
	GeoMean(::Problem{T,N,M}) where {T,M,N} = new{T, M, N}(MVector{M, T}(zeros(T, M)), MVector{M, T}(zeros(T, M)), MVector{N, T}(zeros(T, N)))
end
struct ArithMean{T, M, N} <: Scaling{T, M, N}
	rowsums::MVector{M, T}
	rowscale::MVector{M, T}
	colscale::MVector{N, T}
	ArithMean(::Problem{T,N,M}) where {T,M,N} = new{T, M, N}(MVector{M, T}(zeros(T, M)), MVector{M, T}(zeros(T, M)), MVector{N, T}(zeros(T, N)))
end

@enum SolverState INDEFINITE OPTIMAL PRIMAL_INFEASIBLE DUAL_INFEASIBLE TIMEOUT
mutable struct State{T,N,M,K,D,P <: Problem{T,N,M,K,D}, G <: Diagnostics{T}, S <: Scaling{T, M, N}}
	w_i1::MVector{M, T}
	z_i1::MVector{N, T}
	z_i2::MVector{N, T}
	v_i1::MVector{M, T}

	col_scale::MVector{N, T}
	row_scale::MVector{M, T}
	
	solver_state::SolverState
	primal::MVector{N, T}
	dual::MVector{M, T} 

	diagnostics::G
	scaling::S
	function State(p::P; diag::G = NoDiagnostics{T}(), scaling::S = GeoMean(p)) where {
			T,N,M,K,D,
			P<:Problem{T,N,M,K,D}, 
			G<:Diagnostics{T},
			S<:Scaling{T, M, N}}
	return new{T,N,M,K,D,P,G,S}(
		MVector{M,T}(zeros(T, M)), MVector{N,T}(zeros(T, N)), MVector{N,T}(zeros(T, N)), MVector{M,T}(zeros(T, M)), 
		MVector{N,T}(ones(T, N)), MVector{M, T}(ones(T, M)),
		INDEFINITE, MVector{N, T}(zeros(T, N)), MVector{M, T}(zeros(T, M)), diag, scaling)
	end
end

function objective_value(p::P, s::State{T,N,M,K,D,P}) where {T,N,M,K,D,P<:Problem{T,N,M,K,D}}
	return transpose(s.primal) * p.P * s.primal * 0.5 + dot(s.primal, p.q)
end

function compute_α(p::P, γ::T) where {T,N,M,K,D,P<:Problem{T,N,M,K,D}}
	λ = max_singular_value(p.P)
	ν = max_singular_value(p.H)
	return (8.0-4.0/γ)/(sqrt(λ^2 + 16*ν^2) + λ)
end

function spmul!(res::MVector{N,T}, A::SparseMatrixCSC{T, Int}, B::SVector{M,T}, α::T, β::T) where {T<:Number,N,M}
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
function spmul!(res::MVector{N,T}, At::Transpose{T, SparseMatrixCSC{T, Int}}, B::SVector{M,T}, α::T, β::T) where {T<:Number,N,M}
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

function pipg(p::P,s::State{T,N,M,K,D,P,G}, iters::Int, α::T, ϵ::T, z::SVector{N,T}, v::SVector{M,T}) where {T,N,M,K,D,P<:Problem{T,N,M,K,D}, G<:Diagnostics{T}}
	w = v
	w_delta = zero(T)
	z_delta = zero(T)
	niters = 0
	for i=1:iters
		w_prev = w
		# w_raw = v + α*(p.H * z - p.g)
		s.w_i1 .= p.g
		spmul!(s.w_i1, p.H, z, α, -α)
		w_raw = v + s.w_i1
		w = project(polar(p.k), w_raw)
		z_prev = z
		# z_raw = z - α*(p.P * z + p.q + transpose(p.H) * w)
		s.z_i1 .= p.q
		spmul!(s.z_i1, p.P, z, α, α)
		spmul!(s.z_i1, transpose(p.H), w, α, 1.0)
		z_raw = z - s.z_i1
		z = project(p.d, z_raw)
		# v = w + α * p.H * (z - z_prev)
		spmul!(s.v_i1, p.H, z - z_prev, α, 0.0)
		v = w + s.v_i1

		w_delta = norm(w - w_prev)
		z_delta = norm(z - z_prev)
		record_diagnostics(s.diagnostics, i, w, z, v, w_delta, z_delta)
		niters = i
		if w_delta < ϵ && z_delta < ϵ
			break
		end
	end
	if w_delta < ϵ && z_delta < ϵ
		s.solver_state = OPTIMAL
		s.primal .= z
		s.dual .= v
		return niters
	elseif w_delta > ϵ
		s.solver_state = PRIMAL_INFEASIBLE
		return niters
	elseif z_delta > ϵ
		s.solver_state = DUAL_INFEASIBLE
		return niters
	end
	s.solver_state = TIMEOUT
	s.primal .= z
	return niters
end

function max_singular_value(m::SparseMatrixCSC{T, Int}; iters=10, ϵ=1e-8) where {T}
	rows, cols = size(m)
	vect = rand(cols)
	temp = zeros(rows)
	for i=1:iters
		if norm(vect) <= ϵ return norm(vect) end
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