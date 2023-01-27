#=
todo:
See pages 11-13 to see how OSQP does it (for QPs): https://arxiv.org/pdf/1711.08013.pdf
See page 1057 to get a rough idea for how SCS does it (for conic problems): https://link.springer.com/content/pdf/10.1007/s10957-016-0892-3.pdf

I’ll share the manuscript with new preconditioner (that worked better than OSQP’s in my experiments) as soon as we’ve finished writing it (in the next week or so) (edited) 
=#
function scale!(p::P, s::State{T,N,M,K,D,A,P,G,SC}) where {T,N,M,K,D,A, P <: Problem{T,N,M,K,D,A}, G, SC}
	Ds = ones(N)
	Ps = ones(M) # P is analogous to E in the OSQP paper 
	δD = ones(N)
	δK = ones(M)
	H_rowmaxes = zeros(M)
	H_colmaxes = zeros(N)
	P_colmaxes = zeros(N)
	P_rowmaxes = zeros(N)
	c = 1
	for i=1:10
		H_rowmaxes .= zero(T); H_colmaxes .= zero(T)
		P_colmaxes .= zero(T); P_rowmaxes .= zero(T)
		compute_maxvals(H_rowmaxes, H_colmaxes, p.H, Ds, Ps)
		compute_maxvals(P_colmaxes, P_rowmaxes, p.P, Ds, Ds) 
		P_colmaxes .*= c
		notzero(n) = n == zero(T) ? one(T) : n
		δD .= 1 ./ notzero.(sqrt.(max.(H_colmaxes, P_colmaxes)))
		δK .= 1 ./ notzero.(sqrt.(H_rowmaxes))
		#println("Ds: $Ds Ps: $Ps")
		#println("delta D len: $(length(δD)) delta K len: $(length(δK)) M: $M N $N")
		#println("H_rowmaxes: $H_rowmaxes H_colmaxes: $H_colmaxes P_colmaxes: $P_colmaxes")
		constrain!(K, δK, 0)
		colmean = M > 0 ? sum(@~ P_colmaxes .* δD .* δD ./ M) : zero(T)
		#println("δD: $δD δK: $δK cm: $colmean") 
		Ds .*= δD

		δD .*= c
		δD .*= abs.(p.q)
		γ = 1 / notzero(max(colmean, maximum(δD)))
		c *= γ
		Ps .*= δK
	end
	Ds .= one(T)
	apply_scaling!(p, s, Ps, Ds)
end

function compute_maxvals(H_colmaxes, H_rowmaxes, mat::SparseMatrixCSC{T, Int}, row_scaling, col_scaling) where T
	N, M = size(mat)
	#H_colmaxes = zeros(M)
	#H_rowmaxes = zeros(N)
	rows = rowvals(mat)
	vals = nonzeros(mat)
	for col = 1:M for i in nzrange(mat, col)
		row = rows[i]
		val = vals[i] * row_scaling[row] * col_scaling[col]
		H_rowmaxes[row] = max(H_rowmaxes[row], abs(val))
		H_colmaxes[col] = max(H_colmaxes[col], abs(val))
	end end
end

@generated function constrain!(::Type{<:PTCone{T, CS}}, vector::Vector{T}, offs) where {T,CS}
	out = :(begin end)
	current_offs = 0
	for cone in CS.parameters
		push!(out.args, :(constrain!($cone, vector, offs+$current_offs)))
		current_offs += cone_dim(cone)
	end
	return out
end
function constrain!(::Type{<:Union{Reals{T,D}, Zeros{T,D}, SignCone{T,D}, HalfspaceCone{T, D}}}, ::Vector{T}, offs) where {T, D} end
function constrain!(::Type{SOCone{T, D}}, v::Vector{T}, offs) where {T, D} 
	total = zero(T)
	for i=2+offs:offs+D
		total += v[i]
	end
	mean = total/(D-1)
	for i=2+offs:offs+D
		v[i] = mean
 	end
end

function apply_scaling!(p::P, s::State{T,N,M,K,D,A,P,G,SC}, rscale, cscale) where {T,N,M,K,D,A, P <: Problem{T,N,M,K,D,A}, G, SC}
	scale_cone!(p.k, rscale, 0)
	scale_cone!(p.d, cscale, 0)
	row_scale(p, s, rscale)
	col_scale(p, s, cscale)
end

@generated function scale_cone!(c::C, scaling::Vector{T}, offs::Int) where {T, CS, C<:Union{PTSpace{T, CS}, PTCone{T, CS}}}
	out = :(begin end)
	current_offs = 0
	cone_index = 1
	for cone in CS.parameters
		push!(out.args, :(scale_cone!(c.cones[$cone_index], scaling, offs+$current_offs)))
		current_offs += dim(cone)
		cone_index += 1
	end
	return out
end
scale_cone!(::C, scaling::Vector{T}, offs::Int) where {T, D, C <: Union{Reals{T, D}, Zeros{T, D}, SignCone{T, D}}} = nothing
function scale_cone!(c::HalfspaceCone{T,D}, scaling::Vector{T}, offs::Int) where {T, D}
	c.d .*= @view scaling[1+offs:offs+D]
	c.d_norm2 = sum(c.d .^ 2)
end
function scale_cone!(c::SOCone{T,D}, scaling::Vector{T}, offs::Int) where {T, D}
	total = zero(T)
	for i=2+offs:offs+D 
		total += scaling[i]
	end
	mean = total/(D-1)
	c.angle *= mean/scaling[1+offs]
end

scale_cone!(cs::Equality{T, D}, col_scale::Vector{T}, arg_offs::Int) where {T, D} = cs.v ./= col_scale[arg_offs+1:arg_offs + D]
scale_cone!(i::InfNorm{T, D}, col_scale::Vector{T}, arg_offs::Int) where {T, D} = i.δ ./= col_scale[arg_offs+1:arg_offs + D]
scale_cone!(::Space{T, D}, col_scale::Vector{T}, arg_offs::Int) where {T, D} = nothing # no-op for the most part


function row_scale(p::P, s::State{T,N,M,K,D,A,P,G,S}, row_scaling #=::SVector{M, T}=#) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},G,S}
	#println("row scaling: $(row_scaling)")
	#println(p.H)
	rmul!(p.H, Diagonal(row_scaling))
	#println(p.g)
	p.g .*= row_scaling
	s.row_scale .*= row_scaling
end

function col_scale(p::P, s::State{T,N,M,K,D,A,P,G,S}, col_scaling #=::SVector{N, T}=#) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},G,S}
	#println("col scaling: $(col_scaling)")
	dmat = Diagonal(col_scaling)
	# rescale H
	lmul!(dmat, p.H)
	# rescale P; if scaling is rep. by diagonal matrix D, then 
	# (D z)^t P (D z) = z^t D P D z [since (Dz)^t = z^t D^t = z^t D] so new P=D P D
	lmul!(dmat, p.P)
	rmul!(p.P, dmat)
	# rescale q
	p.q .*= col_scaling
	s.col_scale .*= col_scaling
end