
#=
Ruiz preconditioner 
=#

struct Ruiz <: Preconditioner end
struct RuizState{T} <: PreconditionerState{T}
	# output scalings
	row_scale::Vector{T}
	col_scale::Vector{T}

	# internal state
	δD::Vector{T}
	δK::Vector{T}
	H_rowmaxes::Vector{T}
	H_colmaxes::Vector{T}
	P_colmaxes::Vector{T}
	P_rowmaxes::Vector{T}
	RuizState(p::Problem{T}) where T = 
		new{T}(zeros(T, rows(p)), 
			zeros(T, cols(p)),
			zeros(T, cols(p)),
			zeros(T, rows(p)),
			zeros(T, rows(p)),
			zeros(T, cols(p)),
			zeros(T, cols(p)),
			zeros(T, cols(p)))
end

function precondition(p::Problem, pc::Ruiz)
	return (Problem(p), RuizState(p))
end

function precondition!(src::Problem, tgt::Problem, pc::RuizState{T}) where T
	# setup the internal state
	pc.col_scale .= one(T); pc.row_scale .= one(T)
	pc.δD .= one(T); pc.δK .= one(T)
	@assert cols(src) == length(pc.H_colmaxes)
	@assert rows(src) == length(pc.H_rowmaxes)
	c = 1
	for i=1:10
		pc.H_rowmaxes .= zero(T); pc.H_colmaxes .= zero(T)
		pc.P_colmaxes .= zero(T); pc.P_rowmaxes .= zero(T)
		compute_maxvals(pc.H_rowmaxes, pc.H_colmaxes, src.H, pc.col_scale, pc.row_scale)
		compute_maxvals(pc.P_colmaxes, pc.P_rowmaxes, src.P, pc.col_scale, pc.col_scale) 
		pc.P_colmaxes .*= c
		notzero(n) = n == zero(T) ? one(T) : n
		pc.δD .= 1 ./ notzero.(sqrt.(max.(pc.H_colmaxes, pc.P_colmaxes)))
		pc.δK .= 1 ./ notzero.(sqrt.(pc.H_rowmaxes))
		#println("Ds: $Ds Ps: $Ps")
		#println("delta D len: $(length(pc.δD)) delta K len: $(length(pc.δK))")
		#println("H_rowmaxes: $(pc.H_rowmaxes) H_colmaxes: $(pc.H_colmaxes) P_colmaxes: $(pc.P_colmaxes)")
		constrain!(typeof(src.k), pc.δK, 0)
		nrows = rows(src)
		colmean = nrows > 0 ? sum(@~ pc.P_colmaxes .* pc.δD .* pc.δD ./ nrows) : zero(T)
		#println("δD: $(pc.δD) δK: $(pc.δK) cm: $colmean") 
		pc.col_scale .*= pc.δD

		pc.δD .*= c
		pc.δD .*= abs.(src.q)
		γ = 1 / notzero(max(colmean, maximum(pc.δD)))
		c *= γ
		pc.row_scale .*= pc.δK
	end

	scale_cone!(tgt.k, src.k, pc.row_scale, 0)
	scale_cone!(tgt.d, src.d, pc.col_scale, 0)
	#println("tgt H: $(tgt.H)")
	nonzeros(tgt.H) .= nonzeros(src.H)
	row_scale(tgt, src, pc.row_scale)
	col_scale(tgt, src, pc.col_scale)
	#println("tgt H: $(tgt.H)")
end
extract(value::T, pc::RuizState{T}, ::Primal, index) where T = pc.col_scale[index] * value
extract(value::T, pc::RuizState{T}, ::Dual, index) where T = pc.row_scale[index] * value

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

@generated function scale_cone!(tgt::C, src::C, scaling::Vector{T}, offs::Int) where {T, CS, C<:Union{PTSpace{T, CS}, PTCone{T, CS}}}
	out = :(begin end)
	current_offs = 0
	cone_index = 1
	for cone in CS.parameters
		push!(out.args, :(scale_cone!(tgt.cones[$cone_index], src.cones[$cone_index], scaling, offs+$current_offs)))
		current_offs += dim(cone)
		cone_index += 1
	end
	return out
end
scale_cone!(::C, ::C, scaling::Vector{T}, offs::Int) where {T, D, C <: Union{Reals{T, D}, Zeros{T, D}, SignCone{T, D}}} = nothing
function scale_cone!(tgt::HalfspaceCone{T,D}, src::HalfspaceCone{T,D}, scaling::Vector{T}, offs::Int) where {T, D}
	tgt.d .= src.d .* @view scaling[1+offs:offs+D]
	tgt.d_norm2 = sum(tgt.d .^ 2)
end
function scale_cone!(tgt::SOCone{T, D}, src::SOCone{T,D}, scaling::Vector{T}, offs::Int) where {T, D}
	total = zero(T)
	for i=2+offs:offs+D 
		total += scaling[i]
	end
	mean = total/(D-1)
	tgt.angle = src.angle * mean/scaling[1+offs]
end

scale_cone!(tgt::Equality{T, D}, src::Equality{T, D}, col_scale::Vector{T}, arg_offs::Int) where {T, D} = tgt.v .=  src.v ./ col_scale[arg_offs+1:arg_offs + D]
scale_cone!(tgt::InfNorm{T, D}, src::InfNorm{T, D}, col_scale::Vector{T}, arg_offs::Int) where {T, D} = tgt.δ .= src.δ ./ col_scale[arg_offs+1:arg_offs + D]
scale_cone!(::Space{T, D}, ::Space{T, D}, col_scale::Vector{T}, arg_offs::Int) where {T, D} = nothing # no-op for the most part

function row_scale(tgt::P, src::P, row_scaling #=::SVector{M, T}=#) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A}}
	#println("row scaling: $(row_scaling)")
	rmul!(tgt.H, Diagonal(row_scaling))
	#println(p.g)
	tgt.g .= src.g .* row_scaling
end

function col_scale(tgt::P, src::P, col_scaling #=::SVector{N, T}=#) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A}}
	dmat = Diagonal(col_scaling)
	# rescale H
	lmul!(dmat, tgt.H)
	# rescale P; if scaling is rep. by diagonal matrix D, then 
	# (D z)^t P (D z) = z^t D P D z [since (Dz)^t = z^t D^t = z^t D] so new P=D P D
	nonzeros(tgt.P) .= nonzeros(src.P)
	lmul!(dmat, tgt.P)
	rmul!(tgt.P, dmat)
	# rescale q
	tgt.q .= src.q .* col_scaling
end