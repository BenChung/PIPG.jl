

function scale(p::P, s::State{T,N,M,K,D,P,G,NoScaling{T,M,N}}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G}
	# do nothing
end
function scale(p::P, s::State{T,N,M,K,D,P,G,GeoMean{T,M,N}}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G}
	rs,cs = geometric_scaling(p, s)
	#row_scale(p, s, rs)
	rs,cs = geometric_scaling(p, s)
	#col_scale(p, s, cs)
	rs,cs = equilibration(p, s)
	row_scale(p, s, rs)
	rs,cs = equilibration(p, s)
	#col_scale(p, s, cs)
end

function geometric_scaling(p::P, s::State{T,N,M,K,D,P,G,GeoMean{T,M,N}}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G}
	rows = rowvals(p.H)
	vals = nonzeros(p.H)
	m, n = size(p.H)
	# set up row bound storage; use the storage in State (free at this point)
	rowmaxes = s.scaling.rowmaxes
	rowmaxes .= typemin(T)
	rowmins = s.scaling.rowmins
	rowmins .= typemax(T)

	colscale = s.scaling.colscale
	if m == 0 || n == 0 
		rowmaxes .= one(T)
		colscale .= one(T)
		return rowmaxes, colscale
	end
	# compute bounds on the elements of the constraint matrix
	for j = 1:n
		colmax, colmin = typemin(T), typemax(T)
		for i in nzrange(p.H, j)
			row = rows[i]
			val = vals[i]
			rowmaxes[row] = max(rowmaxes[row], val)
			rowmins[row] = min(rowmins[row], val)
			colmax = max(colmax, val)
			colmin = min(colmin, val)
		end
		colscale[j] = if n > 0 1/sqrt(abs(colmax*colmin)) else one(T) end
	end
	for i = 1:m
		rowmaxes[i] = if m > 0 1/sqrt(abs(rowmaxes[i]*rowmins[i])) else one(T) end
	end
	return rowmaxes, colscale
end
function scale(p::P, s::State{T,N,M,K,D,P,G,ArithMean{T,M,N}}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G}
	rs,cs = arithmetic_scaling(p, s)
	row_scale(p, s, rs)
	rs,cs = arithmetic_scaling(p, s)
	col_scale(p, s, cs)
end
function arithmetic_scaling(p::P, s::State{T,N,M,K,D,P,G,ArithMean{T,M,N}}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G}
	rows = rowvals(p.H)
	vals = nonzeros(p.H)
	m, n = size(p.H)
	# set up row bound storage; use the storage in State (free at this point)
	rowsums = s.scaling.rowsums
	rowsums .= zero(T)

	rowscale = s.scaling.rowscale
	colscale = s.scaling.colscale
	if m == 0 || n == 0 
		rowscale .= one(T)
		colscale .= one(T)
		return rowscale, colscale
	end
	# compute bounds on the elements of the constraint matrix
	for j = 1:n
		colsum = zero(T)
		for i in nzrange(p.H, j)
			row = rows[i]
			val = vals[i]
			rowsums[row] += val
			colsum += val
		end
		colscale[j] = if n > 0 abs(n/colsum) else one(T) end
	end
	for i = 1:m
		rowscale[i] = if m > 0 abs(m/rowsums[i]) else one(T) end
	end
	return rowscale, colscale
end

function equilibration(p::P, s::State{T,N,M,K,D,P,G,S}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G,S<:Scaling{T}}
	rows = rowvals(p.H)
	vals = nonzeros(p.H)
	m, n = size(p.H)

	rowmaxes = s.w_i1
	rowmaxes .= typemin(T)
	colscale = s.z_i1
	colscale .= zero(T)

	if m == 0 || n == 0 
		rowmaxes .= one(T)
		colscale .= one(T)
		return rowmaxes, colscale
	end
	for j = 1:n
		colmax = typemin(T)
		for i in nzrange(p.H, j)
			row = rows[i]
			val = vals[i]
			rowmaxes[row] = max(rowmaxes[row], abs(val))
			colmax = max(colmax, abs(val))
		end
		colscale[j] = 1/colmax
	end
	for i = 1:m
		rowmaxes[i] = 1/rowmaxes[i]
	end
	return rowmaxes, colscale
end

function row_scale(p::P, s::State{T,N,M,K,D,P,G,S}, row_scaling::MVector{M,T}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G,S<:Scaling{T}}
	lmul!(Diagonal(row_scaling), p.H)
	p.g .*= row_scaling
	s.row_scale .*= row_scaling
end

function col_scale(p::P, s::State{T,N,M,K,D,P,G,S}, col_scaling::MVector{N, T}) where {T,N,M,K,D,P <: Problem{T,N,M,K,D},G,S<:Scaling{T}}
	dmat = Diagonal(col_scaling)
	# rescale H
	rmul!(p.H, dmat)
	# rescale P; if scaling is rep. by diagonal matrix D, then 
	# (D z)^t P (D z) = z^t D P D z [since (Dz)^t = z^t D^t = z^t D] so new P=D P D
	lmul!(dmat, p.P)
	rmul!(p.P, dmat)
	# rescale q
	p.q .*= col_scaling
	s.col_scale .*= col_scaling
end