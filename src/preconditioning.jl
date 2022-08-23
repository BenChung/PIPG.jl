@generated function scale(p::P, s::State{T,N,M,K,D,A,P,G,SC}) where {T,N,M,K,D,A,
	P <: Problem{T,N,M,K,D,A},
	SC <: Tuple{Vararg{<:Scaling{T, M, N}}},G}
	scale_ast = [quote
		row_scale(p, s, row_scaling(p, s.scaling[$i], 0, p.k))
		col_scale(p, s, col_scaling(p, s.scaling[$i]))
	end for (i, sc) in enumerate(SC.parameters) ]
	return quote $(scale_ast...) end
end
function maxmin_abs(v::AbstractArray{T}) where T
	minv, maxv = typemax(T), typemin(T)
	for val in v
		minv, maxv = min(minv, abs(val)), max(maxv, abs(val))
	end
	return minv, maxv
end

@generated function row_scaling(p::P, 
	s::Scaling{T,M,N}, 
	arg_offs::Int,
	cs::PTCone{T, CS}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},CS}
	if N == 0 || M == 0
		return :(ones(SVector{M, T}))
	end
	arr = Any[]
	offs = 0
	for (i,cone) in enumerate(CS.parameters)
		push!(arr, :(row_scaling(p, s, arg_offs + $offs, cs.cones[$i])))
		offs += cone_dim(cone)
	end
	return :(vcat($(arr...)))
end
function col_scaling(p::P, s::ArithMean{T,M,N}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A}}
	if N == 0 || M == 0
		return ones(SVector{N, T})
	end
	s.colsum .= zero(T)
	cols = rowvals(p.H)
	vals = nonzeros(p.H)
	for j = 1:M
		for nz in nzrange(p.H, j) 
			col = cols[nz]
			val = vals[nz]
			@inbounds s.colsum[col] += abs(val)
		end
	end
	for i in 1:length(s.colsum)
		@inbounds if s.colsum[i] == zero(T)
			@inbounds s.colsum[i] = one(T)
		else 
			@inbounds s.colsum[i] = M / s.colsum[i]
		end
	end

	return SVector{N}(s.colsum)
end
@generated function row_scaling(p::P, 
	s::ArithMean{T,M,N}, 
	offs::Int,
	::Union{Reals{T, CD}, Zeros{T, CD}, POCone{T, CD}, NOCone{T, CD}}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},CD}
	return quote 
		cols = rowvals(p.H)
		vals = nonzeros(p.H)
		return SVector($((:(@inbounds N/sum(@~ abs.(@view vals[nzrange(p.H, $j+offs)]))) for j in 1:CD)...),)
	end
end
@generated function row_scaling(p::P, 
	s::ArithMean{T,M,N}, 
	offs::Int,
	::Union{SOCone{T, CD}, NSOCone{T, CD}}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},CD}
	return quote 
		cols = rowvals(p.H)
		vals = nonzeros(p.H)
		total = +($((:(@inbounds sum(@~ abs.(@view vals[nzrange(p.H, $j+offs)]))) for j in 1:CD)...),)
		return SVector($((:(N/total) for j in 1:CD)...), )
	end
end
function col_scaling(p::P, s::GeoMean{T,M,N}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A}}
	if N == 0 || M == 0
		return ones(SVector{N, T})
	end
	s.colmin .= typemax(T)
	s.colmax .= typemin(T)
	cols = rowvals(p.H)
	vals = nonzeros(p.H)
	for j = 1:M
		for nz in nzrange(p.H, j) 
			col = cols[nz]
			val = vals[nz]
			@inbounds s.colmin[col] = min(s.colmin[col], abs(val))
			@inbounds s.colmax[col] = max(s.colmax[col], abs(val))
		end
	end
	for i=1:N 
		if s.colmin[i] != typemax(T)
			s.colmin[i] = 1.0/sqrt(s.colmin[i] + s.colmax[i]) 
		else 
			s.colmin[i] = one(T)
		end
	end
	return SVector{N}(s.colmin)
end
@generated function row_scaling(p::P, 
	s::GeoMean{T,M,N}, 
	offs::Int,
	::Union{Reals{T, CD}, Zeros{T, CD}, POCone{T, CD}, NOCone{T, CD}}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},CD}
	return quote 
		cols = rowvals(p.H)
		vals = nonzeros(p.H)
		return SVector($( (:(begin 
			minv,maxv = typemax(T), typemin(T)
			for nz in nzrange(p.H, $j+offs)
				@inbounds val = abs(vals[nz])
				minv = min(minv, val)
				maxv = max(maxv, val)
			end; 
			body = minv*maxv;
			(body > zero(T)) ? 1/sqrt(minv*maxv) : one(T) end) for j in 1:CD)...), )
	end
end
@generated function row_scaling(p::P, 
	s::GeoMean{T,M,N}, 
	offs::Int,
	::Union{SOCone{T, CD}, NSOCone{T, CD}}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},G,CD}
	return quote 
		cols = rowvals(p.H)
		vals = nonzeros(p.H)
		omin, omax = typemax(T), typemin(T)
		for j in 1:CD
			minv,maxv = maxmin_abs(vals[nzrange(p.H, j+offs)])
			omin,omax = min(omin, minv), max(omax, maxv)
		end
		scalar = 1/sqrt(omin*omax)
		return SVector($((:scalar for j in 1:CD)...), )
	end
end
function col_scaling(p::P, s::Equilibration{T,M,N}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A}}
	if N == 0 || M == 0
		return ones(SVector{N, T})
	end
	s.colmax .= typemin(T)
	cols = rowvals(p.H)
	vals = nonzeros(p.H)
	for j = 1:M
		for nz in nzrange(p.H, j)
			col = cols[nz]
			val = vals[nz]
			@inbounds s.colmax[col] = max(s.colmax[col], abs(val))
		end
	end
	s.colmax .= @~ (1.0 ./ s.colmax)
	for i in 1:length(s.colmax)
		@inbounds if s.colmax[i] == zero(T)
			@inbounds s.colmax[i] = one(T)
		end
	end
	return SVector{N}(s.colmax)
end

function max_abs(vect::AbstractArray{T}) where {T}
	if length(vect) == 0 return one(T) end
	maxv = typemin(T)
	for el in vect
		maxv = max(abs(el), maxv)
	end
	return maxv
end
@generated function row_scaling(p::P, 
	s::Equilibration{T,M,N}, 
	offs::Int,
	::Union{Reals{T, CD}, Zeros{T, CD}, POCone{T, CD}, NOCone{T, CD}}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},CD}
	return quote 
		cols = rowvals(p.H)
		vals = nonzeros(p.H)
		return SVector($((:(@inbounds 1.0 / max_abs(@view vals[nzrange(p.H, $j+offs)])) for j in 1:CD)...),)
	end
end
@generated function row_scaling(p::P, 
	s::Equilibration{T,M,N}, 
	offs::Int,
	::Union{SOCone{T, CD}, NSOCone{T, CD}}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},CD}
	return quote 
		cols = rowvals(p.H)
		vals = nonzeros(p.H)
		total = +($((:(@inbounds max_abs(@view vals[nzrange(p.H, $j+offs)])) for j in 1:CD)...),)
		return SVector($((:(1.0/total) for j in 1:CD)...), )
	end
end

function row_scale(p::P, s::State{T,N,M,K,D,A,P,G,S}, row_scaling::SVector{M,T}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},G,S}
	rmul!(p.H, Diagonal(row_scaling))
	p.g .*= row_scaling
	s.row_scale .*= row_scaling
end

function col_scale(p::P, s::State{T,N,M,K,D,A,P,G,S}, col_scaling::SVector{N, T}) where {T,N,M,K,D,A,P <: Problem{T,N,M,K,D,A},G,S}
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