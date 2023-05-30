# problem
# minimize 1/2 z^t P z + q^T z 
# s.t. H z - g ∈ K, z ∈ D

struct Problem{T,N,M,K<:Cone{T,M},D<:Space{T,N},A<:AbstractArray{T}}
	k::K
	d::D
	H::SparseMatrixCSC{T, Int} # stored as H^T
	P::SparseMatrixCSC{T, Int}
	q::A
	g::A
	c::T
	function Problem(k::K, d::D, H::SparseMatrixCSC{T, Int}, P::SparseMatrixCSC{T, Int}, q::A, g::A, c::T) where {T,N,M,K<:Cone{T,M},D<:Space{T,N},A<:AbstractArray{T}}
		@assert length(q) == N
		@assert length(g) == M
		return new{T,N,M,K,D,A}(k,d,transpose(H),P,q,g,c)
	end
	function Problem(p::Problem{T,N,M,K,D,A}) where {T,N,M,K,D,A}
		return new{T,N,M,K,D,A}(copy(p.k),copy(p.d),copy(p.H),copy(p.P),copy(p.q),copy(p.g),p.c)
	end
end

rows(::Problem{T,N,M}) where {T,N,M} = M
cols(::Problem{T,N,M}) where {T,N,M} = N