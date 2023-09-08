struct CholeskyPreconditioner <: Preconditioner
    pcf::Float64
end
struct CholeskyPCState{T} <: PreconditionerState{T}
    pcf::Float64
    scaling::Vector{T}
	col_scale::Vector{T}
	row_scale::Vector{T}
	col_temp::Vector{T}
	row_temp::Vector{T}
end

function precondition(p::Problem{T}, pcs::CholeskyPreconditioner) where {T}
    I, J, _ = findnz(p.P)
    if any(I .!= J) # are there any non-diagonal nonzeros? If so, cholesky can't be used
        throw("Cannot use cholesky factorization if there are off-diagonal elements in P")
    end
    return (Problem(p), CholeskyPCState(pcs.pcf, zeros(T, cols(p)), zeros(T, cols(p)), zeros(T, rows(p)), zeros(T, cols(p)), zeros(T, rows(p))))
end

function precondition!(src::Problem{T}, tgt::Problem{T}, pcs::CholeskyPCState{T}) where T 
    for i in eachindex(pcs.scaling)
        value = src.P[i,i]
        pcs.scaling[i] = value == zero(T) ? one(T) : sqrt(value)
    end
    #constrain!(typeof(src.d), scaling, 0) # TODO: ignore primals that are in cones that are not scaled homogenously
    
	scale_cone!(tgt.d, src.d, pcs.scaling, 0)
    tgt.H .= src.H
    nonzeros(tgt.P) .= one(T) ./ pcs.pcf
    for i in eachindex(pcs.scaling)
        pcs.scaling[i] = one(T) / pcs.scaling[i]
    end
	lmul!(Diagonal(pcs.scaling), tgt.H) # note that H is transposed so this whole thing is transposed
	tgt.q .= src.q .* pcs.scaling ./ pcs.pcf
    
    pcs.col_scale .= one(T)
    pcs.row_scale .= one(T)
    compute_norms!(pcs.col_temp, pcs.row_temp, tgt.H, pcs.col_scale, pcs.row_scale)
    cleanup_norm!(pcs.row_temp)
    pcs.row_scale .= one(T) ./ pcs.row_temp
    constrain!(typeof(src.k), pcs.row_scale, 0)
    scale_cone!(tgt.k, src.k, pcs.row_scale, 0)
	rmul!(tgt.H, Diagonal(pcs.row_scale))
	tgt.g .= src.g .* pcs.row_scale
end
extract(next::Function, pc::CholeskyPCState{T}, ::Primal, index::Int) where T = pc.scaling[index] * next(index)
#extract(next::Function, pc::CholeskyPCState{T}, ::Dual, index::Int) where T = pc.row_scale[index] * next(index)   