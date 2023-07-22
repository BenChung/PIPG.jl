struct ColumnNormalization <: Preconditioner end
struct ColNormState{T} <: PreconditionerState{T}
	# output scalings
	col_scale::Vector{T}
	row_scale::Vector{T}
	col_temp::Vector{T}
	row_temp::Vector{T}
	ColNormState(p::Problem{T}) where T = 
    new{T}(zeros(T, cols(p)), zeros(T, rows(p)),
           zeros(T, cols(p)), zeros(T, rows(p)))
end


function precondition(p::Problem, pc::ColumnNormalization)
	return (Problem(p), ColNormState(p))
end


function compute_norms!(col_norm, row_norm, mat, col_scale, row_scale)
    cols = rowvals(mat)
    vals = nonzeros(mat)
    m, n = size(mat)
    col_norm .= zero(eltype(typeof(col_norm)))
    row_norm .= zero(eltype(typeof(row_norm)))
    for row = 1:n
        for i in nzrange(mat, row)
            col = cols[i]
            value_here = (col_scale[col] * row_scale[row] * vals[i])^2
            col_norm[col] += value_here
            row_norm[row] += value_here
        end
    end
    @.. col_norm .= sqrt.(col_norm)
    @.. row_norm .= sqrt.(row_norm)
end

function cleanup_norm!(norm)
    for i in eachindex(norm)
        norm[i] = (isinf(norm[i]) || isnan(norm[i]) || norm[i] == 0.0) ? 1.0 : norm[i]
    end
end

function precondition!(src::Problem, tgt::Problem, pc::ColNormState{T}) where T
	# setup the internal state
    pc.row_scale .= one(T)
    pc.col_scale .= one(T)
    compute_norms!(pc.col_temp, pc.row_temp, src.H, pc.col_scale, pc.row_scale)
    cleanup_norm!(pc.row_temp); 
    cleanup_norm!(pc.col_temp); 
    pc.row_scale .= pc.row_scale ./ pc.row_temp
    pc.col_scale .= pc.col_scale ./ pc.col_temp
    
    constrain!(typeof(src.d), pc.col_scale, 0)
    constrain!(typeof(src.k), pc.row_scale, 0)
	scale_cone!(tgt.d, src.d, pc.col_scale, 0)
	scale_cone!(tgt.k, src.k, pc.row_scale, 0)
	col_scale(tgt, src, pc.col_scale)
	row_scale(tgt, src, pc.row_scale)
end
extract(next::Function, pc::ColNormState{T}, ::Primal, index::Int) where T = pc.col_scale[index] * next(index)
extract(next::Function, pc::ColNormState{T}, ::Dual, index::Int) where T = pc.row_scale[index] * next(index)
