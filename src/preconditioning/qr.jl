struct QRPreconditioner <: Preconditioner
end
struct QRPCState{T} <: PreconditionerState{T}
    prow::Vector{Int}
    pcol::Vector{Int}
end

function precondition(p::Problem{T}, pcs::QRPreconditioner) where {T}
    return (Problem(p), QRPCState{T}(zeros(Int, cols(p)), zeros(Int, rows(p))))
end

function precondition!(src::Problem{T}, tgt::Problem{T}, pcs::QRPCState{T}) where T 
    sets = set_ranges(src.k)
    equality_ranges = BitSet()
    other_ranges = BitSet()
    for (set, range) in sets 
        if set isa PIPG.Zeros
            push!.((equality_ranges, ), range)
        else 
            push!.((other_ranges, ), range)
        end
    end
    equality_ranges = collect(equality_ranges)
    other_ranges = collect(other_ranges)
    facted = qr(src.H[:, equality_ranges]) # factorize the parts of H^T that are part of an equality constraint
    #println(facted.prow)
    out_H = [sparse(facted.Q)[:, 1:length(equality_ranges)] src.H[facted.prow, other_ranges]]
    out_g = similar(src.g)
    out_g[equality_ranges] = transpose(facted.R) \ src.g[equality_ranges][facted.pcol]
    # transpose(facted.Q[:,1:18]) * PIPG.primal(state)[facted.prow] - transpose(facted.R) \ scpt3.g[1:18][facted.pcol] = 0
    out_g[other_ranges] = src.g[other_ranges]
    tgt.P .= src.P[facted.prow, facted.prow]
    tgt.q .= src.q[facted.prow]
    tgt.H .= out_H
    tgt.g .= out_g
    copy_to!(tgt.k, src.k)
    copy_to!(tgt.d, src.d)
    pcs.prow[facted.prow] .= eachindex(pcs.prow) 
    pcs.pcol[1:length(equality_ranges)] .= facted.pcol
end
extract(next::Function, pc::QRPCState{T}, ::Primal, index::Int) where T = next(pc.prow[index])
#extract(next::Function, pc::CholeskyPCState{T}, ::Dual, index::Int) where T = pc.row_scale[index] * next(index)   