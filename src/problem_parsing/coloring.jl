struct SetLifting <: Preconditioner
end
struct SetLiftingState{T} <: PreconditionerState{T}
    vertex_map::Dict{Int, Int}
    set_map::Dict{Int, Tuple{Int, Cone}}
    d_indices::Vector{Int}
end

struct NormalizationRequirement 
    row::Int
    var::Int 
end

function precondition(p::Problem{T}, ::SetLifting)::Tuple{Problem{T}, SetLiftingState{T}} where T
    # vertex_map maps vertex numbers/ext vertex numbers to rows in the ORIGINAL H/K/g
    # set_map maps row indices in H/K/g to which set they are in K
    # d_indices is the list of vertex numbers that should be in D
    graph, vertex_map, set_map, adj = build_set_graph(p)
    
    function weight_sets((vertex, degree)::Tuple{Int,Int})
        (index, theset) = set_map[vertex_map[vertex]]
        if theset isa SOCone
            return (vertex, degree/10.0)
        end
        return (vertex, degree * 1.0)
    end
    d_indices = independent_set(graph, WeightedDegreeIndependentSet(weight_sets))
    complete_vertex_map(vertex_map, set_map, p)
    # the transformation leaves the order of the primal variables intact - the permutation on D is internal
    new_D, cone_sources, cone_source_map = extract_d(vertex_map, set_map, d_indices, p) 
    # we repermute K by inverse of perm
    new_K, perm = extract_k(set_map, vertex_map, d_indices) 
    # we now have the cone structure right. now to compute the coefficients
    # these are the variable/row combos that are in SOCs. consider a row h z_i - g_i; we need to 0 out g_i and change h to be 1.
    needs_normalization = compute_normalization(vertex_map, set_map, d_indices, p)

    # normalized vars need to be transformed so that z_i' = p.H_ij (z_i + c) for the one j of interest
    var_normalization = Dict(r.var => r.row for r in needs_normalization)
    
    offset = zeros(T, cols(p))
    # conduct the transformation to prepare intermediate_problem for coefficient lifting
    for var = 1:cols(p)
        if haskey(var_normalization, var)
            offset[var] = p.g[var_normalization[var]]/p.H[var, var_normalization[var]]
        end
    end
    intermediate_problem = Problem(p)
    compute_primal_offset(offset, p, intermediate_problem)
    # we have now set all of the gs to 0 for the rows that are in SOCs in D. Now to convert their coefficients to 1.
    scaling = ones(T, cols(p))
    for var = 1:cols(p)
        if haskey(var_normalization, var)
            scaling[var] = 1/intermediate_problem.H[var, var_normalization[var]]
        end
    end
    col_scale(intermediate_problem, intermediate_problem, scaling)
    lift_to(new_D, intermediate_problem, cone_sources, cone_source_map)
 
    new_g = intermediate_problem.g[perm]
    new_H = intermediate_problem.H[:, perm]
    return (Problem(new_K, new_D, new_H, intermediate_problem.P, intermediate_problem.q, new_g, intermediate_problem.c; transpose_h=false), 
            SetLiftingState{T}(vertex_map, set_map, d_indices))
end

function complete_vertex_map(vertex_map, set_map, problem::Problem{T}) where T  
    # add unliftable sets back in
    set_num = maximum(keys(vertex_map); init=0) + 1
    noverlap = 0
    for (setno, (set, rows)) in enumerate(set_rows(problem.k))
        for (rownum, row) in enumerate(rows)
            if haskey(set_map, row) 
                continue 
            end
            local_setno = separable(set) ? rownum - 1 : 0
            set_map[row] = (setno, set)
            if !haskey(vertex_map, set_num + local_setno)
                vertex_map[set_num + local_setno] = row
            end
        end
        set_num += separable(set) ? length(rows) : 1
    end
end

function lift_to(d::PermutedSpace, intermediate_problem, cone_sources, cone_source_map)
    return lift_to(d.cone, d.inverse_permutation, intermediate_problem, cone_sources, cone_source_map)
end
function lift_to(p::PTSpace, inv_perm, intermediate_problem, cone_sources, cone_source_map)
    for (cone_num, cone) in enumerate(p.cones)
        _lift_to(cone, cone_source_map[cone_num], intermediate_problem.H[inv_perm, cone_sources[cone_num]], intermediate_problem.g[cone_sources[cone_num]])
    end
end

function _lift_to(::SOCone{T}, ::SOCone{T}, h, g) where T
    @assert all(nonzeros(h) .≈ one(T))
    @assert all(g .≈ zero(T))
end
_lift_to(cone::PIPG.MultiHalfspace{T}, src::PIPG.SignCone{T}, h, g) where T = __planar_lift(cone, h, g, src.sign ? -one(T) : one(T))
_lift_to(cone::PIPG.MultiPlane{T}, src::PIPG.Zeros{T}, h, g) where T = __planar_lift(cone, h, g, one(T))
function __planar_lift(cone::Union{PIPG.MultiPlane{T}, PIPG.MultiHalfspace{T}}, h, g, sign)  where T
    offs = 1
    nzs = nonzeros(h)
    for row in 1:size(h, 2) # there's a 1-to-1 mapping between input rows and multiplane cones
        nel = offs
        d_norm2 = zero(T)
        for var in nzrange(h, row)
            cone.d[nel] = sign * nzs[var]
            d_norm2 += nzs[var] ^ 2
            nel += 1
        end
        cone.d_norm2[row] = d_norm2
        cone.o[row] = sign * g[row]
        @assert nel-offs == cone.dims[row]
        offs += cone.dims[row]
    end
end

_lift_to(r::Reals, src, h, g) = nothing

function precondition!(src::Problem, tgt::Problem, pc::SetLiftingState) 
    # nothing for now - TODO: implement
end

# this rewrites the problem under th transform z = z' + c. 
# in particular, it reformulates the input problem min 1/2 z^t P z + q^T z + k s.t. H z - g in K, z in D into 
# min 1/2 z'^t P z' + q'^T + k' s.t. H z - g in K, z' in D. K AND D ARE NOT TRANSFORMED
# The algebraic transform it applies is 
#   q' = c^T P + c^T P^T + q
#   k' = c^T P c + q^T c + k
#   g' = g - H c
function compute_primal_offset(c::Vector{T}, input::Problem{T}, tgt::Problem{T}) where T
    @assert cols(input) == length(c)
    
    # compute q
    tgt.q .= input.q
    mul!(tgt.q, input.P, c, 1.0, 1.0)
    mul!(tgt.q, transpose(input.P), c, 1.0, 1.0)

    # compute k
    tgt.c = transpose(c) * input.P * c + dot(input.q, c) + input.c

    # compute g
    tgt.g .= input.g
    mul!(tgt.g, transpose(input.H), c, -1.0, 1.0)
end
extract(next::Function, pc::SetLiftingState, ::Primal, ind) where T = next(ind) # primal is untouched

separable(s::PIPG.Cone) = false
separable(s::Zeros) = true
separable(s::SignCone) = true

separate(s::Zeros{T}) where T = Zeros{T, 1}()
separate(s::SignCone{T}) where T = SignCone{T, 1}(s.sign)
expand(s::Zeros{T}, n::Int) where T = Zeros{T, n}()
expand(s::SignCone{T}, n::Int) where T = SignCone{T, n}(s.sign)

function liftable(c::SOCone, rows, problem::Problem)
    # we need to establish two properties:
    # 1: that each row only has one variable in it
    # 2: that the variables are disjoint between rows
    seen = BitSet()
    for row in rows 
        nvars = length(rowvals(problem.H[:, row]))
        if nvars > 1
            return false 
        end
        if nvars == 1
            var = first(rowvals(problem.H[:, row]))
            if var in seen 
                return false
            end
            push!(seen, var)
        end
    end
    return true
end
liftable(s::SignCone, rows, problem) = true
liftable(s::Zeros, rows, problem) = true
liftable(s::Cone, rows, problem) = false

function build_set_graph(problem::Problem)
    vertex_map = Dict{Int, Int}() # maps vertices to their starting row
    set_adj = Dict{Int, BitSet}()
    set_map = Dict{Int, Tuple{Int, Cone}}()
    set_num = 1
    for (setno, (set, rows)) in enumerate(set_rows(problem.k))
        if !liftable(set, rows, problem)
            continue 
        end
        for (rownum, row) in enumerate(rows)
            local_setno = separable(set) ? rownum - 1 : 0
            set_map[row] = (setno, set)
            if !haskey(vertex_map, set_num + local_setno)
                vertex_map[set_num + local_setno] = row
            end
            this_adj = get!(set_adj, set_num + local_setno, BitSet())
            for var in rowvals(problem.H[:, row])
                push!(this_adj, var)
            end
        end
        set_num += separable(set) ? length(rows) : 1
    end

    adj_graph = SimpleGraph(maximum(keys(set_adj); init=0))
    for (set1, vars1) in set_adj 
        for (set2, vars2) in set_adj
            if set1 <= set2
                continue
            end
            if length(intersect(vars1, vars2)) > 0 # if the two bitsets intersect, then the sets conflict
                add_edge!(adj_graph, set1, set2)
            end
        end
    end
    return adj_graph, vertex_map, set_map, set_adj
end


struct WeightedDegreeIndependentSet{F<:Function}
    get_weight::F
end
function independent_set(g::AbstractGraph{T}, alg::WeightedDegreeIndependentSet{F}) where {T<:Integer, F}
    nvg = nv(g)
    ind_set = Vector{T}()
    sizehint!(ind_set, nvg)
    deleted = falses(nvg)
    degree_queue = PriorityQueue(map(alg.get_weight, enumerate(degree(g))))

    while !isempty(degree_queue)
        v = dequeue!(degree_queue)
        (deleted[v] || has_edge(g, v, v)) && continue
        deleted[v] = true
        push!(ind_set, v)

        for u in neighbors(g, v)
            deleted[u] && continue
            deleted[u] = true
            @inbounds @simd for w in neighbors(g, u)
                if !deleted[w]
                    degree_queue[w] -= 1
                end
            end
        end
    end
    return ind_set
end


function extract_cones_by_row(rows, set_map) where t
    sets = getindex.((set_map,), rows)
    separable_cones = Dict{Cone, BitSet}()
    nonseparable_cones = Pair{Cone, UnitRange{Int}}[]
    for ((_, set),base_row) in Iterators.zip(sets, rows)
        if separable(set)
            set = separate(set)
            push!(get!(separable_cones, set, BitSet()), base_row)
        else 
            push!(nonseparable_cones, set => base_row:base_row + dim(set) - 1)
        end
    end
    return separable_cones, nonseparable_cones
end

function extract_k(set_map, vertex_map, d_indices) where T
    remaining = setdiff(eachindex(vertex_map), d_indices)
    separable_cones, nonseparable_cones = extract_cones_by_row(getindex.((vertex_map,), remaining), set_map)
    separable_len = sum(length.(values(separable_cones)))
    nonseparable_len = sum(dim.(first.(nonseparable_cones)); init=0)

    offset = 1
    row_perm = zeros(Int, separable_len + nonseparable_len)
    cones = Cone[]
    for (cone, rows) in separable_cones
        cone = expand(cone, length(rows))
        row_perm[offset:offset+dim(cone) - 1] .= rows
        push!(cones, cone)
        offset += dim(cone)
    end
    for (cone, rows) in nonseparable_cones 
        row_perm[offset:offset+dim(cone) - 1] .= rows
        push!(cones, cone)
        offset += dim(cone)
    end
    return PTCone{Float64}((cones...,)), row_perm
end


function compute_normalization(vertex_map, set_map, d_indices, problem::Problem{T}) where T
    separable_cones, nonseparable_cones = extract_cones_by_row(getindex.((vertex_map,), d_indices), set_map)
    needs_normalization = NormalizationRequirement[]
    for (cone, rows) in separable_cones 
        push!.((needs_normalization,), normalize_var_inds(cone, rows, problem))
    end
    for (cone, rows) in nonseparable_cones 
        push!.((needs_normalization,), normalize_var_inds(cone, rows, problem))
    end
    return needs_normalization
end

function extract_d(vertex_map, set_map, d_indices, problem::Problem{T}) where T
    separable_cones, nonseparable_cones = extract_cones_by_row(getindex.((vertex_map,), d_indices), set_map)
    offset = 1
    included = BitSet()
    var_perm = zeros(Int, size(problem.H)[1])
    ncone = 1
    cone_sources = Dict{Int, Vector{Int}}() # map cone index to source rows of H/g/K
    cone_source_map = Dict{Int, Cone}()
    cones = Space[]
    for (cone, rows) in separable_cones 
        new_cone, perm = lift(problem, rows, cone)
        var_perm[offset:offset+dim(new_cone) - 1] .= perm
        @assert length(perm) == dim(new_cone)
        push!.((included, ), perm)
        push!(cones, new_cone)
        cone_sources[ncone] = collect(rows)
        cone_source_map[ncone] = cone
        offset += dim(new_cone)
        ncone += 1
    end
    for (cone, rows) in nonseparable_cones 
        new_cone, perm = lift(problem, rows, cone)
        var_perm[offset:offset+dim(new_cone) - 1] .= perm
        @assert length(perm) == dim(new_cone)
        push!.((included, ), perm)
        push!(cones, new_cone)
        cone_sources[ncone] = collect(rows)
        cone_source_map[ncone] = cone
        offset += dim(new_cone)
        ncone += 1
    end
    remaining = size(problem.H)[1] - offset + 1
    if remaining > 0
        cone_source_map[ncone] = Reals{T, remaining}()
        push!(cones, Reals{T, remaining}())
        cone_sources[ncone] = collect(offset:(size(problem.H)[1]))
    end
    for var = 1:(size(problem.H)[1])
        if var in included
            continue
        end
        var_perm[offset] = var
        offset += 1
    end
    inv_perm = zeros(Int, length(var_perm))
    inv_perm[var_perm] .= 1:length(var_perm)
    return PermutedSpace(PTSpace{Float64}((cones...,)), inv_perm), cone_sources, cone_source_map
end

normalize_var_inds(cone, rows, problem) = NormalizationRequirement[]
function normalize_var_inds(cone::SOCone, rows::UnitRange, problem) 
    normalize = NormalizationRequirement[]
    for row in rows 
        push!.((normalize, ), NormalizationRequirement.(row, rowvals(problem.H[:, row])))
    end
    return normalize 
end


function _extract_planes(problem, rows::BitSet, sign::Bool)
    var_offs = 1

    vars = sum(length(rowvals(problem.H[:, row])) for row in rows)
    var_perm = zeros(Int, vars)
    d = zeros(vars)
    d_norm2 = zeros(length(rows))
    o = zeros(length(rows))
    dims = zeros(Int, length(rows))
    ncone = 1

    for row in rows
        row_sign = sign ? -one(eltype(problem.H)) : one(eltype(problem.H))

        coefficient_inds = rowvals(problem.H[:, row])
        var_perm[var_offs:var_offs+length(coefficient_inds)-1] .= coefficient_inds
        d[var_offs:var_offs+length(coefficient_inds)-1] .= row_sign .* problem.H[coefficient_inds, row]
        d_norm2[ncone] = sum(problem.H[coefficient_inds, row] .^ 2)
        o[ncone] = row_sign * problem.g[row]
        dims[ncone] = length(coefficient_inds)
        ncone += 1
        var_offs += length(coefficient_inds)
    end
    return d, o, d_norm2, dims, var_perm, vars
end
function lift(problem, rows::BitSet, cone::SignCone)
    d, o, d_norm2, dims, var_perm, vars = _extract_planes(problem, rows, cone.sign)
    return MultiHalfspace{Float64, vars}(d, o, d_norm2, dims), var_perm
    #=
    r1 = rowvals(problem.H[:, first(rows)])
    println(r1)
    println(map(index_row -> d[first(index_row)], getindex.((var_perm,), r1)))
    println(nonzeros(problem.H[:, first(rows)]))
    =#
end
function lift(problem, rows::BitSet, cone::Zeros)
    d, o, d_norm2, dims, var_perm, vars = _extract_planes(problem, rows, false)
    return MultiPlane{Float64, vars}(d, o, d_norm2, dims), var_perm
end

function lift(problem, rows::UnitRange, cone::SOCone) 
    return cone, [first(rowvals(problem.H[:, row])) for row in rows]
end
