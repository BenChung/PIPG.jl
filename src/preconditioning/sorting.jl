

struct HyperSort <: Preconditioner
    nstages::Int
end
struct HyperSortState{T} <: PreconditionerState{T}
	# output scalings
    row_perm::AbstractArray{Int}
    col_perm::AbstractArray{Int}
end

function precondition(p::Problem{T}, pc::HyperSort)::Tuple{Problem{T}, PreconditionerState{T}} where T
    col_perm, row_perm = sort_matrix(p.H, pc.nstages) # this is doing way too much copying and it probably makes sense to store H untransposed
    @assert dim(p.k) == length(row_perm)
    @assert dim(p.d) == length(col_perm)
    H = p.H[col_perm, row_perm]
    P = p.P[col_perm, col_perm]
    k = PermutedCone(p.k, row_perm)
    d = PermutedSpace(p.d, col_perm)
    q = p.q[col_perm]
    g = p.g[row_perm]
    return (Problem(k,d, sparse(transpose(H)), P, q, g, p.c), HyperSortState{T}(col_perm, row_perm))
end
function precondition!(src::Problem, tgt::Problem, pc::HyperSortState{T}) where T
    col_perm = pc.col_perm
    row_perm = pc.row_perm
    tgt.H .= src.H[col_perm, row_perm]
    tgt.P .= src.P[col_perm, col_perm]
    tgt.q .= src.q[col_perm]
    tgt.g .= src.g[row_perm]
end
extract(next::Function, pc::HyperSortState{T}, ::Primal, ind) where T = next(pc.col_perm[ind])
extract(next::Function, pc::HyperSortState{T}, ::Dual, ind) where T = next(pc.row_perm[ind])

function hyper_degree(adj, v)
    (verts, edges) = size(adj)
    adjacent = zeros(Int, verts)
    for edge in rowvals(adj[v, :])
        for oppo in rowvals(adj[:, edge])
            adjacent[oppo] = 1
        end
    end
    adjacent[v] = 0
    return sum(adjacent)
end


function refine_pseudoperipheral(adj, v)
    (verts, edges) = size(adj)
    q = Queue{Int}()
    visited = zeros(Bool, verts)
    distances = zeros(Int, verts)

    maxdist = 0

    enqueue!(q, v)
    visited[v] = true

    while !isempty(q)
        v = dequeue!(q)
        for edge in rowvals(adj[v, :])
            for vert in rowvals(adj[:, edge])
                visited[vert] && continue
                visited[vert] = true
                distance = distances[v] + 1
                distances[vert] = distance
                if distance > maxdist
                    maxdist = distance
                end
                enqueue!(q, vert)
            end
        end
    end

    mindeg = typemax(Int)
    equidistant = 0
    minvert = -1
    for (vert, distance) in enumerate(distances)
        if distance < maxdist
            continue
        end
        degree = hyper_degree(adj, v)
        if degree < mindeg
            minvert = vert
            mindeg = degree 
        end
        equidistant += 1
    end
    #println("maximum distance: $maxdist mininmum degree $mindeg equidistant $equidistant")
    return minvert, maxdist
end

function build_hypergraph(g) 
    (I,J,V) = findnz(g)
    res = sparse(I, J, ones(Int, length(V)))
    nonempty = (sum(res,dims = 1) .> 0)[1,:]
    return res[:, nonempty], nonempty
end

function fix(h, p::Vector{Bool}, d::Int)
    (verts, edges) = size(h)
    @assert length(p) == verts
    f = zeros(Bool, verts)
    dist = [typemax(Int) for i=1:verts]
    mark = zeros(Bool, verts)
    q = Queue{Int}()
    for (v, m) in enumerate(p)
        !m && continue
        enqueue!(q, v)
        dist[v] = 0
        mark[v] = true
    end
    while !isempty(q) 
        v = dequeue!(q)
        f[v] = true
        if dist[v] < d - 1
            for edge in rowvals(h[v, :])
                for u in rowvals(h[:, edge])
                    mark[u] && continue # skip marked
                    enqueue!(q, u)
                    dist[u] = dist[v] + 1
                    mark[u] = true
                end
            end
        end
    end
    return f
end

one_hot(n, l) = [i == n ? true : false for i=1:l]
function form_sub_hypergraph(adj, h, n)
    adj[h, n]
end
function HP_BDCO(adj, k::Int, p1::Vector{Bool}, p2::Vector{Bool}; inds = nothing, offs = 0, edge_offs = 0)
    if inds == nothing
        inds = [i for i=1:length(p1)]
    end
    if length(p1) == 0 && length(p2) == 0
        return [], []
    end
    (verts, edges) = size(adj)
    println("iteration size: $verts, $edges")
    edge_assignment = zeros(Int, edges)
    k1 = floor(Int, k/2)
    k2 = ceil(Int, k/2)
    f1 = fix(adj, p1, k1)
    f2 = fix(adj, p2, k2)
    #println("p1: $(inds .=> p1) f1: $f1 k1: $k1 k:$k")
    #println("p2: $(inds .=> p2) f2: $f2 k2: $k2 k:$k")
    fixed = [f1[i] ? 0 : (f2[i] ? 1 : -1) for i=1:verts]
    #println("fixed: $(inds .=> fixed)")
    h = KaHyPar.HyperGraph(adj)
    KaHyPar.fix_vertices(h, 2, fixed)
    partition = KaHyPar.partition(h, 2; imbalance=0.05, configuration = joinpath(@__DIR__ ,"cut_rKaHyPar_sea20.ini")) .+ 1
    #println(inds .=> (partition .+ offs))
    #println()
    #println()
    
    v1 = partition .== 1; n1 = zeros(Bool, edges)
    v2 = partition .== 2; n2 = zeros(Bool, edges)
    cut = zeros(Bool, edges)
    nn1 = nn2 = ncut = 0
    for n = 1:edges 
        if all(v1[rowvals(adj[:, n])])
            n1[n] = true; nn1 += 1
        elseif all(v2[rowvals(adj[:, n])])
            n2[n] = true; nn2 += 1
        else 
            cut[n] = true; ncut += 1
        end
    end
    edge_assignment[n1] .= 1:nn1
    edge_assignment[cut] .= nn1+1:nn1+ncut
    edge_assignment[n2] .= nn1+ncut+1:nn1+ncut+nn2
    #k == 2 && println(edge_assignment)
    #k == 2 && println(edge_assignment .+ edge_offs)
    k == 2 && return partition, edge_assignment
    h1 = adj[v1,n1]
    h2 = adj[v2,n2]

    p11 = p1[v1]
    p22 = p2[v2]
    p12 = zeros(Bool, length(p11))
    p21 = zeros(Bool, length(p22))
    for cut_edge=1:edges
        !cut[cut_edge] && continue
        p12[((adj[:, cut_edge] .== 1) .&& v1)[v1]] .= true
        p21[((adj[:, cut_edge] .== 1) .&& v2)[v2]] .= true   
    end
    if k1 >= 2
        Pi1, e1 = HP_BDCO(h1, k1, p11, p12; inds = inds[v1], offs = offs, edge_offs = edge_offs)
    else
        Pi1 = [1 for i=1:length(p11)]
        e1 = [i for i=1:sum(n1)]
    end
    max_part = maximum(Pi1)
    if k2 >= 2
        Pi2, e2 = HP_BDCO(h2, k2, p21, p22; inds = inds[v2], offs = offs + max_part, edge_offs = edge_offs+sum(n1))
        Pi2 .+= max_part
    else
        Pi2 = [max_part + 1 for i=1:length(p21)]
        e2 = [i for i=1:sum(n2)]
    end
    result = zeros(Int, verts)
    max_n1 = maximum(edge_assignment[n1]; init=0)
    edge_assignment[(!).(n1 .|| n2)] .= 1 + nn1:nn1 + ncut
    edge_assignment[n1] .= e1
    edge_assignment[n2] .= e2 .+ nn1 .+ ncut
    #println("e1: $e1 n1: $n1")
    #println("e2: $e2 n2: $n2")
    #println("late ea: $edge_assignment")
    result[v1] .= Pi1
    result[v2] .= Pi2
    return result, edge_assignment
end

function sort_hypergraph(adj, k)
    p1, _ = refine_pseudoperipheral(adj, 1)
    p2, md = refine_pseudoperipheral(adj, p1)
    (verts, edges) = size(adj)
    (part, edgepart) =  HP_BDCO(adj, k, one_hot(p1, verts), one_hot(p2, verts))
    partitioning = (1:verts) .=> part
    edge_output = zeros(Int, edges)
    edge_output[edgepart] .= 1:edges
    vert_output = map(x->x[1], Base.sort(partitioning, by=el->el[2]))

    counts = zeros(Int, k)
    for vert in 1:verts 
        counts[part[vert]] += 1
    end
    #println(counts)
    #println(vert_output)
    #println(edge_output)
    #println(map(x->x[1], Base.sort(partitioning, by=el->el[2])))
    adj[vert_output,edge_output], vert_output, edge_output
end

function sort_matrix(g, k)
    adj, nonzero_flag = build_hypergraph(g)
    nonzero_cols = findall(nonzero_flag)
    zero_cols = findall((!).(nonzero_flag))
    sorted, vert_perm, edge_perm = sort_hypergraph(adj, k)
    return (vert_perm, [nonzero_cols[edge_perm]; zero_cols])
end