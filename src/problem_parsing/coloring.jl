function build_set_hypergraph(problem::Problem)
    I = [] # row/vertex/PIPG variable indices
    J = [] # column/net/PIPG set indices

    for (set, row) in variables(problem.k)
        for var in rowvals(problem.H[:, row])
            push!(I, var)
            push!(J, set)
        end
    end
    V = [1 for i=1:length(I)]
    return sparse(I, J, V)
end

function make_packing(adj)
    (verts, nets) = size(adj)
    covered = BitSet()
    lifted = BitSet()
    for net in sort(1:nets; by=net -> length(rowvals(adj[:, net])))
        verts = rowvals(adj[:, net])
        any(v->v in covered, verts) && continue
        push!(lifted, net)
        push!.((covered, ), verts)
    end
    return lifted
end