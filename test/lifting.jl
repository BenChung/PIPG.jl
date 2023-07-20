using Graphs

@testset "Build set graph" begin
    problem = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SOCone{Float64, 2}(1.0))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)
    adj_graph, vertex_map, set_map, set_adj = PIPG.build_set_graph(problem)
    @test adj_graph == SimpleGraph{Int64}(6, [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]])
    @test vertex_map == Dict(4 => 4, 2 => 2, 3 => 3, 1 => 1)
    @test isequal(set_map,
        Dict{Int64, Tuple{Int64, PIPG.Cone}}(
            5 => (4, PIPG.SOCone{Float64, 2}(1.0)), 
            4 => (4, PIPG.SOCone{Float64, 2}(1.0)), 
            2 => (2, PIPG.SignCone{Float64, 1}(true)), 
            3 => (3, PIPG.SignCone{Float64, 1}(true)), 
            1 => (1, PIPG.SignCone{Float64, 1}(true))))
    @test set_adj == Dict{Int64, BitSet}(
        4 => BitSet([1, 2]), 
        2 => BitSet([1, 2]), 
        3 => BitSet([1, 2]), 
        1 => BitSet([1, 2]))
end

@testset "Compute K and D" begin 
    problem = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SOCone{Float64, 2}(1.0))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)
    adj_graph, vertex_map, set_map, set_adj = PIPG.build_set_graph(problem)
    k, prm = PIPG.remove_unused_from_k(vertex_map, set_map, [1], problem)
    new_d = PIPG.extract_d_by_index(vertex_map, set_map, [1], problem)
    @test isequal(k, PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 2}(true), PIPG.SOCone{Float64, 2}(1.0))))
    @test prm == [2, 3, 4, 5]
    @test isequal(new_d, 
        PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.MultiHalfspace{Float64, 2}([100.0, 200.0], [10000.0], [50000.0], [2]),)), [1, 2]))

    k, prm = PIPG.remove_unused_from_k(vertex_map, set_map, [2], problem)
    new_d = PIPG.extract_d_by_index(vertex_map, set_map, [2], problem)
    @test isequal(k, PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 2}(true), PIPG.SOCone{Float64, 2}(1.0))))
    @test prm == [1, 3, 4, 5]
    @test isequal(new_d, 
        PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.MultiHalfspace{Float64, 2}([10.0, 30.0], [1200.0], [1000.0], [2]),)), [1, 2]))

    k, prm = PIPG.remove_unused_from_k(vertex_map, set_map, [3], problem)
    new_d = PIPG.extract_d_by_index(vertex_map, set_map, [3], problem)
    @test isequal(k, PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 2}(true), PIPG.SOCone{Float64, 2}(1.0))))
    @test prm == [1, 2, 4, 5]
    @test isequal(new_d, 
        PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.MultiHalfspace{Float64, 2}([1.0, 1.0], [110.0], [2.0], [2]),)), [1, 2]))

    k, prm = PIPG.remove_unused_from_k(vertex_map, set_map, [4], problem)
    new_d = PIPG.extract_d_by_index(vertex_map, set_map, [4], problem)
    @test isequal(k, PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 3}(true), )))
    @test prm == [1, 2, 3]
    @test isequal(new_d, PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.SOCone{Float64, 2}(1.0),)), [1, 2]))
end

@testset "Simple lifting" begin 
    problem = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SOCone{Float64, 2}(1.0))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)

    new_k, new_d, new_h, new_g = PIPG.compute_k_and_d(problem)
    @test isequal(new_k, PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 3}(true), )))
    @test isequal(new_d, PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.SOCone{Float64, 2}(1.0),)), [1, 2]))
    @test isequal(new_h, sparse([1, 2, 1, 2, 1, 2], [1, 1, 2, 2, 3, 3], [-100.0, -200.0, -10.0, -30.0, -1.0, -1.0], 2, 3))
    @test isequal(new_g, [-10000.0, -1200.0, -110.0])
end

@testset "Solve trivial lifted problem" begin 
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SignCone{Float64, 2}(true))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)
	prob = make_prob()

	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9))
	result, niters = PIPG.solve(prob, state)
    println(niters)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01

    prob2 = PIPG.Problem(
        PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 3}(true), )), 
        PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.MultiHalfspace{Float64, 2}([-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [1, 1]),)), [1, 2]), 
        sparse([1, 2, 3, 1, 2, 3], [1, 1, 1, 2, 2, 2], [-100.0, -10.0, -1.0, -200.0, -30.0, -1.0], 3, 2), sparse(Int64[], Int64[], Float64[], 2, 2), [-50.0, -120.0], [-10000.0, -1200.0, -110.0], 0.0)


    state = PIPG.State(prob2, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained(()))
    result, niters = PIPG.solve(prob2, state)
    @test norm(PIPG.primal(state) .- [60, 20]) < 0.01
    
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained((PIPG.SetLifting(),)))
    println(state.preconditioned)
	result, niters = PIPG.solve(prob, state)
    println(niters)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
end


@testset "Solve SOCP" begin 
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SOCone{Float64, 3}(1.0), PIPG.Zeros{Float64, 1}()),), PIPG.Reals{Float64, 3}(), 
        sparse([1.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 1.0 0.0 0.0]), 
        spzeros(3,3), 
        [-1, -1, -1.0], 
        [zeros(3); 1.0], 0.0)

	prob = make_prob()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained(()))
	result, niters = PIPG.solve(prob, state)
    println(niters)
	@test norm(PIPG.primal(state) .- [1.0, 0.707, 0.707]) < 0.01

	prob = make_prob()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9))
	result, niters = PIPG.solve(prob, state)
    println(niters)
	@test norm(PIPG.primal(state) .- [1.0, 0.707, 0.707]) < 0.01

	prob = make_prob()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained((PIPG.SetLifting(),)))
	result, niters = PIPG.solve(prob, state)
    println(niters)
	@test norm(PIPG.primal(state) .- [1.0, 0.707, 0.707]) < 0.01

    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SOCone{Float64, 3}(1.0), PIPG.Zeros{Float64, 1}()),), PIPG.Reals{Float64, 3}(), 
        sparse([2.0 0.0 0.0 ; 0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 1.0 0.0 0.0]), 
        spzeros(3,3), 
        [-1, -1, -1.0], 
        [1.0, 0,0, 1.0], 0.0)

    prob2 = PIPG.Problem(
        PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 3}(true), )), 
        PIPG.PermutedSpace(PIPG.PTSpace{Float64}((PIPG.MultiHalfspace{Float64, 2}([-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [1, 1]),)), [1, 2]), 
        sparse([1, 2, 3, 1, 2, 3], [1, 1, 1, 2, 2, 2], [-100.0, -10.0, -1.0, -200.0, -30.0, -1.0], 3, 2), sparse(Int64[], Int64[], Float64[], 2, 2), [-50.0, -120.0], [-10000.0, -1200.0, -110.0], 0.0)


    state = PIPG.State(prob2, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained(()))
    result, niters = PIPG.solve(prob2, state)
    @test norm(PIPG.primal(state) .- [60, 20]) < 0.01
    
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained((PIPG.SetLifting(),)))
    println(state.preconditioned)
	result, niters = PIPG.solve(prob, state)
    println(niters)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
end
