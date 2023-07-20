using PIPG
import MathOptInterface
using LinearAlgebra
using StaticArrays
using SparseArrays
using Profile
const MOI = MathOptInterface
using Test
@testset "Projections" begin
    long_inp = [zeros(6)..., 0, -4.0, 4.0, -10.0]
    res = zeros(4)
    PIPG.project!(res, 1, PIPG.SignCone{Float64, 4}(true), SVector(0.0, -4.0, 4.0, -10.0))
    @test res == MVector(0.0, 0.0, 4.0, 0.0)
    res = zeros(4)
    PIPG.project!(res, 1, PIPG.SignCone{Float64, 4}(false), SVector(0.0, -4.0, 4.0, -10.0))
    @test res == MVector(0.0, -4.0, 0.0, -10.0)
    PIPG.project!(res, 1, PIPG.polar(PIPG.SignCone{Float64, 4}(true)), SVector(0.0, -4.0, 4.0, -10.0))
    @test res == MVector(0.0, -4.0, 0.0, -10.0)
    res = zeros(10)
    PIPG.project!(res, 7, PIPG.SignCone{Float64, 4}(true), long_inp)
    @test res == [zeros(6); [0, 0, 4, 0]]

    soc_project = [7.744562646538029, -2.696310623822791, 2.696310623822791, -6.740776559556978]
    res = zeros(4)
    PIPG.project!(res, 1, PIPG.SOCone{Float64, 4}(1.0), [4.0, -4.0, 4.0, -10.0])
    @test res ≈ soc_project
    res = zeros(10)
    long_inp = [zeros(6)..., 4.0, -4.0, 4.0, -10.0]
    PIPG.project!(res, 7, PIPG.SOCone{Float64, 4}(1.0), long_inp)
    @test res ≈ (vcat(zeros(6), soc_project))

    nsoc_project = [-7.744562646538029, -2.696310623822791, 2.696310623822791, -6.740776559556978]
    res = zeros(4)
    PIPG.project!(res, 1, PIPG.SOCone{Float64, 4}(-1.0), [-4.0, -4.0, 4.0, -10.0])
    @test res ≈ nsoc_project
    res = zeros(10)
    long_inp = [zeros(6)..., -4.0, -4.0, 4.0, -10.0]
    PIPG.project!(res, 7, PIPG.SOCone{Float64, 4}(-1.0), long_inp)
    @test res ≈ (vcat(zeros(6), nsoc_project))

    combined = [-1.0, 1.0, -1.0, 1.0]
    long_combined = [zeros(6); combined]
    result = [0.0, 1.0, -1.0, 0.0]
    cone = PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 2}(true), PIPG.SignCone{Float64, 2}(false)))
    res = zeros(4)
    PIPG.project!(res, 1, cone, combined)
    @test res ≈ result
    res = zeros(10)
    combined = [zeros(6)..., -1.0, 1.0, -1.0, 1.0]
    PIPG.project!(res, 7, cone, combined)
    @test res ≈ (vcat(zeros(6), result))
    
    cone = PIPG.HalfspaceCone{Float64, 2}([1.0, 0.0], 1.0)
    res = zeros(2)
    PIPG.project!(res, 1, cone, [0.0, 1.0])
    @test res ≈ [0.0, 1.0]
    PIPG.project!(res, 1, cone, [2.0, 1.0])
    @test res ≈ [1.0, 1.0]
    res = zeros(10)
    combined = [zeros(8); 2.0; 1.0]
    PIPG.project!(res, 9, cone, combined)
    @test res ≈ [zeros(8); 1.0; 1.0]

    
    cone_inner = PIPG.SOCone{Float64, 4}(-1.0)
    perm = [4, 3, 2, 1]
    cone = PIPG.PermutedCone(cone_inner, perm)
    res = zeros(4)
    arg = [-4.0, -4.0, 4.0, -10.0][perm]
    result = [-7.744562646538029, -2.696310623822791, 2.696310623822791, -6.740776559556978][perm]
    PIPG.project!(res, 1, cone, arg)
    @test res ≈ result

    cone_inner = PIPG.SOCone{Float64, 4}(-1.0)
    perm = [2, 4, 1, 3]
    cone = PIPG.PermutedCone(cone_inner, perm)
    res = zeros(4)
    arg = [-4.0, -4.0, 4.0, -10.0][perm]
    result = [-7.744562646538029, -2.696310623822791, 2.696310623822791, -6.740776559556978][perm]
    PIPG.project!(res, 1, cone, arg)
    @test res ≈ result


    a = PIPG.HalfspaceCone{Float64, 2}([1.0, 2.0], 1.0)
    b = PIPG.HalfspaceCone{Float64, 3}([3.0, 4.0, 5.0], -1.0)
    combined = PIPG.PTCone{Float64}((a, b))
    arg = [1.0, 1.0, 1.0, 1.0, 1.0]
    res = zeros(5)
    PIPG.project!(res, 1, combined, arg)
    res2 = zeros(5)
    equivalent_cone = PIPG.MultiHalfspace{Float64, 5}([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, -1.0], [5.0, 50.0], [2, 3])
    PIPG.project!(res2, 1, combined, arg)
    @test res2 ≈ res
    println(res2)

    arg = [1.0, 1.0, 1.0, 1.0, 1.0]
    res = zeros(5)
    cone = PIPG.MultiPlane{Float64, 5}([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, -1.0], [5.0, 50.0], [2, 3])
    PIPG.project!(res, 1, cone, arg)
    println(res)
    @test false
end

include("lifting.jl")
include("hypergraph_algs.jl")

@testset "Intervals" begin 
    opt = PIPG.Optimizer(ϵ=1e-7)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    x = MOI.add_variables(model, 2)
    MOI.set(model, 
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
        MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.([1, 1.0], x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[1])], 0.0),
        MOI.Interval(0.0, 1.0))
        MOI.add_constraint(model,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x[2])], 0.0),
            MOI.Interval(0.0, 1.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [1.0, 0.5]) < 0.01
end

@testset "Ruiz Equilibration" begin 
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

    make_prob_plane() = PIPG.Problem(PIPG.PTCone{Float64}(( 
        PIPG.SignCone{Float64, 1}(true), 
        PIPG.SignCone{Float64, 1}(true), 
        PIPG.SignCone{Float64, 2}(true))), PIPG.HalfspaceCone{Float64, 2}([100.0, 200.0], 10000.0), 
        sparse([-10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [ -1200.0, -110, -0.0, -0.0], 0.0)
    prob = make_prob_plane()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9))
	result, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
    println(niters)
end
@testset "Chained Preconditioning" begin 
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SignCone{Float64, 2}(true))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)
	prob = make_prob()

	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained((PIPG.Ruiz(),)))
	result, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
    
	prob = make_prob()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained((PIPG.Ruiz(),PIPG.Ruiz())))
	result, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
    
	prob = make_prob()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.Chained(()))
	result, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
end
@testset "Hypergraph permutation preconditioning" begin 
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SignCone{Float64, 2}(true))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)
	prob = make_prob()

	state = PIPG.State(prob, PIPG.xPIPG(0.0001, 0.9); preconditioner=PIPG.HyperSort(2))
	result, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01
end

@testset "Ruiz Equilibration SOCP" begin 
    make_prob() = PIPG.Problem(PIPG.SOCone{Float64, 2}(1.0), PIPG.PTSpace{Float64}((PIPG.Equality{Float64, 1}([1.0]), PIPG.Reals{Float64, 1}())), 
        sparse([1.0 0.0; 0.0 2.0]), 
        spzeros(2,2), 
        -1 .* [0.0, 1.0], 
        [0.0, 0.0], 0.0)
    prob = make_prob()
    γ = 0.9
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, γ))
	result, niters = PIPG.solve(prob, state)
    @test norm(PIPG.primal(state) .- [1.0, 0.5]) < 0.01
    
	γ = 0.9
	# QP example
    function make_qp()
    	P_ex = spdiagm([1/MathConstants.e, 2.0])
    	q = [1/MathConstants.e,2.0]
    	H_ex = sparse([-1.0 -1.0])
    	g = [0.0]
    	return PIPG.Problem(PIPG.SignCone{Float64, 1}(false), PIPG.Reals{Float64, 2}(), H_ex, P_ex, q, g, 0.0)
    end
	
    # no scaling
    prob = make_qp()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, γ))
	_, niters = PIPG.solve(prob, state)
    result = PIPG.primal(state)
	@test norm(result .- [0.689, -0.689]) < 0.001

    
    function make_basic_qp()
    	P_ex = sparse([1, 2, 3, 1, 2, 3, 1, 2, 3], [1, 1, 1, 2, 2, 2, 3, 3, 3], [5.0, -2.0, -1.0, -2.0, 4.0, 3.0, -1.0, 3.0, 5.0], 3, 3)
    	q = [2.0, -35.0, -47.0]
    	H_ex = sparse(Int64[], Int64[], Float64[], 0, 3)
    	g = Float64[]
    	return PIPG.Problem(PIPG.SignCone{Float64, 0}(false), PIPG.Reals{Float64, 3}(), H_ex, P_ex, q, g, 5.0)
    end
	
    # no scaling
    prob = make_basic_qp()
	state = PIPG.State(prob, PIPG.xPIPG(0.0001, γ))
	_, niters = PIPG.solve(prob, state)
    result = PIPG.primal(state)
	@test norm(result .- [3, 5, 7]) < 0.001
end
@testset "Moving from K to D" begin 
	γ = 0.9
    prob = PIPG.Problem(PIPG.SignCone{Float64, 3}(false), PIPG.SignCone{Float64, 2}(true), 
        sparse([1.0 2.0; 1.0 3.0; 1.0 1.0]), 
        spzeros(2,2), 
        [-50.0, -120.0], 
        [100.0, 120.0, 110.0], 0.0)
    state = PIPG.State(prob, PIPG.xPIPG(0.0001, γ; iters = 2000000))
    _, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01

    opt = PIPG.Optimizer(ϵ=1e-7)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    c = [50.0, 120.0]
    x = MOI.add_variables(model, length(c))
    MOI.set(model, 
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
        MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.(c, x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], x), 0.0),
        MOI.LessThan(100.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 3.0], x), 0.0),
        MOI.LessThan(120.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.LessThan(110.0))
    tc = MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
    MOI.set(model, PIPG.SetMembership(), tc, PIPG.D)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [60, 20]) < 0.01
end

@testset "Masses raw" begin
    t = 20
    l = 32
    x0 = repeat([0.1, 0.0], l)
    Pd = 3*l*t + 2*l
    P = sparse(1.0I, Pd, Pd)
    q = zeros(3*t*l + 2*l)
    Hd = 2*l*t
    L = Tridiagonal(repeat([-1], l-1),repeat([2], l), repeat([-1], l-1))
    Δ = 0.1
    Ac = [zeros(l,l) 1.0I(l); -L zeros(l,l)]
    A = exp(collect(Δ*Ac))
    Bc = [zeros(l,l); 1.0I(l)]
    B = Ac\((A - I(2*l))*Bc)
    H = [([zeros(2*l*t, 2*l) sparse(I, Hd, Hd)]-[kron(I(t), A) zeros(2*l*t, 2*l)]) -kron(I(t), B)]
    g = zeros(2*l*t)
    K = PIPG.Zeros{Float64, 2*t*l}()
    ρx = 1.0
    ρu = 0.5
    Xes = [PIPG.InfBound{Float64, 2*l}([ρx for i=1:2*l], zeros(2*l)) for j=1:t-1]
    Us = [PIPG.InfBound{Float64, l}([ρu for i=1:l], zeros(l)) for j=1:t]
    D = PIPG.PTSpace{Float64}((PIPG.Equality{Float64, 2*l}(x0), Xes..., PIPG.Zeros{Float64, 2*l}(), Us...))
    prob = PIPG.Problem(K, D, sparse(H), sparse(P), q, g, 0.0)
	state = PIPG.State(prob, PIPG.xPIPG(1e-7, 0.95; ρ=1.55, ω=195.0))
	_, niters = PIPG.solve(prob, state)
    println(niters)
end


@testset "Basic optimizer examples" begin
	γ = 0.9
	# QP example
    function make_qp()
    	P_ex = spdiagm([1/MathConstants.e, 2.0])
    	q = [1/MathConstants.e,2.0]
    	H_ex = sparse([-1.0 -1.0])
    	g = [0.0]
    	return PIPG.Problem(PIPG.SignCone{Float64, 1}(false), PIPG.Reals{Float64, 2}(), H_ex, P_ex, q, g, 0.0)
    end
	
    # no scaling
    prob = make_qp()
    state = PIPG.State(prob, PIPG.xPIPG(0.0001, γ; iters = 2000000))
    _, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [0.689, -0.689]) < 0.001

	# LP example
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), PIPG.SignCone{Float64, 1}(true), 
        PIPG.SignCone{Float64, 2}(true))), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* [50.0, 120.0], 
        [-10000.0, -1200.0, -110, -0.0, -0.0], 0.0)
	prob = make_prob()
    state = PIPG.State(prob, PIPG.xPIPG(0.0001, γ; iters = 2000000))
    _, niters = PIPG.solve(prob, state)
	@test norm(PIPG.primal(state) .- [60, 20]) < 0.01

    #=
    # geometric scaling
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.GeoMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01
    println(niters)

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.GeoMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001

    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.GeoMean(prob), ))
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01
    println(niters)

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.GeoMean(prob), ))
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001


    # arithmetic scaling
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.ArithMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.ArithMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.ArithMean(prob), ))
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.ArithMean(prob), ))
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001


    # equilibration scaling
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.Equilibration(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.Equilibration(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001

    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.Equilibration(prob), ))
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.Equilibration(prob), ))
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001

    # default scaling (geomean + equlibration)
    prob = make_prob()
    state = PIPG.State(prob)
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob)
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001

    # mutable scaling (geomean + equlibration)
    prob = make_prob()
    state = PIPG.State(prob)
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0])
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob)
    PIPG.scale!(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, [0.0, 0.0], [0.0])
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001
    =#
end
#=
@testset "Profiling" begin 
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.POCone{Float64, 1}(), PIPG.POCone{Float64, 1}(), PIPG.POCone{Float64, 1}(), 
        PIPG.POCone{Float64, 2}())), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* MVector(50.0, 120.0), 
        MVector(-10000.0, -1200.0, -110, -0.0, -0.0), 0.0)
    prob = make_prob()
    state = PIPG.State(prob)
    γ = 0.9
    PIPG.scale(prob, state)
    #scale_a_bunch(prob,state,niters) = for i=1:niters PIPG.scale(prob, state) end
    #scale_a_bunch(prob, state, 1)
    solve_a_bunch(prob, state, niters) = 
        for i=1:niters 
            PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, SVector(0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0, 0.0)) 
        end
    solve_a_bunch(prob, state, 1)
    Profile.clear()
    @profile solve_a_bunch(prob, state, 10000)
    @time solve_a_bunch(prob, state, 10000)
    Profile.print(C=true)
    @test false
end
=#
function basic_lp()
    opt = PIPG.Optimizer(ϵ=1e-7)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    c = [50.0, 120.0]
    x = MOI.add_variables(model, length(c))
    MOI.set(model, 
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
        MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.(c, x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([100.0, 200.0], x), 0.0),
        MOI.LessThan(10000.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([10.0, 30.0], x), 0.0),
        MOI.LessThan(1200.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.LessThan(110.0))
    MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
    return opt, model, x
end

@testset "Basic LP" begin
    opt,model,x = basic_lp()
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [60, 20]) < 0.01
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 5400

    opt,model,x = basic_lp()
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [60, 20]) < 0.01
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 5400
end

@testset "Basic LP - VectorAffineFunction" begin
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    c = [50.0, 120.0]
    x = MOI.add_variables(model, length(c))
    o = MOI.set(model, 
    	MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
    	MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.(c, x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    cv = MOI.add_constraint(model,
    	MOI.VectorAffineFunction(MOI.VectorAffineTerm.(
    	[1,1,2,2,3,3],
    	[MOI.ScalarAffineTerm.([100.0, 200.0], x);
    	 MOI.ScalarAffineTerm.([10.0, 30.0], x);
    	 MOI.ScalarAffineTerm.([1.0, 1.0], x)]), [-10000.0, -1200.0, -110.0]), 
    	MOI.Nonpositives(3))
    pos = MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [60, 20]) < 0.01
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 5400
end

@testset "Basic SOCP" begin
    opt,model,x = basic_lp()
    tip = MOI.add_variable(model)
    MOI.add_constraint(model, tip, MOI.EqualTo(50.0))
    MOI.add_constraint(model, MOI.VectorOfVariables([tip, x...]), 
        MOI.SecondOrderCone(3)) # |x|_2 < 50.0
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [42.881, 25.706]) < 0.01
end

@testset "Trivial QP" begin
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
	P_ex = spdiagm([1/MathConstants.e, 2.0])
	q = SVector(1/MathConstants.e,2.0)
	H_ex = sparse([-1.0 -1.0])
	g = SVector(0.0)

    x = MOI.add_variables(model, 2)
    MOI.set(model, 
    	MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
    	MOI.ScalarQuadraticFunction(
    		MOI.ScalarQuadraticTerm{Float64}.([1/MathConstants.e, 2.0], x, x), 
    		MOI.ScalarAffineTerm.([1/MathConstants.e,2.0], x), 0.0))
    MOI.add_constraint(model,
    	MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
    	MOI.GreaterThan(0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
	@test norm(MOI.get.(model, MOI.VariablePrimal(), x) .- [0.6893, -0.6893]) < 0.001
end

@testset "Basic QP" begin
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )

    x = MOI.add_variables(model, 3)
    MOI.set(model, 
    	MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
    	MOI.ScalarQuadraticFunction(
    		MOI.ScalarQuadraticTerm{Float64}.([5.0, -2, -1, 4, 3, 5], 
    			[repeat([x[1]], 3); repeat([x[2]], 2); repeat([x[3]], 1)], 
    			[x; x[2:3]; x[3]]), 
    		MOI.ScalarAffineTerm.([2.0, -35, -47], x), 5.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
	@test norm(MOI.get.(model, MOI.VariablePrimal(), x) .- [3, 5, 7]) < 0.001
end

@testset "LP scalar 2" begin 
    opt = PIPG.Optimizer(niters=100000000)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    c = [50.0, 120.0]
    x = MOI.add_variables(model, length(c))
    MOI.set(model, 
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
        MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.(c, x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([100.0, 200.0], x), 0.0),
        MOI.LessThan(10000.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([10.0, 30.0], x), 0.0),
        MOI.LessThan(1200.0))
    sum_cstr = MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.LessThan(60.0))
    lb_cstr = MOI.add_constraint(model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(0.3, x[1])], 0.0), MOI.GreaterThan(10.0))
    MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))

    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [33.333, 26.666]) < 0.01
end
#=
prob = PIPG.Problem(
           PIPG.PTCone{Float64}((PIPG.POCone{Float64, 1}(), PIPG.NOCone{Float64, 5}())),
           PIPG.Reals{Float64, 2}(),
sparse([   0.3    0
         100.0  200.0
          10.0   30.0
           1.0    1.0
          -1.0    0
           0     -1.0]), spzeros(2,2) , MVector(-50.0, -120.0), MVector(10.0, 10000.0, 1200.0, 60.0, -0.0, -0.0), 0.0)
state = PIPG.State(prob, PIPG.LogDiagnostics{Float64}())
α = PIPG.compute_α(prob, 0.9)
PIPG.pipg(prob, state, 100000000, α, 1e-5, SVector{2, Float64}(zeros(2)), SVector{6, Float64}(zeros(6)))
# baseline run (opt) 15270124


prob = PIPG.Problem(
           PIPG.PTCone{Float64}((PIPG.POCone{Float64, 1}(), PIPG.NOCone{Float64, 3}())),
           PIPG.POCone{Float64, 2}(),
sparse([   0.3    0
         100.0  200.0
          10.0   30.0
           1.0    1.0]), spzeros(2,2) , MVector(-50.0, -120.0), MVector(10.0, 10000.0, 1200.0, 60.0), 0.0)
PIPG.State(prob, PIPG.LogDiagnostics{Float64}(), PIPG.GeoMean{Float64, 4, 2}())
α = PIPG.compute_α(prob, 0.9)
PIPG.pipg(prob, state, 100000000, α, 1e-5, SVector{2, Float64}(zeros(2)), SVector{4, Float64}(zeros(4)))


prob = PIPG.Problem(
           PIPG.PTCone{Float64}((PIPG.POCone{Float64, 1}(), PIPG.NOCone{Float64, 3}())),
           PIPG.POCone{Float64, 2}(),
sparse([ 0.3 0
         .1  .2
         .1   .3
         1.0    1.0]), spzeros(2,2) , MVector(-50.0, -120.0), MVector(10.0, 10.0, 12.0, 60.0), 0.0)
state = PIPG.State(prob, PIPG.LogDiagnostics{Float64}())
α = PIPG.compute_α(prob, 0.9)
PIPG.pipg(prob, state, 100000000, α, 1e-5, SVector{2, Float64}(zeros(2)), SVector{4, Float64}(zeros(4)))
=#

@testset "LP scalar modification" begin
    opt = PIPG.Optimizer()
    opt.niters *= 100
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    c = [50.0, 120.0]
    x = MOI.add_variables(model, length(c))
    MOI.set(model, 
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
        MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.(c, x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([100.0, 200.0], x), 0.0),
        MOI.LessThan(10000.0))
    MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([10.0, 30.0], x), 0.0),
        MOI.LessThan(1200.0))
    sum_cstr = MOI.add_constraint(model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
        MOI.LessThan(110.0))
    lb_cstr = MOI.add_constraint(model,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[1])], 0.0), MOI.GreaterThan(0.0))
    MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
    MOI.optimize!(model)
    MOI.modify(model, sum_cstr, MOI.ScalarConstantChange(60.0))
    MOI.modify(model, lb_cstr, MOI.ScalarConstantChange(10.0))
    MOI.optimize!(model)
    change = MOI.ScalarCoefficientChange(x[1], 0.3)
    MOI.modify(model, lb_cstr, change)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [33.333, 26.666]) < 0.01
end

@testset "LP vector modification" begin
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    c = [50.0, 120.0]
    x = MOI.add_variables(model, length(c))
    MOI.set(model, 
    	MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), 
    	MOI.ScalarQuadraticFunction(MOI.ScalarQuadraticTerm{Float64}[], MOI.ScalarAffineTerm.(c, x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    cv = MOI.add_constraint(model,
    	MOI.VectorAffineFunction(MOI.VectorAffineTerm.(
    	[1,1,2,2,3,3],
    	[MOI.ScalarAffineTerm.([100.0, 200.0], x);
    	 MOI.ScalarAffineTerm.([10.0, 30.0], x);
    	 MOI.ScalarAffineTerm.([1.0, 1.0], x)]), [-10000.0, -1200.0, -110.0]), 
    	MOI.Nonpositives(3))
    MOI.add_constraint(model, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
    MOI.optimize!(model)
    cst = MOI.VectorConstantChange([-10000.0, -1200.0, -60.0])
    MOI.modify(model, cv, cst)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [30, 30]) < 0.01
    cst = MOI.VectorConstantChange([-10000.0, -600.0, -60.0])
    MOI.modify(model, cv, cst)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test norm(MOI.get.(model, MOI.VariablePrimal(), x)  .- [60, 0]) < 0.01
end

@testset "Conic clone" begin
    T = Float64
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    abx, rsoc =
        MOI.add_constrained_variables(model, MOI.RotatedSecondOrderCone(4))
    a, b, x1, x2 = abx
    x = [x1, x2]
    vc1 = MOI.add_constraint(model, a, MOI.EqualTo(T(1 // 2)))
    # We test this after the creation of every `VariableIndex` constraint
    # to ensure a good coverage of corner cases.
    @test vc1.value == a.value
    vc2 = MOI.add_constraint(model, b, MOI.EqualTo(T(1)))
    @test vc2.value == b.value
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(T(1), x), T(0)),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    MOI.optimize!(model)
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
    @test ≈(MOI.get(model, MOI.VariablePrimal(), a), T(1 // 2), atol=0.01)
    @test ≈(MOI.get(model, MOI.VariablePrimal(), b), 1, atol=0.01)
end

@testset "Geometric mean cone clone" begin 
    opt = PIPG.Optimizer()
    #opt.scaling = PIPG.NoScaling
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    n = 3
    t = MOI.add_variable(model)
    x = MOI.add_variables(model, n)
    vov = MOI.VectorOfVariables([t; x])
    gmc = MOI.add_constraint(model, vov, MOI.GeometricMeanCone(n + 1))
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(Float64(1), x), Float64(0)),
        MOI.LessThan(Float64(n)),
    )
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(Float64(1), t)], Float64(0)),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)
    @test ≈(MOI.get(model, MOI.ObjectiveValue()), 1.0, atol=0.01)
end

@testset "Geometric mean cone clone2" begin 
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    T = Float64
    n = 3
    n = 9
    t = MOI.add_variable(model)
    x = MOI.add_variables(model, n)
    vov = MOI.VectorOfVariables([t; x])
    gmc = MOI.add_constraint(model, vov, MOI.GeometricMeanCone(n + 1))
    cx =
        Vector{MOI.ConstraintIndex{MOI.ScalarAffineFunction{T},MOI.EqualTo{T}}}(
            undef,
            n,
        )
    for i in 1:n
        cx[i] = MOI.add_constraint(
            model,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(T(1), x[i])], T(0)),
            MOI.EqualTo(T(1)),
        )
    end
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(T(1), t)], T(0)),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    MOI.optimize!(model)

    @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
end

@testset "Infeasible SOC2" begin
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    T = Float64
    t = MOI.add_variable(model)
    x = MOI.add_variables(model, 2)
    MOI.add_constraints(model, x, MOI.GreaterThan.(T[3, 4]))
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), t)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
end
@testset "Infeasible SOC" begin
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    T = Float64
    t = MOI.add_variable(model)
    x = MOI.add_variables(model, 2)
    MOI.add_constraints(model, x, MOI.GreaterThan.(T[3, 4]))
    c_soc = MOI.add_constraint(
        model,
        MOI.VectorOfVariables([t; x]),
        MOI.SecondOrderCone(3),
    )
    MOI.set(model, MOI.ObjectiveFunction{MOI.VariableIndex}(), t)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test ≈(MOI.get(model, MOI.VariablePrimal(), t), T(5), atol=0.01)
    MOI.delete(model, c_soc)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
end

@testset "Quadratic test copies" begin 
    opt = PIPG.Optimizer()
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    T = Float64
    obj_attr = MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{T}}()
    x = MOI.add_variables(model, 2)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    vc1 = MOI.add_constraint(model, x[1], MOI.GreaterThan(T(1)))
    @test vc1.value == x[1].value
    vc2 = MOI.add_constraint(model, x[2], MOI.GreaterThan(T(2)))
    @test vc2.value == x[2].value
    # Basic model
    # min x^2 + y^2 | x>=1, y>=2
    MOI.set(
        model,
        obj_attr,
        MOI.ScalarQuadraticFunction(
            MOI.ScalarQuadraticTerm.(T(2), x, x),  # quad
            MOI.ScalarAffineTerm{T}[],  # affine terms
            T(0),  # constant
        ),
    )
    MOI.optimize!(model)
    variable_primal = [(x[1], T(1)), (x[2], T(2))]
    for (index, solution_value) in variable_primal
        @test isapprox(
            MOI.get(model, MOI.VariablePrimal(), index),
            solution_value, atol=0.01
        )
    end
    @test isapprox(
            MOI.get(model, MOI.ObjectiveValue()),
            T(5), atol=0.01
        )
    MOI.set(
        model,
        obj_attr,
        MOI.ScalarQuadraticFunction(
            MOI.ScalarQuadraticTerm.(
                T[2, 1//4, 1//4, 1//2, 2],
                [x[1], x[1], x[2], x[1], x[2]],
                [x[1], x[2], x[1], x[2], x[2]],
            ),  # quad
            MOI.ScalarAffineTerm{T}[],  # affine terms
            T(0),  # constant
        ),
    )
    MOI.optimize!(model)
    variable_primal = [(x[1], T(1)), (x[2], T(2))]
    for (index, solution_value) in variable_primal
        @test ≈(
            MOI.get(model, MOI.VariablePrimal(), index),
            solution_value, atol=0.01
        )
    end
    @test isapprox(
            MOI.get(model, MOI.ObjectiveValue()),
            T(7), atol=0.01
        )
end
 
@testset "Dual infeasibility clone" begin 
    T=Float64
    opt = PIPG.Optimizer(ϵ=1e-7)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    x = MOI.add_variables(model, 5)
    MOI.set(
        model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(T(1), x), T(0)),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
end


@testset "Linear feasibility clone" begin 
    atol = 1e-5
    rtol = 1e-5
    T = Float64
    opt = PIPG.Optimizer(ϵ=1e-7)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
            opt,
        ),
        Float64,
    )
    x = MOI.add_variable(model)
    y = MOI.add_variable(model)
    c1 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction{T}(
            MOI.ScalarAffineTerm{T}.(T[2, 3], [x, y]),
            T(0),
        ),
        MOI.GreaterThan(T(1)),
    )
    c2 = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction{T}(
            MOI.ScalarAffineTerm{T}.(T[1, -1], [x, y]),
            T(0),
        ),
        MOI.EqualTo(T(0)),
    )
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    @test MOI.get(model, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        MOI.optimize!(model)
        @test MOI.get(model, MOI.ResultCount()) > 0
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        xsol = MOI.get(model, MOI.VariablePrimal(), x)
        ysol = MOI.get(model, MOI.VariablePrimal(), y)
        c1sol = 2 * xsol + 3 * ysol
        @test c1sol >= 1 || isapprox(c1sol, T(1), atol = atol, rtol = rtol)
        @test xsol - ysol ≈ T(0) atol = atol rtol = rtol
        c1primval = MOI.get(model, MOI.ConstraintPrimal(), c1)
        @test c1primval >= 1 || isapprox(c1sol, T(1), atol = atol, rtol = rtol)
        @test MOI.get(model, MOI.ConstraintPrimal(), c2) ≈ T(0) atol = atol rtol =
            rtol
        @test MOI.get(model, MOI.DualStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.ConstraintDual(), c1) ≈ T(0) atol = atol rtol =
            rtol
        @test MOI.get(model, MOI.ConstraintDual(), c2) ≈ T(0) atol = atol rtol =
            rtol
end

    const OPTIMIZER = MOI.instantiate(
        MOI.OptimizerWithAttributes(PIPG.Optimizer, MOI.Silent() => true),
    )

    const BRIDGED = MOI.instantiate(
        MOI.OptimizerWithAttributes(PIPG.Optimizer, MOI.Silent() => true),
        with_bridge_type = Float64,
    )
    const CONFIG = MOI.Test.Config(
        atol=1e-5, 
        rtol=1e-5, 
        optimal_status=MOI.OPTIMAL, 
        exclude=Any[MOI.VariableName, MOI.delete,
            # not an integer solver
            MOI.ObjectiveBound, MOI.RelativeGap, 
            # not simplex
            MOI.ConstraintBasisStatus, MOI.SimplexIterations, 
            # not barrier, nor branch-and-cut
            MOI.BarrierIterations, MOI.NodeCount])

    MOI.Test.runtests(
        BRIDGED,
        CONFIG,
        exclude = [
            "test_attribute_NumberOfThreads",
            "test_linear_Indicator_",
            "test_model_UpperBoundAlreadySet",
            "test_objective_ObjectiveFunction_blank" # this is annoying for us
        ],
        # This argument is useful to prevent tests from failing on future
        # releases of MOI that add new tests. Don't let this number get too far
        # behind the current MOI release though! You should periodically check
        # for new tests in order to fix bugs and implement new features.
        exclude_tests_after = v"0.10.5",
    )