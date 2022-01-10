using PIPG
import MathOptInterface
using LinearAlgebra
using StaticArrays
using SparseArrays
using Profile
const MOI = MathOptInterface
using Test

@testset "Linear algebra routines" begin 
    for i=1:100
        res = MVector{10,Float64}(zeros(10))
        mat = sprand(10,10,0.5)
        inp = SVector{10, Float64}(rand(10))
        PIPG.spmul!(res, mat, inp, 1.0, 0.0)
        @test all(abs.(res .- mat*inp) .< 1e-10)
        PIPG.spmul!(res, mat, inp, 1.0, -1.0)
        @test all(abs.(res) .< 1e-10)
        PIPG.spmul!(res, mat, inp, -1.0, 0.0)
        @test all(abs.(res .+ mat*inp) .< 1e-10)
    end
end


@testset "Basic optimizer examples" begin
	γ = 0.9
	# QP example
    function make_qp()
    	P_ex = spdiagm([1/MathConstants.e, 2.0])
    	q = MVector(1/MathConstants.e,2.0)
    	H_ex = sparse([-1.0 -1.0])
    	g = MVector(0.0)
    	return PIPG.Problem(PIPG.NOCone{Float64, 1}(), PIPG.Reals{Float64, 2}(), H_ex, P_ex, q, g, 0.0)
    end
	
    # no scaling
    prob = make_qp()
    state = PIPG.State(prob; scaling=())
	PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(1.0, 1.0), SVector(0.0))
    result = state.primal
	@test norm(result .- [0.689, -0.689]) < 0.001

	# LP example
    make_prob() = PIPG.Problem(PIPG.PTCone{Float64}((PIPG.POCone{Float64, 1}(), PIPG.POCone{Float64, 1}(), PIPG.POCone{Float64, 1}(), 
        PIPG.POCone{Float64, 2}())), PIPG.Reals{Float64, 2}(), 
        sparse([-100.0 -200.0; -10.0 -30.0; -1.0 -1.0; 1.0 0.0; 0.0 1.0]), 
        spzeros(2,2), 
        -1 .* MVector(50.0, 120.0), 
        MVector(-10000.0, -1200.0, -110, -0.0, -0.0), 0.0)
	prob = make_prob()
	state = PIPG.State(prob; scaling=())
	PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0, 0.0))
	@test norm(state.primal .- [60, 20]) < 0.01

    # geometric scaling
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.GeoMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0, 0.0))
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.GeoMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(0.0, 0.0), SVector(0.0))
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001


    # arithmetic scaling
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.ArithMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0, 0.0))
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.ArithMean(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, SVector(0.0, 0.0), SVector(0.0))
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001


    # equilibration scaling
    prob = make_prob()
    state = PIPG.State(prob; scaling=(PIPG.Equilibration(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0, 0.0))
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob; scaling=(PIPG.Equilibration(prob), ))
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, SVector(0.0, 0.0), SVector(0.0))
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001

    # default scaling (geomean + equlibration)
    prob = make_prob()
    state = PIPG.State(prob)
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.0001, SVector(0.0, 0.0), SVector(0.0, 0.0, 0.0, 0.0, 0.0))
    @test norm(state.primal .* state.col_scale .- [60, 20]) < 0.01

    prob = make_qp()
    state = PIPG.State(prob)
    PIPG.scale(prob, state)
    niters = PIPG.pipg(prob, state, 2000000, PIPG.compute_α(prob, γ), 0.00001, SVector(0.0, 0.0), SVector(0.0))
    @test norm(state.primal .* state.col_scale .- [0.689, -0.689]) < 0.001
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
    scale_a_bunch(prob,state,niters) = for i=1:niters PIPG.scale(prob, state) end
    scale_a_bunch(prob, state, 1)
    Profile.clear()
    @profile scale_a_bunch(prob, state, 1000000)
    @time scale_a_bunch(prob, state, 1000000)
    Profile.print(C=true)
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
    opt.scaling = (p->(PIPG.GeoMean(p), ))
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
    opt.scaling = p->(PIPG.ArithMean(p),)
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
    opt.niters *= 10
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
    @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1
end

@testset "Geometric mean cone clone2" begin 
    opt = PIPG.Optimizer()
    opt.scaling = x->(PIPG.Equilibration(x), )
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
    opt.scaling = x->(PIPG.Equilibration(x), )
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
    opt.scaling = x->(PIPG.GeoMean(x), )
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
    @test isapprox(MOI.get(model, MOI.VariablePrimal(), t), T(5))
    MOI.delete(model, c_soc)
    MOI.optimize!(model)
    println(opt.problem)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
end

@testset "Quadratic test copies" begin 
    opt = PIPG.Optimizer()
    opt.scaling = x->()
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
    println(opt.problem.P)
    variable_primal = [(x[1], T(1)), (x[2], T(2))]
    for (index, solution_value) in variable_primal
        @test isapprox(
            MOI.get(model, MOI.VariablePrimal(), index),
            solution_value
        )
    end
    @test isapprox(
            MOI.get(model, MOI.ObjectiveValue()),
            T(5),
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
    println(opt.problem.P)
    variable_primal = [(x[1], T(1)), (x[2], T(2))]
    for (index, solution_value) in variable_primal
        @test isapprox(
            MOI.get(model, MOI.VariablePrimal(), index),
            solution_value
        )
    end
    @test isapprox(
            MOI.get(model, MOI.ObjectiveValue()),
            T(7),
        )
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
            "test_model_UpperBoundAlreadySet"
        ],
        # This argument is useful to prevent tests from failing on future
        # releases of MOI that add new tests. Don't let this number get too far
        # behind the current MOI release though! You should periodically check
        # for new tests in order to fix bugs and implement new features.
        exclude_tests_after = v"0.10.5",
    )