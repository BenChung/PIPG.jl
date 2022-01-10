import MathOptInterface
const MOI = MathOptInterface

MOI.Utilities.@product_of_sets(
	Cones,
	MOI.Zeros,
	MOI.Nonnegatives,
	MOI.Nonpositives,
	MOI.SecondOrderCone)
const OptimizerCache{T} = MOI.Utilities.GenericModel{
	Float64,
	MOI.Utilities.ObjectiveContainer{Float64},
	MOI.Utilities.VariablesContainer{Float64},
	MOI.Utilities.MatrixOfConstraints{
		Float64, 
		MOI.Utilities.MutableSparseMatrixCSC{Float64, T, MOI.Utilities.OneBasedIndexing},
		Vector{Float64},
		Cones{Float64}
	}
}

mutable struct Optimizer <: MOI.AbstractOptimizer
	name::String
	cones::Union{Nothing, Cones{Float64}}
	problem::Union{Nothing, Problem}
	objective_constant::Float64
	state::Union{Nothing, State}
	sets::Union{Nothing, PIPG.Cones{Float64}}
	silent::Bool
	elapsed_time::Float64
	max_sense::Bool

	scaling

	niters::Int
	ϵ::Float64
	γ::Float64
	function Optimizer(; niters=1000000, ϵ=1e-9, γ=0.9)
		return new("", nothing, nothing, 0.0, nothing, nothing, false, 0.0, false, nothing, niters, ϵ, γ)
	end
end

# basics
MOI.is_empty(opt::Optimizer) = opt.cones === nothing
function MathOptInterface.empty!(opt::Optimizer) 
	opt.cones = nothing	
	opt.problem = nothing
	opt.state = nothing
	opt.sets = nothing
end

# silent
MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(o::Optimizer, ::MOI.Silent, value::Bool) = (o.silent = value)
MOI.get(o::Optimizer, ::MOI.Silent) = o.silent

MOI.supports(::Optimizer, ::MOI.Name) = true
MOI.set(o::Optimizer, ::MOI.Name, value::String) = (o.name = value)
MOI.get(o::Optimizer, ::MOI.Name) = o.name

MOI.get(o::Optimizer, ::MOI.SolverName) = "PIPG"
MOI.get(o::Optimizer, ::MOI.SolverVersion) = "v0.1.0"
MOI.get(o::Optimizer, ::MOI.RawSolver) = o.problem
MOI.get(o::Optimizer, ::MOI.SolveTimeSec) = o.elapsed_time

MOI.supports(::Optimizer, ::MOI.ConstraintBasisStatus) = false
# supported sets
MOI.supports(::Optimizer, ::Union{
		MOI.ObjectiveSense, 
		MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}, 
		MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}}) = true
MOI.supports_constraint(
	::Optimizer, 
	::Type{MOI.VectorAffineFunction{Float64}}, 
	::Type{<: Union{
		MOI.Zeros,
		MOI.Nonnegatives,
		MOI.Nonpositives,
		MOI.SecondOrderCone
	}}) = true


function MOI.modify(
    model::MOI.ModelLike,
    bridge::MOI.Bridges.Constraint.VectorizeBridge,
    change::MOI.ScalarConstantChange)
    MOI.modify(
        model,
        bridge.vector_constraint,
        MOI.VectorConstantChange([-change.new_constant]))
    return
end

convert_cone(z::MOI.Zeros) = Zeros{Float64, z.dimension}()
convert_cone(n::MOI.Nonnegatives) = POCone{Float64, n.dimension}()
convert_cone(n::MOI.Nonpositives) = NOCone{Float64, n.dimension}()
convert_cone(s::MOI.SecondOrderCone) = SOCone{Float64, s.dimension}()

function build_problem(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache{T}}) where {T}
	Ab = src.model.constraints
	A = Ab.coefficients
	dest.cones = Ab.sets

	for (F, S) in keys(src.constraints)
		throw(MOI.UnsupportedConstraint{F,S}())
    end

    model_attributes = MOI.get(src, MOI.ListOfModelAttributesSet())
    quad_objective_function = MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}()
    aff_objective_function = MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
    for attr in model_attributes
        if attr != MOI.ObjectiveSense() && attr != quad_objective_function && attr != aff_objective_function
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    max_sense = false
    if MOI.ObjectiveSense() in model_attributes
        max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    end

    objective_constant = 0.0
    q = zeros(A.n)
    if quad_objective_function in model_attributes
    	obj = MOI.get(src, quad_objective_function)
        objective_constant += MOI.constant(obj)

        I = Vector{Int}()
        J = Vector{Int}()
        V = Vector{Float64}()

        for term in obj.quadratic_terms
        	push!(I, Int(term.variable_1.value))
        	push!(J, Int(term.variable_2.value))
        	push!(V, (max_sense ? -1 : 1) * term.coefficient)
        	if (term.variable_1 != term.variable_2) # off-diagonal
	        	push!(I, Int(term.variable_2.value))
	        	push!(J, Int(term.variable_1.value))
	        	push!(V, (max_sense ? -1 : 1) * term.coefficient)
        	end
        end
        for term in obj.affine_terms
            q[term.variable.value] += (max_sense ? -1 : 1) * term.coefficient
        end
        P = sparse(I, J, V, A.n, A.n)
    else 
    	P = spzeros(A.n, A.n)
    end

    if aff_objective_function in model_attributes
    	obj = MOI.get(src, aff_objective_function)
        objective_constant += MOI.constant(obj)
    	for term in obj.terms
            q[term.variable.value] += (max_sense ? -1 : 1) * term.coefficient
    	end
    end

    function _map_sets(f, ::Type{T}, sets, ::Type{S}) where {T,S}
	    F = MOI.VectorAffineFunction{Float64}
	    cis = MOI.get(sets, MOI.ListOfConstraintIndices{F,S}())
	    return T[f(MOI.get(sets, MOI.ConstraintSet(), ci)) for ci in cis]
	end
	cones = Cone[]
	for set in MOI.Utilities.set_types(Ab.sets)
		append!(cones, _map_sets(convert_cone, Cone, Ab, set))
	end
	k = PTCone{Float64}((cones..., ))
    d = Reals{Float64, A.n}()
    H = convert(SparseMatrixCSC{Float64, Int64}, A)
    g = -1 .* Ab.constants

    p = Problem(k, d, H, P, MVector{A.n, Float64}(q), MVector{A.m, Float64}(g), objective_constant)
    if dest.scaling != nothing
    	s = State(p; scaling=dest.scaling(p))
	else
		s = State(p) 
	end
	dest.max_sense = max_sense
    dest.problem = p
    dest.state = s
    dest.sets = Ab.sets
end

# modification routines
check_constructed(d::Optimizer) = if isnothing(d.problem) || isnothing(d.state) error("Must optimize before modification!") end
function MOI.modify(
	d::Optimizer, 
	ci::MOI.ConstraintIndex{T} where T<:Union{MOI.ScalarAffineFunction, MOI.VectorAffineFunction},
	change::Union{MOI.ScalarConstantChange, MOI.VectorConstantChange})
	check_constructed(d)
	rows = MOI.Utilities.rows(d.sets, ci)
	d.problem.g[rows] .= -1.0 .* d.state.row_scale[rows] .* change.new_constant
end
function MOI.modify(
	d::Optimizer, 
	ci::MOI.ConstraintIndex{T} where T<:MOI.VectorAffineFunction,
	change::MOI.MultirowChange)
	new_coeiffs = change.new_coefficients
	rows = MOI.Utilities.rows(d.sets, ci)[getindex.(new_coeiffs, 1)]
	scale = d.state.row_scale[rows] .* d.state.col_scale[change.variable.value]
	d.problem.H[change.variable.value, rows] .= scale .* getindex.(new_coeiffs, 2)
end

# optimizer
function MOI.optimize!(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache{T}}) where {T}
	index_map = MOI.Utilities.identity_index_map(src)

	if isnothing(dest.problem) || isnothing(dest.state)
		build_problem(dest, src)
	end

	Ab = src.model.constraints
	A = Ab.coefficients
	PIPG.scale(dest.problem, dest.state)
	α = compute_α(dest.problem, dest.γ)
    res = @timed pipg(dest.problem, dest.state, dest.niters, α, dest.ϵ, SVector{A.n, Float64}(zeros(A.n)), SVector{A.m, Float64}(zeros(A.m)))
    dest.elapsed_time = res[2]
end
function MOI.optimize!(dest::Optimizer)
	if isnothing(dest.problem) || isnothing(dest.state)
		error("Optimizer not initialized!")
	end

	n = length(dest.problem.q)
	m = length(dest.problem.g)
	PIPG.scale(dest.problem, dest.state)
	α = compute_α(dest.problem, dest.γ)
    res = @timed pipg(dest.problem, dest.state, dest.niters, α, dest.ϵ, SVector{n, Float64}(zeros(n)), SVector{m, Float64}(zeros(m)))
    println("niters=$(res[1])")
    dest.elapsed_time = res[2]
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = MOI.Utilities.UniversalFallback(OptimizerCache{Int64}())
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, true
end

function MOI.get(o::Optimizer, ::MOI.TerminationStatus)
	if isnothing(o.state) return MOI.OPTIMIZE_NOT_CALLED end
	s = o.state.solver_state
	if s == INDEFINITE
		return MOI.OPTIMIZE_NOT_CALLED
	elseif s == OPTIMAL
		return MOI.OPTIMAL
	elseif s == PRIMAL_INFEASIBLE
		return MOI.INFEASIBLE
	elseif s == DUAL_INFEASIBLE
		return MOI.DUAL_INFEASIBLE
	elseif s == TIMEOUT
		return MOI.INTERRUPTED
	end
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return "" 
end

function MOI.get(o::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(o, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif !isnothing(o.state) && o.state.solver_state == OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif !isnothing(o.state) && o.state.solver_state == PRIMAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif !isnothing(o.state) && o.state.solver_state == DUAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    end
    return MOI.NO_SOLUTION
end
function MOI.get(o::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(o, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif !isnothing(o.state) && o.state.solver_state == OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif !isnothing(o.state) && o.state.solver_state == PRIMAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif !isnothing(o.state) && o.state.solver_state == DUAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::Optimizer, ::MOI.ResultCount) = 1

function MOI.get(o::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(o, attr)
	raw_value = objective_value(o.problem, o.state)
    return (o.max_sense ? -1.0 : 1.0) * raw_value + o.problem.c
end

function MOI.get(o::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(o, attr)
    return (o.max_sense ? -1.0 : 1.0) * dual_objective_value(o.problem, o.state) + o.problem.c
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.state.primal[vi.value] * optimizer.state.col_scale[vi.value]
end

function MOI.get(
	optimizer::Optimizer,
	attr::MOI.ConstraintDual,
	ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    rows = MOI.Utilities.rows(optimizer.cones, ci)
    return .- optimizer.state.dual[rows] .* optimizer.state.row_scale[rows]
end
