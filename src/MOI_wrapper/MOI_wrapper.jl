@enum Membership K=1 D=2
MOI.Utilities.@product_of_sets(
	Cones,
	MOI.Zeros,
	MOI.Nonnegatives,
	MOI.Nonpositives,
	MOI.SecondOrderCone,
	MOI.Interval)
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


mutable struct SpecializedOptimizer{T,N,M,R<:Problem{T,N,M}, S<:State{T,N,M}} <: MOI.AbstractOptimizer
	name::String
	
	problem::R
	state::S
	solve_result::Union{Nothing, OptimizationResult}
	
	sets::PIPG.Cones{T}
	silent::Bool
	elapsed_time::Float64
	max_sense::Bool
end

mutable struct Optimizer <: MOI.AbstractOptimizer
	name::String
	cones::Union{Nothing, Cones{Float64}}
	sets::Union{Nothing, PIPG.Cones{Float64}}
	silent::Bool
	elapsed_time::Float64
	max_sense::Bool
	membership::Dict{MOI.ConstraintIndex, Membership}
	nstages::Union{Nothing, Int}
	specialized::Union{Nothing, SpecializedOptimizer}
	preconditioner::Preconditioner

	niters::Int
	ϵ::Float64
	γ::Float64
	function Optimizer(; niters=1000000, ϵ=1e-6, γ=0.9, preconditioner=Ruiz())
		return new("", nothing, nothing, false, 0.0, false, Dict{MOI.ConstraintIndex, Membership}(), nothing, nothing, preconditioner, niters, ϵ, γ)
	end
end

# basics
MOI.is_empty(opt::Optimizer) = opt.cones === nothing
function MathOptInterface.empty!(opt::Optimizer) 
	opt.cones = nothing	
	opt.specialized = nothing
	opt.sets = nothing
end

# silent
MOI.supports(::Union{Optimizer, SpecializedOptimizer}, ::MOI.Silent) = true
MOI.set(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.Silent, value::Bool) = (o.silent = value)
MOI.get(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.Silent) = o.silent

MOI.supports(::Union{Optimizer, SpecializedOptimizer}, ::MOI.Name) = true
MOI.set(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.Name, value::String) = (o.name = value)
MOI.get(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.Name) = o.name

MOI.get(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.SolverName) = "PIPG"
MOI.get(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.SolverVersion) = "v0.1.0"
MOI.get(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.RawSolver) = o.problem
MOI.get(o::Union{Optimizer, SpecializedOptimizer}, ::MOI.SolveTimeSec) = o.elapsed_time

MOI.supports(::Union{Optimizer, SpecializedOptimizer}, ::MOI.ConstraintBasisStatus) = false
# supported sets
const SupportedSets = Union{
	MOI.Zeros,
	MOI.Nonnegatives,
	MOI.Nonpositives,
	MOI.SecondOrderCone,
	MOI.Interval
}
MOI.supports(::Union{Optimizer, SpecializedOptimizer}, ::Union{
		MOI.ObjectiveSense, 
		MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}, 
		MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}}) = true
MOI.supports_constraint(
	::Union{Optimizer, SpecializedOptimizer}, 
	::Type{MOI.VectorAffineFunction{Float64}}, 
	::Type{<: SupportedSets}) = true

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
convert_cone(n::MOI.Nonnegatives) = SignCone{Float64, n.dimension}(true)
convert_cone(n::MOI.Nonpositives) = SignCone{Float64, n.dimension}(false)
convert_cone(s::MOI.SecondOrderCone) = SOCone{Float64, s.dimension}(1.0)
function convert_cone(s::MOI.Interval) 
	span = s.lower - s.upper 
	InfBound{Float64,1}([span], [lower + span/2])
end

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

	cones_K = Cone[]
	moved = Pair{MOI.ConstraintIndex, Cone}[]
	if SetMembership() ∈ keys(src.conattr)	
		dest.membership = copy(src.conattr[SetMembership()])
		for set in MOI.Utilities.set_types(Ab.sets)
			cis = MOI.get(Ab, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{Float64}, set}())
			for ci in cis 
				cone = MOI.get(Ab, MOI.ConstraintSet(), ci)
				if ci in keys(dest.membership) && dest.membership[ci] == D
					push!(moved, ci => convert_cone(cone))
				else 
					push!(cones_K, convert_cone(cone))
				end
			end
		end
	else 
		for set in MOI.Utilities.set_types(Ab.sets)
			cis = MOI.get(Ab, MOI.ListOfConstraintIndices{MOI.VectorAffineFunction{Float64}, set}())
			for ci in cis 
				cone = MOI.get(Ab, MOI.ConstraintSet(), ci)
				push!(cones_K, convert_cone(cone))
			end
		end
	end
	cones_K = simplify_cones(cones_K)
	k = PTCone{Float64}((cones_K..., ))

	if length(moved) > 0
		cones_d = Cone[]
		variable_assignment = Array{Union{Nothing, Pair{Int, Float64}}}(nothing, A.n)
		used_dest_rows = BitSet()
		used_src_rows = BitSet()
		offset = 0
		for (ci, cone) ∈ moved
			func = MOI.get(Ab, MOI.ConstraintFunction(), ci)
			cone_scaling = zeros(Float64, dim(cone))
			cone_constant = func.constants
			total = 0
			for term ∈ func.terms # we know that func is a VectorAffineFunction
				if !isnothing(variable_assignment[term.scalar_term.variable.value])
					throw("repeated variable in sets lifted into D")
				end
				if term.output_index + offset ∈ used_dest_rows
					throw("multiple variables appearing in a single row in D")
				end
				#=
				if any(abs.(func.constants) .>= 1e-8) 
					throw("constant terms not allowed for D constraints $(func.constants)")
				end
				=#
				push!(used_dest_rows, term.output_index + offset)
				cone_scaling[term.output_index] = term.scalar_term.coefficient
				variable_assignment[term.scalar_term.variable.value] = (term.output_index + offset) => term.scalar_term.coefficient
				total += 1
			end
			cone = scale_cone(cone, cone_scaling, cone_constant)
			push!.((used_src_rows, ), MOI.Utilities.rows(Ab, ci))
			if total != dim(cone)
				throw("Insufficiently specified constraint lifted into D")
			end
			push!(cones_d, cone)
			offset += dim(cone)
		end
		remaining = A.n - offset # the number of variables that are not bounded by something explicitly in D
		if remaining > 0
			offset += 1 # prep offset for writing all of the remaining variable coefficients
			for var_num=1:A.n
				if isnothing(variable_assignment[var_num])
					variable_assignment[var_num] = offset => 1.0
					offset += 1
				end
			end
			push!(cones_d, Reals{Float64, remaining}())
		end
		scaling = Array{Float64}(undef, A.n)
		for (var_index, coefficient) in variable_assignment
			scaling[var_index] = coefficient
		end
		A_rows = [false for _ in 1:A.m]
		g = Float64[]
		for (row, val) in enumerate(Ab.constants)
			if !(row ∈ used_src_rows)
				push!(g, -val)
				A_rows[row] = true
			end
		end
		d = PTSpace{Float64}((cones_d..., ))
		H = convert(SparseMatrixCSC{Float64, Int64}, A)
		H = H[A_rows,:]
		S = scaling
	else
		d = Reals{Float64, A.n}()
		g = -1 .* Ab.constants
		H = convert(SparseMatrixCSC{Float64, Int64}, A)
		S = ones(A.n)
	end

	col_perm = 1:A.n
	row_perm = 1:A.m
	if !isnothing(dest.nstages)
		row_perm, col_perm = sort_matrix(H, dest.nstages)
		H = H[row_perm, col_perm]
		P = P[col_perm, col_perm]
		k = PermutedCone(k, row_perm)
		d = PermutedSpace(d, col_perm)
		q = q[col_perm]
		g = g[row_perm]
	end

    p = Problem(k, d, H, P, q, g, objective_constant)
	s = PIPG.State(p, PIPG.xPIPG(dest.ϵ, dest.γ; iters=dest.niters); preconditioner=dest.preconditioner)

	dest.specialized = SpecializedOptimizer(dest.name, p, s, nothing, Ab.sets, dest.silent, dest.elapsed_time, max_sense)
	dest.max_sense = max_sense
end

simplify_cones(cones) = foldl(simplify_cone, cones; init=[])
simplify_cone(acc, a) = if length(acc) > 0 simplify_cone(acc, last(acc), a) else [a] end
simplify_cone(acc, a::SignCone{T, D1}, b::SignCone{T, D2}) where {T, D1, D2} = [acc[1:end-1]; a.sign == b.sign ? [SignCone{T, D1+D2}(a.sign)] : [a, b]]
#simplify_cone(acc, a::POCone{T, D1}, b::POCone{T, D2}) where {T, D1, D2} = [acc[1:end-1]; POCone{T, D1+D2}()]
#simplify_cone(acc, a::NOCone{T, D1}, b::NOCone{T, D2}) where {T, D1, D2} = [acc[1:end-1]; NOCone{T, D1+D2}()]
#simplify_cone(acc, a::Reals{T, D1}, b::Reals{T, D2}) where {T, D1, D2} = [acc[1:end-1]; Reals{T, D1+D2}()]
simplify_cone(acc, a::Zeros{T, D1}, b::Zeros{T, D2}) where {T, D1, D2} = [acc[1:end-1]; Zeros{T, D1+D2}()]
simplify_cone(acc, a, b) = [acc; b]

scale_cone(cone::Reals{T, D}, scale, constant) where {T, D} = cone
scale_cone(cone::Zeros{T, D}, scale, constant) where {T, D} = if all(constant .≈ zero(T)) cone else Equality{T, D}(constant) end
scale_cone(cone::SignCone{T, D}, scale, constant) where {T, D} = 
	if all(scale .≈ -1.0) SignCone{T, D}(!cone.sign) 
	elseif all(scale .≈ 1.0) cone
	else let original_sign = (cone.sign ? one(T) : -one(T))
			HalfspaceCone{T, D}(original_sign * scale, original_sign * dot(scale, constant))
		end
	end
scale_cone(cone::HalfspaceCone{T, D}, scale, constant) where {T, D} =
	let new_scale = cone.d .* scale
		if cone.o .≈ zero(T) && constant .≈ zero(T)
			if all(new_scale .≈ one(T)) SignCone{T, D}(true)
			elseif all(new_scale .≈ -one(T)) SignCone{T, D}(false)
			end
		end
		HalfspaceCone{T, D}(new_scale, dot(new_scale, cone.d .* cone.o + constant))
	end
scale_cone(cone::SOCone{T, D}, scale, constant) where {T, D} = 
	if any((!≈).(constant, zero(T))) error("Constant terms not allowed in a lifted second order cone!")
	elseif length(scale) > 0 && any((!≈).(scale[1], scale)) error("All scalar coefficients in a second order cone must be (up to epsilon) the same!")
		SOCone{T, D}(cone.angle/scale[1])
	end
	
# modification routines
check_constructed(d::Optimizer) = if isnothing(d.specialized) error("Must optimize before modification!") end
function MOI.modify(
	d::Optimizer, 
	ci::MOI.ConstraintIndex{T} where T<:Union{MOI.ScalarAffineFunction, MOI.VectorAffineFunction},
	change::Union{MOI.ScalarConstantChange, MOI.VectorConstantChange, MOI.MultirowChange})
	check_constructed(d)
	MOI.modify(d.specialized, ci, change)
end
function MOI.modify(
	d::SpecializedOptimizer, 
	ci::MOI.ConstraintIndex{T} where T<:Union{MOI.ScalarAffineFunction, MOI.VectorAffineFunction},
	change::Union{MOI.ScalarConstantChange, MOI.VectorConstantChange})
	rows = MOI.Utilities.rows(d.sets, ci)
	d.problem.g[rows] .= -1.0 .* change.new_constant
end

function MOI.modify(
	d::SpecializedOptimizer, 
	ci::MOI.ConstraintIndex{T} where T<:MOI.VectorAffineFunction,
	change::MOI.MultirowChange)
	new_coeiffs = change.new_coefficients
	rows = MOI.Utilities.rows(d.sets, ci)[getindex.(new_coeiffs, 1)]
	d.problem.H[change.variable.value, rows] .= getindex.(new_coeiffs, 2)
end

# optimizer
function MOI.copy_to(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache{T}}) where {T}
	index_map = MOI.Utilities.identity_index_map(src)
	build_problem(dest, src)
	return index_map
end

function compose_index_maps(a::MOI.IndexMap, b::MOI.IndexMap) # creates an index map equivalent to composed[x] = a[b[x]]
	output = MOI.IndexMap()
	for bkey in keys(b)
		output[bkey] = a[b[bkey]]
	end
	return output
end
function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    cache = MOI.Utilities.UniversalFallback(OptimizerCache{Int64}())
    index_map1 = MOI.copy_to(cache, src)
	index_map2 = MOI.copy_to(dest, cache)
    return compose_index_maps(index_map2, index_map1)
end

function MOI.optimize!(dest::Optimizer)
	if isnothing(dest.specialized)
		error("Optimizer not initialized!")
	end
	MOI.optimize!(dest.specialized)
end
function MOI.optimize!(dest::SpecializedOptimizer)
	n = length(dest.problem.q)
	m = length(dest.problem.g)
    res = @timed solve(dest.problem, dest.state)
	result, niters = res[1]
    println("niters=$niters")
	dest.solve_result = result
    dest.elapsed_time = res[2]
end
const Forwarded = Union{MOI.ResultCount,MOI.ObjectiveValue, MOI.DualObjectiveValue}
MOI.get(o::Optimizer, s::Forwarded) = MOI.get(o.specialized, s)
MOI.get(o::Optimizer, s::MOI.TerminationStatus) = isnothing(o.specialized) ? MOI.OPTIMIZE_NOT_CALLED : MOI.get(o.specialized, s)
MOI.get(o::Optimizer, s::Union{MOI.PrimalStatus, MOI.DualStatus}) = isnothing(o.specialized) ? MOI.NO_SOLUTION : MOI.get(o.specialized, s)
function MOI.get(o::SpecializedOptimizer, ::MOI.TerminationStatus)
	s = o.solve_result
	if isnothing(s)
		return MOI.OPTIMIZE_NOT_CALLED
	end
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

function MOI.get(optimizer::Union{Optimizer, SpecializedOptimizer}, ::MOI.RawStatusString)
    return "" 
end

function MOI.get(o::SpecializedOptimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(o, MOI.ResultCount())
        return MOI.NO_SOLUTION
	end
	r = o.solve_result
	if isnothing(r)
		return MOI.OPTIMIZE_NOT_CALLED
	end
    if r == OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif r == PRIMAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif r == DUAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    end
    return MOI.NO_SOLUTION
end
function MOI.get(o::SpecializedOptimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(o, MOI.ResultCount())
        return MOI.NO_SOLUTION
	end
	r = o.solve_result
	if isnothing(r)
		return MOI.OPTIMIZE_NOT_CALLED
	end
    if r == OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif r == PRIMAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif r == DUAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    end
    return MOI.NO_SOLUTION
end

MOI.get(::SpecializedOptimizer, ::MOI.ResultCount) = 1

function MOI.get(o::SpecializedOptimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(o, attr)
	raw_value = objective_value(o.problem, o.state)
    return (o.max_sense ? -1.0 : 1.0) * raw_value + o.problem.c
end

function MOI.get(o::SpecializedOptimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(o, attr)
    return (o.max_sense ? -1.0 : 1.0) * dual_objective_value(o.problem, o.state) + o.problem.c
end

function MOI.get(o::Optimizer, v::Union{MOI.VariablePrimal, MOI.ConstraintDual}, i::Union{MOI.VariableIndex, MOI.ConstraintIndex})
	check_constructed(o)
	return MOI.get(o.specialized, v, i)
end
function MOI.get(
    optimizer::SpecializedOptimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return primal(optimizer.state)[vi.value]
end

function MOI.get(
	optimizer::SpecializedOptimizer,
	attr::MOI.ConstraintDual,
	ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    rows = MOI.Utilities.rows(optimizer.sets, ci)
    return .- dual(optimizer.state)[rows]
end


struct SetMembership <: MOI.AbstractConstraintAttribute end
function MOI.set(model::Optimizer, ::SetMembership, c::MOI.ConstraintIndex, m::Membership)
	println("set membership! $c $m")
	model.membership[c] = m
end
MOI.supports(::Optimizer, ::SetMembership, ::MOI.ConstraintIndex{MOI.VectorAffineFunction,T}) where T<: SupportedSets = true
function MOI.set(model::MOI.ModelLike, attr::SetMembership, bridge::T, m::Membership) where T<:MOI.Bridges.AbstractBridge
	for (F,S) in MOI.Bridges.added_constraint_types(T)
		for ci in MOI.get(bridge, MOI.ListOfConstraintIndices{F,S}())
			MOI.set(model, attr, ci, m)
		end
	end
end
function MOI.get(model::Optimizer, ::SetMembership, c::MOI.ConstraintIndex)
	return model.membership[c]
end

struct NStages <: MOI.AbstractModelAttribute end 
MOI.set(model::Optimizer, ::NStages, n::Integer) = model.stages = n
MOI.get(model::Optimizer, ::NStages) = model.stages