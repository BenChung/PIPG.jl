module PIPG
using StaticArrays
using SparseArrays
using LinearAlgebra
using LazyArrays
using DataStructures
using FastBroadcast
import Base: copy
import MathOptInterface
const MOI = MathOptInterface

include("definitions/sets.jl")
include("definitions/problem.jl")
include("definitions/projections.jl")

include("problem_parsing/coloring.jl")
include("preconditioning/preconditioning.jl")
include("preconditioning/chained.jl")
include("preconditioning/ruiz.jl")
include("preconditioning/column_normalization.jl")
include("preconditioning/cholesky.jl")
include("preconditioning/qr.jl")
include("diagnostics.jl")
include("solvers/solver.jl")
include("solvers/xpipg.jl")
include("MOI_wrapper/MOI_wrapper.jl")

end
