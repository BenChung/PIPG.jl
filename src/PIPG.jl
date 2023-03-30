module PIPG
using StaticArrays
using SparseArrays
using LinearAlgebra
using LazyArrays
using KaHyPar
using SparseArrays
using DataStructures
using LoopVectorization

include("definitions/sets.jl")
include("definitions/problem.jl")
include("definitions/projections.jl")

include("problem_parsing/coloring.jl")
include("problem_parsing/partitioning.jl")
include("solver.jl")
include("preconditioning.jl")
include("MOI_wrapper/MOI_wrapper.jl")

end
