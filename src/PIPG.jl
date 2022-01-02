module PIPG
using StaticArrays
using SparseArrays
using LinearAlgebra

include("solver.jl")
include("preconditioning.jl")
include("MOI_wrapper/MOI_wrapper.jl")

# Write your package code here.

end
