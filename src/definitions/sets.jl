abstract type Space{T, D} end	
struct InfBound{T, D} <: Space{T, D}
    δ::Vector{T}
    InfBound{T,D}(x) where {T,D} = begin 
    	@assert length(x) == D
    	return new{T,D}(x)
    end
end
struct Equality{T, D} <: Space{T, D}
	v::Vector{T}
    Equality{T,D}(x) where {T,D} = begin 
    	@assert length(x) == D
    	return new{T,D}(x)
    end
end
struct Ball{T, D} <: Space{T, D}
	r::T
end
struct PTSpace{T, Cs<:Tuple{Vararg{C where C<:Space{T}}}, D} <: Space{T, D}
	cones::Cs
	@generated PTSpace{T}(cs::Cs) where {T, Cs <: Tuple{Vararg{C where C<:Space{T}}}} = Expr(:new, PTSpace{T, Cs, sum(dim.(Cs.parameters); init=0)}, :(cs))
end

struct Product{T, D1, D2, D3, S1 <: Space{T, D1}, S2 <: Space{T, D2}} <: Space{T, D3}
	s1::S1
	s2::S2
	Product(s1::S1, s2::S2) where {T, D1, D2, S1 <: Space{T, D1}, S2 <: Space{T, D2}} = new{T, D1, D2, D1+D2, S1, S2}(s1, s2)
end
abstract type Cone{T, D} <: Space{T, D} end

struct Polar{T, D, C<:Cone{T, D}} <: Cone{T, D}
	inner::C 
end
struct Reals{T, D} <: Cone{T, D} end # R^D
struct Zeros{T, D} <: Cone{T, D} end # 0^D

struct SignCone{T, D} <: Cone{T, D}
	sign::Bool
end # { x | forall i, x_i * (-1)^(!sign) >= 0}
mutable struct HalfspaceCone{T, D} <: Cone{T, D} 
	d::Vector{T}
	o::T
	d_norm2::T
	HalfspaceCone{T, D}(d::Vector{T}, o::T) where {T, D} = new{T, D}(d, o, sum(d .^ 2))
end # {x | forall i, <x, d> <= o}

mutable struct SOCone{T, D} <: Cone{T, D}
	angle::T
end # { [t,x] | |x| <= angle * t }

struct PTCone{T, Cs<:Tuple{Vararg{C where C<:Cone{T}}}, D} <: Cone{T, D}
	cones::Cs
	@generated PTCone{T}(cs::Cs) where {T, Cs <: Tuple{Vararg{C where C<:Cone{T}}}} = Expr(:new, PTCone{T, Cs, sum(dim.(Cs.parameters); init=0)}, :(cs))
end

function Base.show(io::IO, p::PTCone{T}) where T
	print(io, "PTCone{$T}((") 
	for (ind, cone) in enumerate(p.cones)
		show(io, cone)
		if ind != length(p.cones)
			print(io, ", ")
		end
	end
	print(io, "))")
end

struct PermutedCone{T, D, P<:Cone{T, D}} <: Cone{T, D}
	cone::P
	permutation::Vector{Int64} # of length D
	inverse_permutation::Vector{Int64}
	function PermutedCone(cone::P, permutation::Vector{Int64}) where {T, D, P<:Cone{T, D}}
		@assert length(permutation) == D
		inverse_permutation = collect(1:length(permutation))
		inverse_permutation[permutation] .= 1:length(permutation)
		return new{T, D, P}(cone, permutation, inverse_permutation)
	end
end

struct PermutedSpace{T, D, P<:Space{T, D}} <: Space{T, D}
	cone::P
	permutation::Vector{Int64} # of length D
	inverse_permutation::Vector{Int64}
	function PermutedSpace(cone::P, permutation::Vector{Int64}) where {T, D, P<:Space{T, D}}
		@assert length(permutation) == D
		inverse_permutation = collect(1:length(permutation))
		inverse_permutation[permutation] .= 1:length(permutation)
		return new{T, D, P}(cone, permutation, inverse_permutation)
	end
end

copy(s::InfBound{T,D}) where {T,D} = InfBound{T,D}(copy(s.δ))
copy(s::Equality{T,D}) where {T,D} = Equality{T,D}(copy(s.v))
copy(s::PTSpace{T,Cs,D}) where {T,Cs,D} = PTSpace{T}(copy.(s.cones))
copy(s::PTCone{T,Cs,D}) where {T,Cs,D} = PTCone{T}(copy.(s.cones))
copy(c::Union{Reals, Zeros, SignCone}) = c
copy(s::HalfspaceCone{T,D}) where {T,D} = HalfspaceCone{T,D}(s.d, s.o)
copy(s::SOCone{T,D}) where {T,D} = SOCone{T,D}(s.angle)
copy(b::Ball{T,D}) where {T,D} = b
copy(pc::PermutedCone{T, D, P}) where {T, D, P} = PermutedCone{T,D,P}(copy(pc.cone), pc.permutation)
copy(pc::PermutedSpace{T, D, P}) where {T, D, P} = PermutedSpace{T,D,P}(copy(pc.cone), pc.permutation)

dim(::Space{T,D}) where {T,D} = D
dim(::Type{T}) where {D, Tc, T<:Space{Tc,D}} = D
cone_dim(::Cone{T,D}) where {T,D} = D
cone_dim(::Type{C}) where {T, D, C<:Cone{T,D}} = D

# polars
polar(::Reals{T,D}) where {T,D} = Zeros{T,D}()
polar(::Zeros{T,D}) where {T,D} = Reals{T,D}()
polar(s::SignCone{T,D}) where {T,D} = Polar{T, D, SignCone{T,D}}(s)
polar(h::HalfspaceCone{T,D}) where {T,D} = Polar{T, D, HalfspaceCone{T,D}}(h)
polar(s::SOCone{T,D}) where {T,D} = Polar{T, D, SOCone{T,D}}(s)
polar(pc::PermutedCone{T, D, P}) where {T, D, P} = PermutedCone(polar(pc.cone), pc.permutation)

@generated function polar(c::PTCone{T, Cs, D}) where {T, Cs, D} 
	tup = Expr(:tuple, (:(polar(c.cones[$i])) for i in 1:length(Cs.parameters))...)
	return :(PTCone{T}($tup))
end

variables(s::Space) = variables(s, 1, 1)
variables(r::Space{T, D}, cone_index, offs::Int) where {T, D} = (cone_index, ) .=> (offs:offs+D-1)
variables(p::PermutedCone) = throw("Cannot introspect on the variables of a permuted cone.")
function variables(r::Union{PTCone{T, Cs}, PTSpace{T, Cs}}, cone_index, offs::Int) where {T, Cs} 
    vars = []
    for cone in r.cones 
        append!(vars, variables(cone, cone_index, offs))
        cone_index += 1
        offs += dim(cone)
    end
    return vars
end
