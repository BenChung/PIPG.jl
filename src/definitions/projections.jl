project!(t, i, ::Reals{T, D}, x::AbstractArray{T}) where {D, T} = @inbounds (for j in i:i+D-1 @inbounds t[j] = x[j] end)
project!(t, i, ::Zeros{T, D}, x::AbstractArray{T}) where {D, T} = @inbounds (for j in i:i+D-1 @inbounds t[j] = zero(T) end)
function project!(t, i, n::InfNorm{T, D}, x::AbstractArray{T}) where {D, T}
	(for j in i:i+D-1 t[j] = clamp(x[j], -n.δ[j-i+1], n.δ[j-i+1]) end)
end
project!(t, i, c::SignCone{T, D}, x::AbstractArray{T}) where {D, T} = @inbounds let sgn = c.sign ? one(T) : -one(T); (for j in i:i+D-1 t[j] = max(sgn*x[j], zero(T))*sgn end) end
project!(t, i, c::Polar{T, D, SignCone{T, D}}, x::AbstractArray{T}) where {D, T} = @inbounds let sgn = c.inner.sign ? -one(T) : one(T); (for j in i:i+D-1 t[j] = max(sgn*x[j], zero(T))*sgn end) end
project!(t, i, c::HalfspaceCone{T, D}, x::AbstractArray{T}) where {D, T} =
	let scalar = (dot(c.d, x[i:i+D-1]) - c.o)/c.d_norm2;
		if scalar <= 0 
			for j in i:i+D-1 @inbounds t[j] = x[j] end
		else 
			for j in i:i+D-1 t[j] = x[j] - scalar * c.d[j-i+1] end 
		end
	end
project!(t, i, c::Polar{T, D, HalfspaceCone{T, D}}, x::AbstractArray{T}) where {D, T} =
let scalar = (dot(c.inner.d, x[i:i+D-1]) - c.inner.o)/c.inner.d_norm2;
	if scalar > 0 
		for j in i:i+D-1 @inbounds t[j] = x[j] end
	else 
		for j in i:i+D-1 t[j] = x[j] - scalar * c.inner.d[j-i+1] end 
	end
end

project!(t, i, e::Equality{T, D}, x::AbstractArray{T}) where {D, T} = (for j=i:i+D-1 @inbounds t[j] = e.v[j] end)

function project!(t, i, c::SOCone{T, D}, x::AbstractArray{T}) where {T, D}
	angle = c.angle
	xnorm = norm(@~ x[i .+ (1:(D-1))])
	r = x[i]
	if xnorm <= angle * r 
		@inbounds @.. t[i:i+D-1] .= x[i:i+D-1]
	elseif xnorm <= -r/angle
		@inbounds @.. t[i:i+D-1] .= zero(T)
	else
		scalefact = (angle * xnorm + r)/(angle * angle + one(T))
		component_factor = angle * (scalefact)/xnorm
		@inbounds t[i] = scalefact
		@inbounds @.. t[i+1:i+D-1] .= component_factor .* x[i+1:i+D-1]
	end
end

function indexednorm(x::AbstractArray{T}, offs, inds) where T
	sum = zero(T)
	for ind in inds
		sum += x[offs + ind] ^ 2
	end
	return sqrt(sum)
end
@generated function project!(t, i, c::Polar{T, D, SOCone{T, D}}, x::AbstractArray{T}) where {T, D}
	onev = one(T)
	vect_inds = SVector{D-1}((2:D) .- 1)
	return quote
		angle = 1/c.inner.angle
		xnorm = indexednorm(x, i, $vect_inds)
		r = -x[i]
		if xnorm <= angle * r
			@inbounds (t[i:i+$D-1]) .= x[i:i+$D-1]
		elseif xnorm <= -r/angle
			@inbounds (t[i:i+$D-1]) .= zero(T)
		else 
			scalefact = (angle * xnorm + r)/(angle * angle + $onev)
			component_factor = angle * (scalefact)/xnorm
			@inbounds t[i] = -scalefact
			for j = i+1:i+$D-1
				@inbounds t[j] = component_factor * x[j]
			end
		end
	end
end


@generated function project!(t, i, c::Union{PTCone{T, Cs, D}, PTSpace{T, Cs, D}},  x::AbstractArray{T}) where {T, Cs, D}
	if length(Cs.parameters) == 0 || D == 0 return :(return) end
	idxes = prepend!(accumulate(+, dim.(Cs.parameters); init=1), 1)
	out = :(begin end)
	for (ind, offs) in enumerate(idxes[1:end-1])
		offs -= 1
		push!(out.args, :(project!(t, i+$offs, c.cones[$ind], x)))
	end
	push!(out.args, :(return))
	return out
end

project!(t, i, p::Union{PermutedCone{T, D}, PermutedSpace{T, D}}, x::AbstractArray{T}) where {T, D} = project!(view(t, p.inverse_permutation), i, p.cone, view(x, p.inverse_permutation))