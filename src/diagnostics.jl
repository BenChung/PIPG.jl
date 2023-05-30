abstract type Diagnostics{T} end
struct NoDiagnostics{T} <: Diagnostics{T} end
struct LogDiagnostics{T} <: Diagnostics{T} end
record_diagnostics(d::NoDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T = nothing
function record_diagnostics(d::LogDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T 
	if i == 1
		println("iteration | w | w_delta | z | z_delta")
	end
	if i % 500 == 0
		println("$i | $w | $w_delta | $z | $z_delta")
	end
end
