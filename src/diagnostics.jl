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

struct ConvergenceDiagnostics{T} <: Diagnostics{T}
	last_w::Vector{T}
	last_z::Vector{T}
end

function record_diagnostics(d::ConvergenceDiagnostics{T}, i, w, z, v, w_delta, z_delta) where T 
	if i == 1
		println("iteration | w_max | z_max")
	end
	if i % 500 == 0
		diff_w = w .- d.last_w
		diff_z = z .- d.last_z
		println("$i | $(diff_w[sortperm(abs.(diff_w); rev=true)[1:10]]) | $(sortperm(abs.(diff_z); rev=true)[1:10])")
	end
	d.last_w .= w
	d.last_z .= z
end