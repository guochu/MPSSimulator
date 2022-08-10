push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test
using QuantumCircuits, QuantumCircuits.Gates, QuantumSpins
using MPSSimulator, MPSSimulator.Utilities
using Zygote

include("util.jl")

println("-----------test quantum circuit simulation-----------------")

function to_digits(s::Vector{Int})
	r = 0.
	for i = 1:length(s)
		r = r + s[i]*(2.)^(-i)
	end
	return r
end

function phase_estimate_circuit(j::Vector{Int})
	L = length(j)
	circuit = QCircuit()
	phi = to_digits(j)
	U = [exp(2*pi*im*phi) 0; 0. 1.]
	for i = 1:L
		push!(circuit, gate(i, H))
	end

	tmp = U
	for i = L:-1:1
		push!(circuit, gate((i, L+1), CONTROL(tmp)))
		tmp = tmp * tmp
	end
	append!(circuit, QFT(L)')
	return circuit
end


function simple_phase_estimation_1(L::Int, auto_reset::Bool=false)
	j = rand(0:1, L)
	state = statevector_mps(L+1)
	phi = to_digits(j)
	circuit = phase_estimate_circuit(j)
	apply!(circuit, state)
	res = Int[]
	for i = 1:(L+1)
		i, p = measure!(state, i, keep=true, auto_reset=auto_reset)
		push!(res, i)
	end
	phi_out = to_digits(res)
	return (phi == phi_out) && (j[1:L] == res[1:L])
end

function simple_phase_estimation_2(L::Int, auto_reset::Bool=false)
	j = rand(0:1, L)
	state = statevector_mps(L+1)
	phi = to_digits(j)
	circuit = phase_estimate_circuit(j)
	apply!(circuit, state)
	res = Int[]
	for i = 1:(L+1)
		i, p = measure!(state, 1, keep=false, auto_reset=auto_reset)
		push!(res, i)
	end
	phi_out = to_digits(res)
	return (phi == phi_out) && (j[1:L] == res[1:L])
end

function dm_phase_estimation_1(L::Int, auto_reset::Bool=false)
	j = rand(0:1, L)
	state = densitymatrix_mps(L+1)
	phi = to_digits(j)
	circuit = phase_estimate_circuit(j)
	apply!(circuit, state)

	res = Int[]
	for i = 1:(L+1)
		i, p = measure!(state, i, auto_reset=auto_reset)
		push!(res, i)
	end
	phi_out = to_digits(res)
	return (phi == phi_out) && (j[1:L] == res[1:L])
end

@testset "test quantum phase estimation algorithm" begin
	for L in [5, 6]
		@test simple_phase_estimation_1(L, false)
		@test simple_phase_estimation_1(L, true)
		@test simple_phase_estimation_2(L, false)
		@test simple_phase_estimation_2(L, true)

		@test dm_phase_estimation_1(L, true)
	end
end

println("-----------test parametric quantum circuit-----------------")

@testset "test expectation value gradient" begin
    include("check_diff.jl")
end


