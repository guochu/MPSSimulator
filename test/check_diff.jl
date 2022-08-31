
function check_mps_qterm_expec_grad_a(::Type{T}, L::Int) where {T<:Number}
	v0 = randn(L)
	v = randn(L)
	target_state = qubit_encoding_mps(T, v0)
	observer = QubitsTerm(1=>"Z", 3=>"Z", coeff=0.37)
	loss(x) = real(expectation(observer, qubit_encoding_mps(T, x)))

	grad1 = gradient(loss, v)[1]
	grad2 = fdm_gradient(loss, v)
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_mps_qterm_expec_grad_b(::Type{T}, L::Int) where {T<:Number}
	v0 = randn(L)
	v = randn(L)
	target_state = qubit_encoding_mps(T, v0)
	observer = QubitsTerm(1=>"+", 3=>"-", coeff=0.7)
	loss(x) = real(expectation(observer, qubit_encoding_mps(T, x)))

	grad1 = gradient(loss, v)[1]
	grad2 = fdm_gradient(loss, v)
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_mps_ham_expec_grad_a(::Type{T}, L::Int) where {T<:Number}
	v0 = randn(L)
	v = randn(L)
	target_state = qubit_encoding_mps(T, v0)
	observer = QubitsTerm(1=>"Z", 3=>"Z", coeff=0.37) + QubitsTerm(1=>"X", coeff=0.7)
	loss(x) = real(expectation(observer, qubit_encoding_mps(T, x)))

	grad1 = gradient(loss, v)[1]
	grad2 = fdm_gradient(loss, v)
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end

function check_mps_ham_expec_grad_b(::Type{T}, L::Int) where {T<:Number}
	v0 = randn(L)
	v = randn(L)
	target_state = qubit_encoding_mps(T, v0)
	observer = QubitsTerm(1=>"Z", 3=>"Z", coeff=0.37) + QubitsTerm(1=>"+", 2=>"-", coeff=0.7)
	loss(x) = real(expectation(observer, qubit_encoding_mps(T, x)))

	grad1 = gradient(loss, v)[1]
	grad2 = fdm_gradient(loss, v)
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end



function circuit_grad(L::Int, depth::Int)
	target_state = qubit_encoding_mps(ComplexF64, randn(L))
	initial_state = qubit_encoding_mps(ComplexF64, randn(L))
	circuit =  variational_circuit_1d(L, depth)

	observer = QubitsTerm(1=>"Z", 3=>"X", coeff=0.37)

	loss(x) = abs(expectation(observer, apply(x, initial_state)))
	loss_fd(θs) = loss(variational_circuit_1d(L, depth, θs=θs))

	grad1 = gradient(loss, circuit)[1]
	grad2 = fdm_gradient(loss_fd, active_parameters(circuit))
	return maximum(abs.(grad1 - grad2)) < 1.0e-6
end


@testset "gradient of quantum operator expectation value" begin
	for L in 3:6
		@test check_mps_qterm_expec_grad_a(Float64, L)
		@test check_mps_qterm_expec_grad_a(ComplexF64, L)
		@test check_mps_qterm_expec_grad_b(Float64, L)
		@test check_mps_qterm_expec_grad_b(ComplexF64, L)
		@test check_mps_ham_expec_grad_a(Float64, L)
		@test check_mps_ham_expec_grad_a(ComplexF64, L)
		@test check_mps_ham_expec_grad_b(Float64, L)
		@test check_mps_ham_expec_grad_b(ComplexF64, L)
	end
end


@testset "gradient of quantum circuit" begin
	for L in 3:5
		for depth in 0:5
		    @test circuit_grad(L, depth)
		end	    
	end
end
