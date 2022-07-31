

function LinearAlgebra.ishermitian(x::QubitsTerm)
	(imag(QuantumCircuits.coeff(x)) ≈ 0.) || return false
	for item in oplist(x)
		ishermitian(item) || return false
	end
	return true
end