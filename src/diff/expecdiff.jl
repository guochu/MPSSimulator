# gradient of expectation values


function apply!(m::QubitsTerm, state::AbstractMPS; trunc::TruncationScheme=DefaultMPSTruncation)
	errs = [QuantumSpins._apply_impl((k,), Array(v), state, trunc) for (k, v) in zip(QuantumCircuits.positions(m), oplist(m))]
	state[1] *= QuantumCircuits.coeff(m)
	return errs
end


function _qterm_expec_util(m::QubitsTerm, state::AbstractMPS; trunc::TruncationScheme=DefaultMPSTruncation)
	return expectation(m, state), z -> begin
		if ishermitian(m)
			r = copy(state)
			apply!((2 * real(z)) * m, r; trunc=trunc)
		else
			state_a = copy(state)
			state_b = copy(state)
			apply!(conj(z) * m, state_a; trunc=trunc)
			apply!(z * m', state_b; trunc=trunc)
			r = state_a + state_b
		end
		canonicalize!(r, normalize=false, trunc=trunc)
		return (nothing, r)
	end 
end

@adjoint expectation(m::QubitsTerm, state::AbstractMPS) = _qterm_expec_util(m, state)

# function _qop_expec_util(m::QubitsOperator, state::AbstractMPS)
# 	return expectation(m, state), z -> begin
# 		r = (conj(z) * m + z * m') * state
# 		canonicalize!(r)
# 		return (nothing, r)
# 	end 
# end

# @adjoint expectation(m::QubitsOperator, state::StateVector) = _qop_expec_util(m, state)


