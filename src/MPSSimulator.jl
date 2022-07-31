module MPSSimulator

using LinearAlgebra, TensorOperations

using QuantumCircuits, QuantumCircuits.Gates
using QuantumCircuits: ordered_positions, ordered_op
using QuantumSpins
using QuantumSpins: updateright, updateleft, OverlapCache
import QuantumSpins: apply!,expectation, _is_commutative, commutative_blocks, fuse_gates

using Zygote
using Zygote: @adjoint


export apply, apply!, measure!, amplitude, statevector_mps, qubit_encoding_mps, fuse_gates
export expectation
export QFT



const DefaultMPSTruncation = MPSTruncation(D=1000, ϵ=1.0e-6)


include("initializers.jl")

include("apply_gates.jl")

include("gate_fusion.jl")

include("measure.jl")

# expectation value
include("expecs.jl")

# differentiation
include("diff/circuitdiff.jl")
include("diff/expecdiff.jl")


# utilities
include("utility/utility.jl")

end