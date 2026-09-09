# quantumcircuits.jl — QuantumCircuits 桥接层（聚合入口）
#
# 把 IR 指令 / Pauli 代数桥接到核心原语，提供经典寄存器运行时，
# 以及 Interface 后端 `MatrixProductBackend`。

include("classical.jl")
include("ops.jl")
include("hamiltonian.jl")
include("simulate.jl")
include("backend.jl")
