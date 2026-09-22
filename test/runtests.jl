# runtests.jl — MPSSimulator 测试
#
# 运行方式（本地包经 LOAD_PATH 解析；注意顺序：QuantumCircuits/FiniteMPSAlgorithms/VQC
# 必须在 MPSSimulator 之前入栈——后者把前三者声明为依赖且无 Manifest，先入栈会短路解析）：
#   JULIA_PROBE_LIBSTDCXX=0 JULIA_DEPOT_PATH=/tmp/jdepot: julia --startup-file=no -e '
#     push!(LOAD_PATH, "../QuantumCircuits"); push!(LOAD_PATH, "../FiniteMPSAlgorithms");
#     push!(LOAD_PATH, "../VQC"); push!(LOAD_PATH, "../MPSSimulator");
#     include("test/runtests.jl")'

using Test
using LinearAlgebra
using Random

using QuantumCircuits
using VQC

using MPSSimulator
const MS = MPSSimulator

# 两个后端共享的泛型函数名（apply!/measure!/expectation/…）在此统一绑定
# 到 MPSSimulator 的版本；`simulate` 来自 QuantumCircuits.Interface（按态类型
# 派发到对应后端）；VQC 特有的调用用 `VQC.` 限定。
import MPSSimulator: apply, apply!, apply_kraus!, reset_qubit_zero!,
                     measure!, probabilities, marginal_probabilities, sample,
                     post_select, post_select!,
                     partial_tr, expectation, fidelity, amplitude, statevector,
                     canonicalize!, normalize!,
                     distance, distance2
import QuantumCircuits.Interface: simulate!

Random.seed!(20260908)

@testset "MPSSimulator" begin
    include("tensorops.jl")
    include("states.jl")
    include("apply.jl")
    include("mixed.jl")
    include("measure.jl")
    include("expect.jl")
    include("circuit.jl")
end
