# canonical_form.jl — 随机线路上验证门作用过程的正则形式保持
#
# 对随机生成的量子线路逐门作用，检查：
#   1. 纯态 MPS：每个酉门（含非相邻 SWAP 聚集路径）作用后 `iscanonical == true`；
#   2. 混合态 DMPS：每个酉门作用后 `iscanonical == true`（迹保持不变）；
#   3. 非幺正 Kraus 信道：正则性被破坏（预期），`canonicalize!` 可恢复，
#      且全程与 VQC 稠密后端逐步对照一致。
#
# 运行：
#   JULIA_PROBE_LIBSTDCXX=0 JULIA_DEPOT_PATH=/tmp/jdepot: julia --startup-file=no \
#     debug/canonical_form.jl

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(dirname(ROOT), "QuantumCircuits"))
push!(LOAD_PATH, joinpath(dirname(ROOT), "FiniteMPSAlgorithms"))
push!(LOAD_PATH, joinpath(dirname(ROOT), "VQC"))
push!(LOAD_PATH, ROOT)

using Random
using LinearAlgebra
using QuantumCircuits
using QuantumCircuits.Hamiltonian
using MPSSimulator
using VQC
import MPSSimulator: apply!   # VQC 与 MPSSimulator 均导出 apply!，显式消歧

Random.seed!(20260909)

# ── 随机线路构造 ─────────────────────────────────────────────────────────────

"随机量子线路：`depth` 层，每层 = 随机单比特层 + 随机两比特对（含非相邻）。"
function random_circuit(rng::AbstractRNG, n::Int, depth::Int)
    c = Circuit(n)
    for _ in 1:depth
        # 随机单比特层
        for q in 1:n
            if rand(rng) < 0.5
                push!(c, H(q))
            else
                push!(c, RX(2π * rand(rng), q))
            end
        end
        # 随机两比特对（允许非相邻 → 触发 SWAP 聚集路径）
        for _ in 1:n
            a, b = rand(rng, 1:n, 2)
            a == b && continue
            if rand(rng) < 0.5
                push!(c, CZ(a, b))
            else
                push!(c, RZZ(2π * rand(rng), a, b))
            end
        end
    end
    return c
end

"逐步作用线路并检查正则形式；返回每步是否正则。"
function check_canonical_sweep(state, ops; label)
    flags = Bool[]
    for (k, op) in enumerate(ops)
        op isa BarrierOp && continue
        apply!(state, op)
        ok = iscanonical(state)
        push!(flags, ok)
        println("  [$label] gate $k/$(length(ops)): iscanonical = $ok",
                "  (键维 = ", bonddim(state), ")")
    end
    return flags
end

# ── 1. 纯态 MPS：酉门全程保持正则形式 ────────────────────────────────────────

n, depth = 5, 3
rng = Random.MersenneTwister(42)
circuit = random_circuit(rng, n, depth)
println("随机线路：$n 比特，$(length(circuit.ops)) 个指令（含非相邻两比特门）")

println("\n[1] 纯态 MPS 逐门正则性检查")
ψ = zero_state_mps(n)
flags_pure = check_canonical_sweep(ψ, circuit.ops; label = "MPS")
@assert all(flags_pure) "纯态 MPS 存在正则性破坏！"

# 与 VQC 稠密后端终态对照
ref = simulate(circuit, VQC.zero_state(n))
println("  终态 vs VQC maxdiff = ", maximum(abs.(statevector(ψ) .- VQC.storage(ref))))
@assert statevector(ψ) ≈ VQC.storage(ref) atol = 1e-10

# ── 2. 混合态 DMPS：酉门全程保持正则形式（迹不变）────────────────────────────

println("\n[2] 混合态 DMPS 逐门正则性检查（同一酉线路）")
ρ = densitymatrix_mps(n)
flags_mixed = check_canonical_sweep(ρ, circuit.ops; label = "DMPS")
@assert all(flags_mixed) "混合态 DMPS 存在正则性破坏！"
@assert abs(tr(ρ) - 1) < 1e-10
println("  tr(ρ) = ", tr(ρ))

refr = simulate(circuit, VQC.DensityMatrix(VQC.zero_state(n)))
d = maximum(abs.(densitymatrix(ρ) .- VQC.storage(refr)))
println("  终态 vs VQC maxdiff = ", d)
@assert d < 1e-10

# ── 3. 非幺正 Kraus 信道：正则性破坏 → canonicalize! 恢复 ────────────────────

println("\n[3] 插入 Kraus 信道（振幅阻尼，非幺正）")
γ = 0.3
ch = KrausChannel([[1 0; 0 sqrt(1 - γ)], [0 sqrt(γ); 0 0]])
ρ2 = densitymatrix_mps(n)
for (k, op) in enumerate(circuit.ops)
    op isa BarrierOp && continue
    apply!(ρ2, op)
    if k == length(circuit.ops) ÷ 2
        apply!(ρ2, QuantumCircuits.KrausOp(ch, 2))  # 线路中段插入信道
        # 非幺正破坏正则性（站点 2 不再右正则）——预期 false
        println("  [DMPS] Kraus@q2 后 iscanonical = ", iscanonical(ρ2; atol = 1e-8),
                "（预期 false：非幺正破坏正则性）")
        canonicalize!(ρ2)
        println("  [DMPS] canonicalize! 后 iscanonical = ",
                iscanonical(ρ2; atol = 1e-8), "，tr = ", tr(ρ2))
        @assert abs(tr(ρ2) - 1) < 1e-10 "信道后迹应保持 1"
        @assert iscanonical(ρ2; atol = 1e-8) "canonicalize! 应恢复正则形式"
    end
end

# 带信道的完整线路与 VQC 对照
c_noisy = Circuit(n)
for (k, op) in enumerate(circuit.ops)
    op isa BarrierOp && continue
    push!(c_noisy, op)
    k == length(circuit.ops) ÷ 2 && push!(c_noisy, QuantumCircuits.KrausOp(ch, 2))
end
ρ3 = simulate(c_noisy, densitymatrix_mps(n))
ref3 = simulate(c_noisy, VQC.DensityMatrix(VQC.zero_state(n)))
d3 = maximum(abs.(densitymatrix(ρ3) .- VQC.storage(ref3)))
println("  带信道终态 vs VQC maxdiff = ", d3)
@assert d3 < 1e-10

println("\n全部检查通过：酉门演化保持正则形式；非幺正破坏可由 canonicalize! 恢复；")
println("全程与 VQC 稠密后端一致。")
