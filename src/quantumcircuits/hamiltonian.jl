# hamiltonian.jl — Pauli 代数 / 自旋算符 / GateOp 可观测量的期望值（接口层）
#
# 均不构造满算符矩阵：逐项化为局域算符链 / 门作用处理
# （SpinOp 的稠密展开 `mat(t, n)` 仅供小规模验证 / 对拍）。

using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum, SpinOpTerm, SpinOpSum, _spin_matrix

const _PAULI1 = Dict{Symbol,Matrix{ComplexF64}}(
    :I => [1 0; 0 1],
    :X => [0 1; 1 0],
    :Y => [0 -im; im 0],
    :Z => [1 0; 0 -1],
)

"""
    expectation(t::PauliTerm, s::Union{MPS,DensityOperatorMPS}) -> Complex

Pauli 项的期望值：纯态 `⟨ψ|t|ψ⟩`；混合态 `tr(t ρ)`。
恒等项返回 `t.coeff * ⟨ψ|ψ⟩`（或 `t.coeff * tr(ρ)`）。
"""
function expectation(t::PauliTerm, s::Union{MPS,DensityOperatorMPS})
    n = length(s)
    for (q, _) in t.ops
        1 <= q <= n || throw(ArgumentError("Pauli term qubit index out of range for state with $n qubit(s)"))
    end
    ops = [q => _PAULI1[sym] for (q, sym) in t.ops]
    val = isempty(ops) ? (s isa MPS ? dot(s, s) : tr(s)) : expectation(ops, s)
    return t.coeff * val
end

"""
    expectation(h::PauliSum, s::Union{MPS,DensityOperatorMPS}) -> Complex

Pauli 和的期望值（逐项求和）。
"""
function expectation(h::PauliSum, s::Union{MPS,DensityOperatorMPS})
    acc = zero(ComplexF64)
    for t in h.terms
        acc += expectation(t, s)
    end
    return acc
end

"""
    expectation(t::SpinOpTerm, s::Union{MPS,DensityOperatorMPS}) -> Complex
    expectation(h::SpinOpSum, s::Union{MPS,DensityOperatorMPS}) -> Complex

自旋算符项 / 和的期望值：同一位上的多个因子按乘积序合并为单个 2×2
矩阵后走局域算符链，不构造满矩阵。
"""
function expectation(t::SpinOpTerm, s::Union{MPS,DensityOperatorMPS})
    n = length(s)
    ops = Dict{Int,AbstractMatrix}()
    for (q, a) in t.ops
        1 <= q <= n || throw(ArgumentError("spin term qubit index out of range for state with $n qubit(s)"))
        m = _spin_matrix(a)
        ops[q] = haskey(ops, q) ? ops[q] * m : m    # 同位因子按乘积序右乘合并
    end
    return t.coeff * expectation(collect(ops), s)
end

function expectation(h::SpinOpSum, s::Union{MPS,DensityOperatorMPS})
    isempty(h.terms) && return zero(ComplexF64)
    return mapreduce(t -> expectation(t, s), +, h.terms)
end

"""
    expectation(op::GateOp, s::Union{MPS,DensityOperatorMPS}) -> Complex

把酉门 `op` 当作可观测量：`⟨ψ|U|ψ⟩` 或 `tr(ρ U)`。
"""
function expectation(op::GateOp, s::Union{MPS,DensityOperatorMPS})
    n = length(s)
    _check_gateop(op, n)
    U = _reorder_gate(mat(op), Tuple(qubits(op)))
    if s isa MPS
        # ⟨ψ|U|ψ⟩：把 U 作用到 ket 副本上再取重叠
        return dot(s, apply!(copy(s), U, Tuple(qubits(op))))
    else
        # tr(ρ U)：构造仅作用 ket 侧的超算符 M[(i,j),(i',j')] = U[i,i'] δ[j,j']
        # （已按块约定重排，内部直接走 _apply_matrix! 避免二次重排）
        rho2 = copy(s)
        _apply_matrix!(rho2, _ket_only_superoperator(U, length(qubits(op))),
                       Tuple(qubits(op)), DefaultMPSTruncation)
        return tr(rho2)
    end
end

function _ket_only_superoperator(U::AbstractMatrix, N::Int)
    d = 1 << N
    (size(U, 1) == size(U, 2) == d) ||
        throw(ArgumentError("matrix size $(size(U)) does not match $N qubit(s)"))
    # K = kron(I, U)：行/列复合轴（快→慢）为 (ī 块, j̄ 块)，M[(ī,j̄),(ī',j̄')] = δ_{j̄j̄'} U[ī,ī']
    # 与 `_superoperator` 相同的轴交织（(ī₁,j̄₁,ī₂,j̄₂,…)，site 升序、i 为快位）
    K = kron(Matrix{eltype(U)}(I, d, d), U)
    K4 = reshape(K, ntuple(_ -> 2, 4N)...)
    rp = ntuple(k -> isodd(k) ? (k + 1) ÷ 2 : N + k ÷ 2, 2N)
    perm = (rp..., rp .+ 2N...)
    return reshape(permutedims(K4, perm), d * d, d * d)
end
