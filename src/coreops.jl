# coreops.jl — 通用态操作原语：局域矩阵 / Kraus 算子集 + 比特位置
#
# 约定（与 QuantumCircuits 一致）：
#   * 比特位置 1-based、小端序（qubit 1 = 最低有效位）；
#   * `positions[1]` 为矩阵最高位（`GateOp.qubits` 的矩阵约定）；
#   * 位置可为任意互异比特（非相邻时自动以相邻 SWAP 聚集后作用、再散开）；
#   * 所有作用型接口接受 `trunc` 关键字（`TruncationScheme`），
#     默认 `DefaultMPSTruncation`。
#
# 正则形式管理：若键谱未初始化，多体门作用前会先做一次
# `canonicalize!`（单比特门不需要键谱，无需正则化）。门作用机制复用
# FiniteMPSAlgorithms：两体门走 `apply!(::UnitaryGate/GeneralGate, CanonicalMPS)`
# （Hastings 更新；`GeneralGate` 作用后自动重新正则化），SWAP 走
# `FiniteMPSAlgorithms.swap!`；仅单比特门与 N ≥ 3 体相邻块门使用本地内核。
# 酉门严格保持正则形式；非幺正算子（投影 / Kraus）可能破坏正则性——
# 状态始终精确，需要时可用 `canonicalize!` 恢复。

_check_positions(positions::NTuple{N,Int}, n::Int) where {N} = begin
    allunique(positions) || throw(ArgumentError("duplicate qubit positions"))
    all(q -> 1 <= q <= n, positions) ||
        throw(ArgumentError("qubit position out of range [1, $n]"))
    positions
end

function _ensure_complex(psi::AbstractMPS)
    eltype(psi) <: Complex && return psi
    return convert(AbstractMPS{complex(float(eltype(psi)))}, psi)
end

"把站点升序的 `d^N × d^N` 矩阵包成 FiniteMPSAlgorithms 的两体门（酉性自动检查）。"
function _local_gate(m::AbstractMatrix, positions::NTuple{2,Int}, d::Int)
    G = reshape(m, d, d, d, d)   # (i₁′, i₂′, i₁, i₂)：site 1（最低位）为慢位
    return isunitary(Matrix(m); atol=1.0e-10) ? UnitaryGate(positions, G) :
           GeneralGate(positions, G)
end

function _apply_matrix!(psi::AbstractMPS, m::AbstractMatrix, positions::NTuple{N,Int},
                        trunc::TruncationScheme) where {N}
    n = length(psi)
    _check_positions(positions, n)
    d = size(psi[1], 2)
    (size(m, 1) == size(m, 2) == d^N) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N site(s) of dimension $d"))
    ps = sort(collect(positions))
    if N == 1
        _apply_gate_1site!(psi, m, ps[1])
    else
        svectors_uninitialized(psi) && canonicalize!(psi; normalize=false)
        if N == 2
            # 两体门：复用 FiniteMPSAlgorithms（Hastings 更新 + 非相邻自动交换）
            apply!(_local_gate(m, Tuple(ps), d), psi.core; trunc=trunc)
            _sync!(psi)
        elseif ps[end] - ps[1] == N - 1
            _apply_gate_block!(psi, m, ps[1], N, trunc)
        else
            swaps = _gather!(psi, ps, trunc)
            _apply_gate_block!(psi, m, ps[1], N, trunc)
            _scatter!(psi, swaps, trunc)
        end
    end
    return psi
end

"""
    apply!(psi::MPS, m::AbstractMatrix, positions::NTuple{N,Int}; trunc=DefaultMPSTruncation) -> state

把 `N` 比特局域矩阵 `m`（`positions[1]` = 矩阵最高位）**就地**作用到
`psi` 上（`ψ ← m ψ`）。实数态遇复矩阵自动提升为 `ComplexF64`
（返回值可能是新对象）。若 `m` 是 `d²×d²` 局域超算符（如 `KrausOp`
的 `mat`），纯态会被提升为 `DensityOperatorMPS`（返回值类型改变）。
"""
function apply!(psi::MPS, m::AbstractMatrix, positions::NTuple{N,Int};
                trunc::TruncationScheme=DefaultMPSTruncation) where {N}
    d = 1 << N
    if size(m, 1) == size(m, 2) == d * d
        # 局域超算符：纯态自动提升为密度算符 MPS
        rho = densitymatrix(psi)
        return apply!(rho, m, positions; trunc=trunc)
    end
    (size(m, 1) == size(m, 2) == d) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N qubit(s)"))
    eltype(psi) <: Real && !(eltype(m) <: Real) && (psi = _ensure_complex(psi))
    return _apply_matrix!(psi, _reorder_gate(m, positions), positions, trunc)
end

"""
    apply!(rho::DensityOperatorMPS, m::AbstractMatrix, positions::NTuple{N,Int}; trunc=DefaultMPSTruncation) -> rho

`ρ ← m ρ m†`（就地）：`m` 为 `d×d` 酉门时先映射为超算符；`m` 为
`d²×d²` 局域超算符（如 `KrausOp` 的 `mat`）时直接作用。
"""
function apply!(rho::DensityOperatorMPS, m::AbstractMatrix, positions::NTuple{N,Int};
                trunc::TruncationScheme=DefaultMPSTruncation) where {N}
    eltype(rho) <: Real && !(eltype(m) <: Real) && (rho = _ensure_complex(rho))
    N == 0 && return rho
    d = 1 << N
    if size(m, 1) == size(m, 2) == d * d
        super_m = _reorder_superop(m, positions)
    else
        super_m = _superoperator(_reorder_gate(m, positions), N)
    end
    _apply_matrix!(rho, super_m, positions, trunc)
    return rho
end

apply!(psi::Union{MPS,DensityOperatorMPS}, m::AbstractMatrix, positions::Vector{Int};
       trunc::TruncationScheme=DefaultMPSTruncation) =
    apply!(psi, m, Tuple(positions); trunc=trunc)

"""
    apply(state, m, positions; trunc=DefaultMPSTruncation) -> state'

`apply!` 的非就地版本。
"""
function apply(psi::Union{MPS,DensityOperatorMPS}, m::AbstractMatrix,
               positions::Union{NTuple{N,Int},Vector{Int}};
               trunc::TruncationScheme=DefaultMPSTruncation) where {N}
    out = copy(psi)
    return apply!(out, m, positions; trunc=trunc)
end

# ── Kraus 信道 ───────────────────────────────────────────────────────────────

"""
    apply_kraus!(rho::DensityOperatorMPS, ks, positions::NTuple{N,Int}; trunc=DefaultMPSTruncation) -> rho

把 Kraus 算子集 `ks`（满足 `Σₖ kₖ† kₖ = I`）作用到 `rho` 上：
`ρ ← Σₖ kₖ ρ kₖ†`（就地）。先把算子集组合为单个局域超算符再作用，
过程中不复制整个态。注意：非幺正演化可能破坏正则形式
（状态保持精确，需要时可用 `canonicalize!` 恢复）。
"""
function apply_kraus!(rho::DensityOperatorMPS, ks, positions::NTuple{N,Int};
                      trunc::TruncationScheme=DefaultMPSTruncation) where {N}
    n = length(rho)
    _check_positions(positions, n)
    isempty(ks) && return rho
    d = 1 << N
    eltype(rho) <: Real && any(K -> !(eltype(K) <: Real), ks) && (rho = _ensure_complex(rho))
    T = eltype(rho)
    M = zeros(T, d * d, d * d)
    for K in ks
        (size(K, 1) == size(K, 2) == d) ||
            throw(ArgumentError("all Kraus operators must be $(d)×$(d)"))
        M .+= _superoperator(K, N)
    end
    _apply_matrix!(rho, M, positions, trunc)
    return rho
end

"""
    apply_kraus!(psi::MPS, ks, positions::NTuple{N,Int}; trunc=DefaultMPSTruncation) -> DensityOperatorMPS

纯态遇信道自动提升为 `DensityOperatorMPS`（返回值类型因此改变）。
"""
function apply_kraus!(psi::MPS, ks, positions::NTuple{N,Int};
                      trunc::TruncationScheme=DefaultMPSTruncation) where {N}
    eltype(psi) <: Real && any(K -> !(eltype(K) <: Real), ks) && (psi = _ensure_complex(psi))
    return apply_kraus!(densitymatrix(psi), ks, positions; trunc=trunc)
end

apply_kraus!(s::Union{MPS,DensityOperatorMPS}, ks, positions::Vector{Int};
             trunc::TruncationScheme=DefaultMPSTruncation) =
    apply_kraus!(s, ks, Tuple(positions); trunc=trunc)

# ── 相干重置 ─────────────────────────────────────────────────────────────────

"""
    reset_qubit_zero!(state, q; trunc=DefaultMPSTruncation) -> state

把 qubit `q` 相干重置到 `|0⟩`（Kraus：`k₀ = |0⟩⟨0|`、`k₁ = |0⟩⟨1|`），
结果自动归一化。
"""
function reset_qubit_zero!(psi::MPS, q::Int; trunc::TruncationScheme=DefaultMPSTruncation)
    n = length(psi)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    k0 = Matrix{eltype(psi)}([1 0; 0 0])
    k1 = Matrix{eltype(psi)}([0 1; 0 0])
    _apply_matrix!(psi, k0 + k1, (q,), trunc)   # ψ'ₓ₀ = ψₓ₀ + ψₓ₁，ψ'ₓ₁ = 0
    normalize!(psi)
    return psi
end

function reset_qubit_zero!(rho::DensityOperatorMPS, q::Int; trunc::TruncationScheme=DefaultMPSTruncation)
    n = length(rho)
    1 <= q <= n || throw(ArgumentError("qubit index $q out of range [1, $n]"))
    k0 = Matrix{eltype(rho)}([1 0; 0 0])
    k1 = Matrix{eltype(rho)}([0 1; 0 0])
    apply_kraus!(rho, (k0, k1), (q,); trunc=trunc)
    normalize_tr!(rho)
    return rho
end
