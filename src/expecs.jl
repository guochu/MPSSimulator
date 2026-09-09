# expecs.jl — 期望值 / 重叠 / 稠密转换
#
# 纯态：⟨ψA| Πₖ Oₖ |ψB⟩ 经双网络 + 局域插入的转移收缩；
# 混合态：tr(ρ Πₖ Oₖ) 经单网络（融合指标）+ δ/O 插入的转移收缩。

"""
    expectation(ops, psiA::MPS, psiB::MPS) -> Complex

纯态局域算符链矩阵元 `⟨ψA| O₁ O₂ … |ψB⟩`：`ops::Vector{Pair{Int,AbstractMatrix}}`
（qubit → 局域矩阵，任意顺序），缺失比特为恒等。
"""
function expectation(ops::AbstractVector{<:Pair{Int,<:AbstractMatrix}}, psiA::MPS, psiB::MPS)
    L = length(psiA)
    (L == length(psiB)) || throw(DimensionMismatch("MPS length mismatch"))
    ops = Dict{Int,AbstractMatrix}(ops)
    for (q, O) in ops
        1 <= q <= L || throw(ArgumentError("qubit index out of range [1, $L]"))
        (size(O, 1) == size(O, 2) == 2) ||
            throw(ArgumentError("local matrix size $(size(O)) does not match 1 qubit"))
    end
    E = _ones_env(psiA, psiB)
    for k in 1:L
        A = psiA[k]
        B = psiB[k]
        E = if haskey(ops, k)
            O = ops[k]
            ET = promote_type(eltype(E), eltype(A), eltype(B), eltype(O))
            E2 = similar(E, ET, size(A, 3), size(B, 3))
            @tensor E2[a, b] := conj(A[x, i2, a]) * O[i2, i] * E[x, y] * B[y, i, b]
            E2
        else
            _transfer2(E, A, B)
        end
    end
    return E[1, 1]
end

expectation(ops::AbstractVector{<:Pair{Int,<:AbstractMatrix}}, psi::MPS) = expectation(ops, psi, psi)

"""
    expectation(ops, rho::DensityOperatorMPS) -> Complex

`tr(ρ O₁ O₂ …)`：`ops` 为局域算符（qubit → 矩阵），缺失比特为恒等。
空 `ops` 给出 `tr ρ`。
"""
function expectation(ops::AbstractVector{<:Pair{Int,<:AbstractMatrix}}, rho::DensityOperatorMPS)
    L = length(rho)
    ops = Dict{Int,AbstractMatrix}(ops)
    for (q, O) in ops
        1 <= q <= L || throw(ArgumentError("qubit index out of range [1, $L]"))
        (size(O, 1) == size(O, 2) == 2) ||
            throw(ArgumentError("local matrix size $(size(O)) does not match 1 qubit"))
    end
    E = ones(promote_type(eltype(rho), Float64), 1)
    for k in 1:L
        C = rho[k]
        C4 = reshape(C, size(C, 1), 2, 2, size(C, 3))
        E = if haskey(ops, k)
            O = ops[k]
            ET = promote_type(eltype(E), eltype(C), eltype(O))
            E2 = similar(E, ET, size(C, 3))
            @tensor E2[a] := C4[l, i, j, a] * O[j, i] * E[l]
            E2
        else
            _dtransfer(E, C)
        end
    end
    return E[1]
end

"""
    overlap(psiA::MPS, rho::DensityOperatorMPS, psiB::MPS) -> Complex

三网络重叠 `⟨ψA|ρ|ψB⟩`。
"""
function overlap(psiA::MPS, rho::DensityOperatorMPS, psiB::MPS)
    L = length(psiA)
    (L == length(psiB) == length(rho)) || throw(DimensionMismatch("MPS length mismatch"))
    E = ones(promote_type(eltype(psiA), eltype(psiB), eltype(rho), Float64), 1, 1, 1)
    for k in 1:L
        A = psiA[k]
        B = psiB[k]
        C = rho[k]
        C4 = reshape(C, size(C, 1), 2, 2, size(C, 3))
        E2 = similar(E, promote_type(eltype(E), eltype(A), eltype(B), eltype(C)),
                     size(C, 3), size(A, 3), size(B, 3))
        @tensor E2[a2, b2, c2] := conj(A[x, i, a2]) * E[x, y, z] * C4[y, i, j, b2] * B[z, j, c2]
        E = E2
    end
    return E[1, 1, 1]
end

"""
    amplitude(psi::MPS, bits::AbstractVector{Int}; scaling=1.0) -> Complex

读取振幅 `⟨bits|ψ⟩`（`bits[i]` 为 qubit `i-1` 的取值 0/1）。
"""
function amplitude(psi::MPS, bits::AbstractVector{Int}; scaling::Real=1.0)
    L = length(psi)
    length(bits) == L || throw(DimensionMismatch("input basis mismatch with number of qubits"))
    all(b -> b == 0 || b == 1, bits) || throw(ArgumentError("the qubit should either be 0 or 1."))
    E = ones(promote_type(eltype(psi), Float64), 1)
    for k in 1:L
        A = psi[k]
        E = transpose(view(A, :, bits[k] + 1, :)) * E
    end
    return scaling == 1 ? E[1] : E[1] * scaling^L
end

"""
    statevector(psi::MPS) -> Vector

稠密振幅向量（长度 `2^L`，小端序）。指数级内存，仅限小系统。
"""
function statevector(psi::MPS)
    v = reshape(psi[1], 2, :)
    for k in 2:length(psi)
        A = psi[k]
        v = reshape(v, 2^(k-1), :) * reshape(A, size(A, 1), :)
    end
    return vec(v)
end

"""
    densitymatrix(rho::DensityOperatorMPS) -> Matrix

稠密密度矩阵（`2^L × 2^L`，行 = ket，小端序）。指数级内存，仅限小系统。
"""
function densitymatrix(rho::DensityOperatorMPS)
    G = _dm_open_first(rho[1])
    for k in 2:length(rho)
        G = _dm_open_append(G, rho[k])
    end
    return G
end

# C (1, 4, D)：输出 (2, 2D)：行 = i（ket-open），列 = (j, 键)
function _dm_open_first(C::AbstractArray{T,3}) where {T}
    size(C, 1) == 1 || throw(DimensionMismatch("boundary bond must be 1"))
    return reshape(C, 2, 2 * size(C, 3))
end

# G (2^k, 2^k·Dl)（行 = ket-open，列 = (bra-open, 键)）+ C (Dl, 4, Dr)
# → (2^(k+1), 2^(k+1)·Dr)
function _dm_open_append(G::AbstractMatrix, C::AbstractArray{T,3}) where {T}
    K = size(G, 1)
    Dl, Dr = size(C, 1), size(C, 3)
    G4 = reshape(G, K, :, Dl)                                      # (ī, j̄, l)
    Y = reshape(G4, K * K, Dl) * reshape(C, Dl, 4 * Dr)            # (ī j̄), (i j) r
    Y = reshape(Y, K, K, 2, 2, Dr)                                 # (ī, j̄, i, j, r)
    Yp = permutedims(Y, (1, 3, 2, 4, 5))                           # (ī, i, j̄, j, r)
    return reshape(Yp, K * 2, K * 2 * Dr)
end

"""
    fidelity(x, y) -> Float64

Uhlmann 保真度：

* `fidelity(psi::MPS, psi′::MPS)` = `|⟨psi|psi′⟩|²`；
* `fidelity(rho::DensityOperatorMPS, psi::MPS)` = `⟨psi|rho|psi⟩`；
* 参数顺序可交换。

两个混合态之间的保真度未实现（需要 `sqrt(ρ)` 的 MPS 近似）。
"""
fidelity(x::MPS, y::MPS) = abs2(dot(x, y))

function fidelity(x::DensityOperatorMPS, y::DensityOperatorMPS)
    throw(ArgumentError("fidelity between two mixed MPS states is not supported"))
end

fidelity(x::DensityOperatorMPS, y::MPS) = real(overlap(y, x, y))
fidelity(x::MPS, y::DensityOperatorMPS) = fidelity(y, x)
