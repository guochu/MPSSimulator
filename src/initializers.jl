# initializers.jl — 态初始化
#
# 命名与 VQC 对齐（加 `_mps` 后缀以避免与稠密后端的导出冲突），
# 同时保留旧版 `statevector_mps` / `densitymatrix_mps` 别名。

"乘积态 MPS：`amps[i]` 为站点 `i` 的局域振幅（长度 2 向量）。"
function _productmps(::Type{T}, amps::AbstractVector{<:AbstractVector}) where {T<:Number}
    L = length(amps)
    L >= 1 || throw(ArgumentError("empty state"))
    data = [Array{T,3}(undef, 1, 2, 1) for _ in 1:L]
    for (k, a) in enumerate(amps)
        length(a) == 2 || throw(ArgumentError("each local amplitude must have length 2"))
        data[k][1, :, 1] .= a
    end
    R = real(T)
    return MPS(data, [[one(R)] for _ in 1:(L+1)])   # 乘积态：平凡键谱
end

"""
    zero_state_mps([T=ComplexF64,] n) -> MPS

`|0…0⟩`。
"""
zero_state_mps(::Type{T}, n::Int) where {T<:Number} =
    _productmps(T, [[one(T), zero(T)] for _ in 1:n])
zero_state_mps(n::Int) = zero_state_mps(ComplexF64, n)

"""
    statevector_mps([T=ComplexF64,] n) -> MPS
    statevector_mps([T=ComplexF64,] bits::Vector{Int}) -> MPS

`|0…0⟩`；或给定计算基态（`bits[i]` = qubit `i-1` 的取值）。
"""
statevector_mps(::Type{T}, states::Vector{Int}) where {T<:Number} = onehot_encoding_mps(T, states)
statevector_mps(::Type{T}, n::Int) where {T<:Number} = zero_state_mps(T, n)
statevector_mps(n::Int) = zero_state_mps(ComplexF64, n)

"""
    onehot_encoding_mps([T=ComplexF64,] bits) -> MPS

计算基态编码：`bits[i]` 为 qubit `i-1` 的取值（0/1）。
"""
function onehot_encoding_mps(::Type{T}, bits::AbstractVector{Int}) where {T<:Number}
    all(b -> b == 0 || b == 1, bits) || throw(ArgumentError("bits must be 0 or 1"))
    return _productmps(T, [b == 0 ? [one(T), zero(T)] : [zero(T), one(T)] for b in bits])
end
onehot_encoding_mps(bits::AbstractVector{Int}) = onehot_encoding_mps(ComplexF64, bits)

"kernal_mapping(s) = [cos(sπ/2), sin(sπ/2)]（qubit 编码的局域振幅）。"
kernal_mapping(s::Real) = [cos(s * pi / 2), sin(s * pi / 2)]

"""
    qubit_encoding_mps([T=ComplexF64,] θs) -> MPS

直积态编码：qubit `i-1` 处于 `cos(πθ/2)|0⟩ + sin(πθ/2)|1⟩`。
"""
qubit_encoding_mps(::Type{T}, v::AbstractVector{<:Real}) where {T<:Number} =
    _productmps(T, kernal_mapping.(v))
qubit_encoding_mps(v::AbstractVector{<:Real}) = qubit_encoding_mps(ComplexF64, v)

"""
    amplitude_encoding_mps([T=ComplexF64,] v; nqubits=…, trunc=NoTruncation()) -> MPS

振幅编码：向量 `v` 归一化后作为 `nqubits` 比特态的振幅
（逐站点 SVD 分解，`trunc` 可选截断）。
"""
function amplitude_encoding_mps(::Type{T}, v::AbstractVector{<:Number};
                                nqubits::Int=ceil(Int, log2(length(v))),
                                trunc::TruncationScheme=NoTruncation()) where {T<:Number}
    nqubits >= 1 || throw(ArgumentError("nqubits must be positive"))
    d = 1 << nqubits
    length(v) <= d || throw(DimensionMismatch("input length $(length(v)) exceeds 2^$nqubits"))
    vn = norm(v)
    vn > 0 || throw(ArgumentError("input vector is zero"))
    isapprox(vn, 1.0) || @warn "input vector is not normalized; it will be renormalized as a quantum state."
    vv = zeros(T, d)
    vv[1:length(v)] .= v ./ vn
    data = Vector{Array{T,3}}(undef, nqubits)
    Θ = reshape(vv, 1, d)
    Dl = 1
    for k in 1:(nqubits-1)
        M = reshape(Θ, Dl * 2, :)
        u, s, vt, err = tsvd!(M; trunc=trunc)
        r = length(s)
        data[k] = reshape(u, Dl, 2, r)
        Θ = reshape(s .* vt, r, :)
        Dl = r
    end
    data[nqubits] = reshape(Θ, Dl, 2, 1)
    psi = MPS(data)
    trunc isa NoTruncation || normalize!(psi)
    return psi
end

amplitude_encoding_mps(v::AbstractVector{<:Number}; kwargs...) =
    amplitude_encoding_mps(ComplexF64, v; kwargs...)

"""
    rand_state_mps([T=ComplexF64,] n; D=8) -> MPS

键维上限 `D` 的随机纯态（高斯张量 + 左正则归一化）。
"""
function rand_state_mps(::Type{T}, n::Int; D::Int=8) where {T<:Number}
    n >= 1 || throw(ArgumentError("number of qubits must be positive"))
    data = Vector{Array{T,3}}(undef, n)
    for k in 1:n
        dl = (k == 1) ? 1 : min(D, 2^(k-1), 2^(n-k+1))
        dr = (k == n) ? 1 : min(D, 2^k, 2^(n-k))
        data[k] = randn(T, dl, 2, dr)
    end
    psi = MPS(data)
    canonicalize!(psi)
    return psi
end

rand_state_mps(n::Int; kwargs...) = rand_state_mps(ComplexF64, n; kwargs...)

"""
    densitymatrix(psi::MPS; trunc=DefaultMPSTruncation) -> DensityOperatorMPS

纯态 → 密度算符 MPS（精确构造后按 `trunc` 压缩键维）。
注意：构造中局域张量占用 `O(D⁴)` 内存，适合中等键维以下的态。
"""
function densitymatrix(psi::MPS; trunc::TruncationScheme=DefaultMPSTruncation)
    T = eltype(psi)
    L = length(psi)
    ψ = copy(psi)
    canonicalize!(ψ)
    data = Vector{Array{T,3}}(undef, L)
    for k in 1:L
        A = ψ[k]
        Dl, d, Dr = size(A)
        # T[l, l', i, j, r, r'] = A[l, i, r] · conj(A[l', j, r'])
        T6 = reshape(A, Dl, 1, d, 1, Dr, 1) .* reshape(conj(A), 1, Dl, 1, d, 1, Dr)
        data[k] = reshape(T6, Dl * Dl, d * d, Dr * Dr)
    end
    rho = DensityOperatorMPS(data)
    canonicalize!(rho; trunc=trunc)
    return rho
end

"""
    densitymatrix_mps([T=ComplexF64,] n) -> DensityOperatorMPS
    densitymatrix_mps([T=ComplexF64,] bits::Vector{Int}) -> DensityOperatorMPS

计算基态的密度算符 MPS（`|bits⟩⟨bits|`）。
"""
function densitymatrix_mps(::Type{T}, states::Vector{Int}) where {T<:Number}
    L = length(states)
    L >= 1 || throw(ArgumentError("empty state"))
    data = [Array{T,3}(undef, 1, 4, 1) for _ in 1:L]
    for (k, b) in enumerate(states)
        (b == 0 || b == 1) || throw(ArgumentError("bits must be 0 or 1"))
        fill!(data[k], zero(T))
        data[k][1, 1 + 2 * b, 1] = one(T)   # 融合指标 p = i + 2j
    end
    R = real(T)
    return DensityOperatorMPS(data, [[one(R)] for _ in 1:(L+1)])
end

densitymatrix_mps(::Type{T}, n::Int) where {T<:Number} = densitymatrix_mps(T, zeros(Int, n))
densitymatrix_mps(n::Int) = densitymatrix_mps(ComplexF64, n)
