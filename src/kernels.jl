# kernels.jl — 门作用核心：Hastings 式更新 + SWAP 聚集
#
# 对纯态 MPS（物理维 2）与密度算符 MPS（物理维 4）通用；
# `DensityOperatorMPS` 的门先经 `_superoperator` 映成融合指标空间上的
# 等维矩阵，再走与纯态完全相同的门作用机制。
#
# 约定：量子态 = 站点张量网络的普通收缩（网络中不含权重）；
# `svectors[b]` 是键 `b` 的 Schmidt 谱（正则形式的元数据）。
#
# Hastings 更新（相邻块 [a, a+N-1] 上的 N 体门）：
#   1. 把块内站点收缩为未加权块张量 B（Dl, d^N, Dr），门作用在物理轴上；
#   2. 自左向右逐键：以键谱 Λ 对块加权后做（可截断）SVD，
#      右站点 := v（右正则），左站点 := 未加权块 B·v†（Hastings 重构，
#      避免除以小奇异值；无截断时状态精确重构），
#      新键谱 := S，下一级的未加权块 := v、预条件子 := S；
#   3. 键 [a+1, a+N-1] 的谱被更新，块外键维与谱不变。
#
# 性质：门为酉时，若初态正则则全程保持正则形式（键谱同步更新）；
# 非幺正算子（投影 / Kraus）会破坏正则性，但状态始终精确。

const SWAP2 = Matrix{Float64}([1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1])

"把门矩阵的比特轴从 `positions`（MSB-first）顺序重排为站点升序、site a = 最低位。"
function _reorder_gate(m::AbstractMatrix, positions::NTuple{N,Int}) where {N}
    d = 1 << N
    (size(m, 1) == size(m, 2) == d) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N qubit(s)"))
    srt = sortperm(collect(positions); rev=true)
    srt == collect(1:N) && return m   # positions 降序给出时已满足 site a = 最低位
    perm = (srt..., srt .+ N...)
    m2 = reshape(permutedims(reshape(m, ntuple(_ -> 2, 2N)), perm), d, d)
    return m2
end

"把局域超算符（d²×d²）的比特轴重排为站点升序、site a = 最低位。"
function _reorder_superop(m::AbstractMatrix, positions::NTuple{N,Int}) where {N}
    d2 = 1 << (2N)
    (size(m, 1) == size(m, 2) == d2) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N qubit(s)"))
    srt = sortperm(collect(positions); rev=true)
    srt == collect(1:N) && return m
    perm = (srt..., srt .+ N...)
    return reshape(permutedims(reshape(m, ntuple(_ -> 4, 2N)), perm), d2, d2)
end

"""
    _superoperator(m, N) -> Matrix

把作用在 ket 上的 `N` 体门 `m`（`2^N × 2^N`，site 1 为最高位）映射为
密度算符 MPS 融合指标空间上的超算符 `M[(ī,j̄),(ī',j̄')] = m[ī,ī'] conj(m)[j̄,j̄']`
（`4^N × 4^N`，逐站点 (i,j) 交织）。
"""
function _superoperator(m::AbstractMatrix, N::Int)
    d = 1 << N
    (size(m, 1) == size(m, 2) == d) ||
        throw(ArgumentError("matrix size $(size(m)) does not match $N qubit(s)"))
    K = kron(conj(m), m)                                # (4^N × 4^N)；位序：ī 块（i₁ 最快）× j̄ 块
    axes2 = ntuple(_ -> 2, 2N)
    K4 = reshape(K, axes2..., axes2...)                 # (ī₁…ī_N, j̄₁…j̄_N, ī'…, j̄'…)
    rp = ntuple(k -> isodd(k) ? (k + 1) ÷ 2 : N + k ÷ 2, 2N)   # → (ī₁, j̄₁, ī₂, j̄₂, …)
    perm = (rp..., rp .+ 2N...)
    return reshape(permutedims(K4, perm), d * d, d * d)
end

"把站点 `a … a+N-1` 收缩为未加权块张量（Dl, d^N, Dr；合并物理指标的最低位为 site a）。"
function _contract_block(psi::AbstractMPS, a::Int, N::Int)
    Θ = psi[a]                                                   # (Dl, d, Dr)
    d = size(Θ, 2)
    Dl, Dr = size(Θ, 1), size(Θ, 3)
    for k in 1:(N-1)
        A = psi[a+k]
        Θ = reshape(Θ, Dl * d^k, Dr) * reshape(A, Dr, d * size(A, 3))
        Dr = size(A, 3)
        Θ = reshape(Θ, Dl, d^(k+1), Dr)
    end
    return Θ
end

"把门矩阵 `G` 作用在块张量的物理轴上（返回新张量）。"
function _apply_gate_tensor(G::AbstractMatrix, Θ::AbstractArray{T,3}) where {T}
    dn = size(Θ, 2)
    Θp = permutedims(Θ, (2, 1, 3))
    Θm = reshape(Θp, dn, :)
    Θ2 = reshape(G * Θm, dn, size(Θ, 1), size(Θ, 3))
    return permutedims(Θ2, (2, 1, 3))
end

"""
    _apply_gate_block!(psi, G, a, N, trunc) -> Float64

把 `N` 体门 `G`（站点升序、site `a` 为矩阵最高位）作用到相邻块
`a … a+N-1` 上（就地，Hastings 式更新），返回最大单键截断误差。
要求 `psi.svectors` 已初始化（`N ≥ 2` 时需要键 `a` 的谱作预条件子）。
"""
function _apply_gate_block!(psi::AbstractMPS, G::AbstractMatrix, a::Int, N::Int,
                            trunc::TruncationScheme)
    d = size(psi[a], 2)
    B = _apply_gate_tensor(G, _contract_block(psi, a, N))
    Dr = size(B, 3)
    Dl = size(B, 1)
    err = 0.0
    Λ = nothing                             # 键 a 的谱（预条件子，不被门改变）
    for k in 1:(N-1)
        Λ = k == 1 ? psi.svectors[a] : Λ
        # 加权块 → SVD（分组：(l, p') | (其余物理, r)）
        W = reshape(Λ .* B, Dl * d, :)
        u, s, v, e = tsvd!(W; trunc=trunc)
        err = max(err, e)
        D2 = length(s)
        # Hastings 重构：左站点 = 未加权块 B × v†
        Ah = _hastings_left(B, v, d)
        psi.data[a+k-1] = reshape(Ah, Dl, d, D2)
        # 键谱更新；未加权块与预条件子前移
        psi.svectors[a+k] = convert(Vector{real(eltype(psi))}, s)
        B = reshape(copy(v), D2, :, Dr)
        Λ = s
        Dl = D2
    end
    psi.data[a+N-1] = reshape(B, Dl, d, Dr)
    return err
end

"`Ah[l, p, α] = Σ_rest B[l, p, rest] conj(v[α, rest])`（Hastings 左站点重构，`d` 为首物理腿维）。"
function _hastings_left(B::AbstractArray{T,3}, v::AbstractMatrix, d::Int) where {T}
    Bm = reshape(B, size(B, 1) * d, :)
    out = Bm * adjoint(v)
    return reshape(out, size(B, 1), d, size(v, 1))
end

"""
    _apply_gate_1site!(psi, m, key) -> Float64

单比特门：直接缩并进站点张量。酉门严格保持右正则与全部键谱。
"""
function _apply_gate_1site!(psi::AbstractMPS, m::AbstractMatrix, key::Int)
    A = psi[key]
    tmp = similar(A, promote_type(eltype(A), eltype(m)))
    @tensor tmp[-1, -2, -3] := m[-2, 1] * A[-1, 1, -3]
    psi[key] = tmp
    return 0.0
end

"相邻两站点逻辑内容交换（就地），返回截断误差。"
function _swap_adjacent!(psi::AbstractMPS, j::Int, trunc::TruncationScheme)
    d = size(psi[j], 2)
    if d == 2
        return _apply_gate_block!(psi, SWAP2, j, 2, trunc)
    end
    G = zeros(d * d, d * d)   # 交换两条 d 维腿：G[(pₐ,p_b),(qₐ,q_b)] = δ(pₐ-q_b)δ(p_b-qₐ)
    for pa in 0:d-1, pb in 0:d-1, qa in 0:d-1, qb in 0:d-1
        G[pa + d*pb + 1, qa + d*qb + 1] = (pa == qb) && (pb == qa) ? 1.0 : 0.0
    end
    return _apply_gate_block!(psi, G, j, 2, trunc)
end

"""
    _gather!(psi, positions, trunc) -> Vector{Pair{Int,Int}}

用相邻 SWAP 把 `positions`（升序）上的比特聚集到连续块
`positions[1] … positions[1]+N-1`（就地），返回已执行的交换序列
（供 [`_scatter!`](@ref) 撤销）。
"""
function _gather!(psi::AbstractMPS, positions::Vector{Int}, trunc::TruncationScheme)
    N = length(positions)
    swaps = Pair{Int,Int}[]
    for k in 2:N
        s = positions[k]
        t = positions[1] + k - 1
        while s > t
            _swap_adjacent!(psi, s - 1, trunc)
            push!(swaps, (s - 1) => s)
            s -= 1
        end
    end
    return swaps
end

"按逆序撤销 `_gather!` 的交换序列（就地）。"
function _scatter!(psi::AbstractMPS, swaps::Vector{Pair{Int,Int}}, trunc::TruncationScheme)
    for (i, j) in reverse(swaps)
        _swap_adjacent!(psi, i, trunc)
    end
    return psi
end
