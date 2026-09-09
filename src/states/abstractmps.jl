# abstractmps.jl — MPS 状态类型公共抽象与通用接口
#
# 纯态 `MPS`（局域维 2）与混合态 `DensityOperatorMPS`（局域维 4）共用同一
# 数据布局与大多数操作，因此都继承自 `AbstractMPS{T}`：
#
#   * `data`：N 个站点张量 `A::Array{T,3}`，轴序 = (左键, 物理维, 右键)，
#     开放边界（`size(psi[1],1) == 1`、`size(psi[end],3) == 1`）；
#     **量子态就是张量网络的普通收缩**（网络中不含权重）；
#   * `svectors::Union{Nothing, Vector{Vector{R}}}`（`R = real(T)`）：
#     长度 `N+1`，`svectors[b]` 为第 `b` 个键的 Schmidt 谱；`nothing` 表示
#     正则信息未知。所有站点张量右正则且谱正确时 `iscanonical == true`。
#
# 比特 1-based 小端序（站点 `k` ↔ qubit `k-1`，与 QuantumCircuits 一致）。
# `DensityOperatorMPS` 的局域张量携带融合指标 `p = i + 2j`（ket 为低位）。

"""
    AbstractMPS{T<:Number}

MPS 状态类型的抽象基类：纯态 [`MPS`](@ref)（局域维 2）与混合态
[`DensityOperatorMPS`](@ref)（局域维 4）共用同一数据布局
（`data` 站点张量 + `svectors` 键谱）与大多数操作
（索引、拷贝、`canonicalize!`、键维查询等）。
"""
abstract type AbstractMPS{T<:Number} end

# ── 构造校验（具体类型共用） ─────────────────────────────────────────────────

function _check_mps_data(data::Vector{<:AbstractArray{<:Number,3}}, svectors;
                         d::Union{Nothing,Int}=nothing)
    L = length(data)
    L >= 1 || throw(ArgumentError("MPS must have at least one site"))
    for k in 2:L
        size(data[k-1], 3) == size(data[k], 1) ||
            throw(DimensionMismatch("bond dimension mismatch between site $(k-1) and $k"))
    end
    (size(data[1], 1) == 1 && size(data[end], 3) == 1) ||
        throw(ArgumentError("open boundary MPS requires boundary bond dimensions of 1"))
    if d !== nothing
        all(A -> size(A, 2) == d, data) ||
            throw(ArgumentError("physical dimension of each site must be $d"))
    end
    if svectors !== nothing
        length(svectors) == L + 1 ||
            throw(DimensionMismatch("length of svectors must be length of site tensors + 1"))
    end
    return nothing
end

# ── 索引与标量类型 ───────────────────────────────────────────────────────────

"站点数（= 比特数）。"
Base.length(psi::AbstractMPS) = length(psi.data)
Base.getindex(psi::AbstractMPS, k::Int) = psi.data[k]
Base.setindex!(psi::AbstractMPS, A::AbstractArray{<:Number,3}, k::Int) = (psi.data[k] = A; psi)
Base.firstindex(psi::AbstractMPS) = 1
Base.lastindex(psi::AbstractMPS) = length(psi)

function Base.copy(psi::AbstractMPS)
    # 深拷贝：站点张量数组一并复制，保证副本与原态独立（就地演化不串扰）
    sv = psi.svectors === nothing ? nothing : [copy(v) for v in psi.svectors]
    return typeof(psi)([copy(A) for A in psi.data], sv)
end

Base.similar(psi::AbstractMPS) =
    typeof(psi)([similar(A) for A in psi.data],
                psi.svectors === nothing ? nothing : [copy(v) for v in psi.svectors])

function Base.convert(::Type{S}, psi::AbstractMPS) where {T,S<:AbstractMPS{T}}
    eltype(psi) == T && return psi
    R = real(T)
    sv = psi.svectors === nothing ? nothing : [convert(Vector{R}, v) for v in psi.svectors]
    return S([convert(Array{T,3}, A) for A in psi.data], sv)
end

function Base.:(==)(a::AbstractMPS, b::AbstractMPS)
    typeof(a) == typeof(b) || return false
    length(a) == length(b) || return false
    return all(a.data .== b.data) && _svectors_eq(a.svectors, b.svectors)
end
_svectors_eq(a::Nothing, b::Nothing) = true
_svectors_eq(a::Nothing, b) = false
_svectors_eq(a, b::Nothing) = false
_svectors_eq(a, b) = a == b

# ── 形状查询 ─────────────────────────────────────────────────────────────────

"站点 `k` 的物理维（`MPS` 为 2、`DensityOperatorMPS` 为 4）。"
physical_dimension(psi::AbstractMPS, k::Int) = size(psi[k], 2)

"各站点物理维数组。"
physical_dimensions(psi::AbstractMPS) = [size(A, 2) for A in psi.data]

"""
    bond_dimensions(psi) -> Vector{Int}

键维数组（长度 `L+1`，含左右平凡边界）。
"""
bond_dimensions(psi::AbstractMPS) = Int[size(psi[1], 1); [size(A, 3) for A in psi.data]]

"最大键维。"
max_bond_dimension(psi::AbstractMPS) = maximum(bond_dimensions(psi))

"`svectors` 是否未初始化（未知正则信息）。"
svectors_uninitialized(psi::AbstractMPS) = psi.svectors === nothing

# ── 正则性 ───────────────────────────────────────────────────────────────────

"单个站点张量是否右正则（行正交）。"
function isrightcanonical(A::AbstractArray{T,3}; kwargs...) where {T}
    r = zeros(real(T), size(A, 1), size(A, 1))
    @tensor r[-1, -2] := conj(A[-1, 1, 2]) * A[-2, 1, 2]
    return isapprox(r, one(r); kwargs...)
end

"所有站点张量是否右正则。"
isrightcanonical(psi::AbstractMPS; kwargs...) = all(A -> isrightcanonical(A; kwargs...), psi.data)

"""
    iscanonical(psi; kwargs...) -> Bool

psi 是否处于（混合）正则形式：站点 `2…L` 右正则，`svectors` 已初始化
并与各键的真实 Schmidt 谱一致（前缀约化密度矩阵 = diag(Λ²)）。

站点 1 允许携带整体标度（`canonicalize!(; normalize=false)` 的输出即此
形式——对 `DensityOperatorMPS`，迹归一与 HS 范数归一不能同时成立）。
"""
function iscanonical(psi::AbstractMPS; kwargs...)
    length(psi) >= 2 && all(A -> isrightcanonical(A; kwargs...), psi.data[2:end]) ||
        return false
    svectors_uninitialized(psi) && return false
    hold = ones(real(eltype(psi)), 1, 1)
    for i in 1:(length(psi)-1)
        hold = _transfer2(hold, psi[i], psi[i])
        isapprox(hold, Diagonal(psi.svectors[i+1] .^ 2); kwargs...) || return false
    end
    return true
end

"""
    canonicalize!(psi::AbstractMPS; trunc=NoTruncation(), normalize=true) -> Float64

把 `psi` 扫描为混合正则形式（站点右正则 + `svectors` 记录各键 Schmidt 谱）：
先自左向右 QR 等距化（无截断），再自右向左 SVD 扫描捕获谱
（`trunc` 非空时逐键截断，兼作压缩）。返回最大单键截断误差。

`normalize=true` 时归一到融合指标空间的单位范数（对 `DensityOperatorMPS`
即 `sqrt(tr ρ²) = 1`）；迹归一请用 [`normalize_tr!`](@ref)
（对 `DensityOperatorMPS`，`canonicalize!` 默认 `normalize=false` 保持迹不变）。
"""
canonicalize!(psi::AbstractMPS; trunc::TruncationScheme=NoTruncation(), normalize::Bool=true) =
    _canonicalize!(psi; trunc=trunc, normalize=normalize)

function _canonicalize!(psi::AbstractMPS; trunc::TruncationScheme, normalize::Bool)
    L = length(psi)
    T = eltype(psi)
    R = real(T)
    d = size(psi[1], 2)
    # 1) 左扫：QR 等距化（保证右扫各键 SVD 谱为真 Schmidt 谱）
    for k in 1:(L-1)
        A = psi[k]
        Dl = size(A, 1)
        M = reshape(A, Dl * d, :)
        q, r = leftorth!(M, QRpos())
        psi[k] = reshape(q, Dl, d, :)
        A2 = psi[k+1]
        psi[k+1] = reshape(reshape(r, size(r, 1), :) * reshape(A2, size(A2, 1), :),
                           size(r, 1), size(A2, 2), size(A2, 3))
    end
    # 2) 右扫：SVD 捕获 Schmidt 谱
    sv = Vector{Vector{R}}(undef, L + 1)
    sv[1] = [one(R)]
    sv[L+1] = [one(R)]
    err = 0.0
    for k in L:-1:2
        A = psi[k]
        Dl = size(A, 1)
        M = reshape(A, Dl, :)
        u, s, v, e = tsvd!(M; trunc=trunc)
        err = max(err, e)
        psi[k] = reshape(v, Dl, d, :)
        sv[k] = convert(Vector{R}, s)
        # u·s 折入左邻张量
        A2 = psi[k-1]
        psi[k-1] = reshape(reshape(A2, size(A2, 1) * d, :) * (u .* reshape(s, 1, :)),
                           size(A2, 1), d, size(v, 1))
    end
    psi.svectors = sv
    if normalize
        n = norm(psi)
        n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
        psi.data[1] ./= n
        # 态整体缩放 1/n：各键谱同步缩放（边界平凡谱除外）
        for k in 2:L
            psi.svectors[k] ./= n
        end
    end
    return err
end

"""
    LinearAlgebra.normalize!(psi::AbstractMPS) -> psi

把 `psi` 归一化到单位 2-范数（就地）：对纯态即 `⟨ψ|ψ⟩ = 1`，对
`DensityOperatorMPS` 即融合指标空间范数 = `sqrt(tr ρ²) = 1`。
迹归一（`tr ρ = 1`）请用 [`normalize_tr!`](@ref)。
"""
function LinearAlgebra.normalize!(psi::AbstractMPS)
    n = norm(psi)
    n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
    psi.data[1] ./= n
    return psi
end

"""
    LinearAlgebra.norm(psi::AbstractMPS)

态在融合指标空间的 2-范数（双网络全收缩，不依赖正则形式）；
对 `DensityOperatorMPS` 即 `sqrt(tr ρ²)`。
"""
function LinearAlgebra.norm(psi::AbstractMPS)
    E = _ones_env(psi, psi)
    for k in 1:length(psi)
        E = _transfer2(E, psi.data[k], psi.data[k])
    end
    return sqrt(max(real(E[1, 1]), zero(real(E[1, 1]))))
end

# ── 环境收缩原语（双网络）────────────────────────────────────────────────────

_ones_env(::Type{T}, m::Int, n::Int) where {T<:Number} = ones(promote_type(T, Float64), m, n)
_ones_env(a::AbstractArray, b::AbstractArray) = _ones_env(promote_type(eltype(a), eltype(b)), 1, 1)
_ones_env(psiA::AbstractMPS, psiB::AbstractMPS) = _ones_env(promote_type(eltype(psiA), eltype(psiB)), 1, 1)

"`E'[r, s] = Σ conj(A[l', i, r]) E[l', l] B[l, i, s]`（A 为 bra、B 为 ket；输入 E 按左键索引）。"
function _transfer2(E::AbstractMatrix, A::AbstractArray{T,3}, B::AbstractArray{S,3}) where {T,S}
    ET = promote_type(T, S, eltype(E))
    E2 = similar(E, ET, size(A, 3), size(B, 3))
    @tensor E2[r, s] := conj(A[x, i, r]) * E[x, y] * B[y, i, s]
    return E2
end
