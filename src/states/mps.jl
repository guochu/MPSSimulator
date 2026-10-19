# mps.jl — 态层：FiniteMPSAlgorithms.CanonicalMPS 的薄封装
#
# 纯态 `MPS`（局域维 2）与混合态 `DensityOperatorMPS`（局域维 4，融合指标
# `p = i + 2j`，ket 为低位）都是 `CanonicalMPS` 的简单包装：站点张量存储、
# 键 Schmidt 谱、正交化（`canonicalize!`）、相邻内容交换（`swap!`）、
# 两体门作用与线性代数（`dot` / `norm` / `distance`）全部复用
# FiniteMPSAlgorithms 的现成实现。
#
# 封装层不变量：
#   * `scaling(psi.core) == 1`——外部尺度始终存放在站点张量数据中；
#     FiniteMPSAlgorithms 的正交化内核会把范数暂时吸进 `scaling` 字段，
#     每次内核调用后用 `_sync!` 折回数据；
#   * `psi.core.s[b+1]` 为键 `b` 的 Schmidt 谱（数据网络意义下），
#     `missing` 表示正则信息未知。
#
# 比特 1-based 小端序（站点 `k` ↔ qubit `k`）；站点张量轴序 = (左键, 物理维, 右键)，
# 开放边界。

# ── 封装类型 ─────────────────────────────────────────────────────────────────

"""
    AbstractMPS{T<:Number}

MPS 状态类型的抽象基类：纯态 [`MPS`](@ref)（局域维 2）与混合态
[`DensityOperatorMPS`](@ref)（局域维 4）共用同一封装布局
（内部为 `FiniteMPSAlgorithms.CanonicalMPS`）与大多数操作
（索引、拷贝、`canonicalize!`、键维查询等）。
"""
abstract type AbstractMPS{T<:Number} end

"""
    MPS(core::CanonicalMPS)
    MPS(data, [svectors])
    MPS{T,R}(core)

纯态矩阵乘积态（局域物理维 2），`FiniteMPSAlgorithms.CanonicalMPS` 的薄封装。
数据布局与正则形式约定见 [`AbstractMPS`](@ref)；构造时校验键维匹配、
开放边界与局域维。
"""
struct MPS{T<:Number,R<:Real} <: AbstractMPS{T}
    core::CanonicalMPS{T,R}
end

function MPS(core::CanonicalMPS{T,R}) where {T,R}
    R == real(T) || throw(ArgumentError("scalar type of Schmidt values must be real(T)"))
    all(A -> size(A, 2) == 2, core.data) ||
        throw(ArgumentError("physical dimension of each site must be 2"))
    return MPS{T,R}(core)
end

function MPS(data::AbstractVector{<:AbstractArray{T,3}},
             svectors::Union{Nothing,AbstractVector}=nothing) where {T<:Number}
    dat = convert(Vector{Array{T,3}}, collect(data))
    return MPS(svectors === nothing ? CanonicalMPS(dat) : CanonicalMPS(dat, svectors))
end

"""
    DensityOperatorMPS(core::CanonicalMPS)
    DensityOperatorMPS(data, [svectors])
    DensityOperatorMPS{T,R}(core)

混合态密度算符 MPS（局域物理维 4，融合指标 `p = i + 2j`，ket 为低位），
`FiniteMPSAlgorithms.CanonicalMPS` 的薄封装。数据布局与正则形式约定见
[`AbstractMPS`](@ref)；构造时校验键维匹配、开放边界与局域维。
"""
struct DensityOperatorMPS{T<:Number,R<:Real} <: AbstractMPS{T}
    core::CanonicalMPS{T,R}
end

function DensityOperatorMPS(core::CanonicalMPS{T,R}) where {T,R}
    R == real(T) || throw(ArgumentError("scalar type of Schmidt values must be real(T)"))
    all(A -> size(A, 2) == 4, core.data) ||
        throw(ArgumentError("physical dimension of each site must be 4"))
    return DensityOperatorMPS{T,R}(core)
end

function DensityOperatorMPS(data::AbstractVector{<:AbstractArray{T,3}},
                            svectors::Union{Nothing,AbstractVector}=nothing) where {T<:Number}
    dat = convert(Vector{Array{T,3}}, collect(data))
    return DensityOperatorMPS(svectors === nothing ? CanonicalMPS(dat) : CanonicalMPS(dat, svectors))
end

# ── 索引与标量类型 ───────────────────────────────────────────────────────────

"站点数（= 比特数）。"
Base.length(psi::AbstractMPS) = length(psi.core)
Base.getindex(psi::AbstractMPS, k::Int) = psi.core[k]
Base.setindex!(psi::AbstractMPS, A::AbstractArray{<:Number,3}, k::Int) = (psi.core[k] = A; psi)
Base.firstindex(psi::AbstractMPS) = 1
Base.lastindex(psi::AbstractMPS) = length(psi)

"底层 `CanonicalMPS`（FiniteMPSAlgorithms 数据结构）。"
core(psi::AbstractMPS) = psi.core

Base.eltype(::Type{<:AbstractMPS{T}}) where {T} = T

_svec_copy(s) = ismissing(s) ? missing : copy(s)

function Base.copy(psi::AbstractMPS)
    # 深拷贝：CanonicalMPS.copy 已复制站点张量与键谱，保证副本与原态独立
    return typeof(psi)(copy(psi.core))
end

function Base.similar(psi::AbstractMPS)
    return typeof(psi)(CanonicalMPS([similar(A) for A in psi.core.data],
                                    [_svec_copy(s) for s in psi.core.s],
                                    Ref(scaling(psi.core))))
end

function Base.convert(::Type{S}, psi::AbstractMPS) where {T,S<:AbstractMPS{T}}
    eltype(psi) == T && return psi
    R = real(T)
    dat = [convert(Array{T,3}, A) for A in psi.core.data]
    s = [_svec_copy(sv) === missing ? missing : convert(Vector{R}, sv) for sv in psi.core.s]
    # S 为抽象类型（如 `AbstractMPS{T}`）时落到具体封装类型
    Target = isabstracttype(S) ? typeof(psi).name.wrapper : S
    return Target(CanonicalMPS(dat, s, Ref(scaling(psi.core))))
end

function Base.:(==)(a::AbstractMPS, b::AbstractMPS)
    typeof(a) == typeof(b) || return false
    length(a) == length(b) || return false
    a.core.data == b.core.data || return false
    return all(_svec_eq(a.core.s[i], b.core.s[i]) for i in eachindex(a.core.s))
end
_svec_eq(a::Missing, b::Missing) = true
_svec_eq(a::Missing, b) = false
_svec_eq(a, b::Missing) = false
_svec_eq(a, b) = a == b

# ── 形状查询（与 FiniteMPSAlgorithms 对齐：名字、语义与泛型方法一致）─────────
# 封装态转发到 core（CanonicalMPS），phydim(psi, k) 直接走站点张量：
#   bonddims(ψ)：内部键维数组（长度 L-1）  bonddim(ψ)：最大键维
#   bonddim(ψ, b)：键 b 的维数            phydims(ψ)：各站点物理维数组

"站点 `k` 的物理维（`MPS` 为 2、`DensityOperatorMPS` 为 4）。"
phydim(psi::AbstractMPS, k::Int) = phydim(psi[k])

bonddim(psi::AbstractMPS, b::Int) = bonddim(psi.core, b)
bonddims(psi::AbstractMPS) = bonddims(psi.core)
bonddim(psi::AbstractMPS) = bonddim(psi.core)
phydims(psi::AbstractMPS) = phydims(psi.core)

"`svectors` 是否未初始化（未知正则信息）。"
svectors_uninitialized(psi::AbstractMPS) = svectors_uninitialized(psi.core)

# ── 正则性 ───────────────────────────────────────────────────────────────────

"所有站点张量是否右正则。"
isrightcanonical(psi::AbstractMPS; kwargs...) =
    all(A -> isrightcanonical(A; kwargs...), psi.core.data)

"""
    iscanonical(psi; atol=1e-8) -> Bool

psi 是否处于（混合）正则形式：站点 `2…L` 右正则，`svectors` 已初始化
并与各键的真实 Schmidt 谱一致（前缀约化密度矩阵 = diag(Λ²)）。

站点 1 允许携带整体标度（`canonicalize!(; normalize=false)` 的输出即此
形式——对 `DensityOperatorMPS`，迹归一与 HS 范数归一不能同时成立）。
"""
function iscanonical(psi::AbstractMPS; atol::Real=1.0e-8)
    length(psi) >= 2 && all(A -> isrightcanonical(A; atol), view(psi.core.data, 2:length(psi))) ||
        return false
    svectors_uninitialized(psi) && return false
    hold = ones(real(eltype(psi)), 1, 1)
    for i in 1:(length(psi)-1)
        hold = _transfer2(hold, psi[i], psi[i])
        isapprox(hold, Diagonal(psi.core.s[i+1] .^ 2); atol) || return false
    end
    return true
end

# ── 封装层缩放与正则化 ───────────────────────────────────────────────────────

"把 core 暂存的 per-site scaling 折回站点数据（恢复 scaling ≡ 1 不变量）。"
function _sync!(psi::AbstractMPS)
    c = scaling(psi.core)^length(psi.core)
    isone(c) && return psi
    psi.core[1] = psi.core[1] * c
    for i in 2:length(psi)          # 内部键谱：s[2:L]
        s = psi.core.s[i]
        ismissing(s) || (psi.core.s[i] = c .* s)
    end
    setscaling!(psi.core, 1.0)
    return psi
end

"整体缩放数据网络（数据除以 `f`、键谱除以 `|f|`，保持 s = 数据的真实谱）。"
function _rescale_data!(psi::AbstractMPS, f::Number)
    core = psi.core
    core[1] = core[1] / f
    g = abs(f)
    for i in 2:length(core)         # 内部键谱：s[2:L]
        s = core.s[i]
        ismissing(s) || (core.s[i] = s ./ g)
    end
    return psi
end

function _canonicalize!(psi::AbstractMPS; trunc::TruncationScheme, normalize::Bool)
    _, err = canonicalize!(psi.core; alg=Orthogonalize(SVD(), trunc, false))
    _sync!(psi)
    if normalize
        n = norm(psi)
        n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
        _rescale_data!(psi, n)
    end
    return err
end

"""
    canonicalize!(psi::MPS; trunc=NoTruncation(), normalize=true) -> Float64

把 `psi` 扫描为混合正则形式（站点右正则 + `svectors` 记录各键 Schmidt 谱），
复用 `FiniteMPSAlgorithms.canonicalize!`：先自左向右 QR 等距化（无截断），
再自右向左 SVD 扫描捕获谱（`trunc` 非空时逐键截断，兼作压缩）。
返回最大单键截断误差。

`normalize=true` 时归一到融合指标空间的单位范数（对 `DensityOperatorMPS`
即 `sqrt(tr ρ²) = 1`）；迹归一请用 [`normalize_tr!`](@ref)
（对 `DensityOperatorMPS`，`canonicalize!` 默认 `normalize=false` 保持迹不变）。
"""
canonicalize!(psi::MPS; trunc::TruncationScheme=NoTruncation(), normalize::Bool=true) =
    _canonicalize!(psi; trunc=trunc, normalize=normalize)

canonicalize!(rho::DensityOperatorMPS; trunc::TruncationScheme=NoTruncation(), normalize::Bool=false) =
    _canonicalize!(rho; trunc=trunc, normalize=normalize)

"""
    LinearAlgebra.normalize!(psi::AbstractMPS) -> psi

把 `psi` 归一化到单位 2-范数（就地）：对纯态即 `⟨ψ|ψ⟩ = 1`，对
`DensityOperatorMPS` 即融合指标空间范数 = `sqrt(tr ρ²) = 1`。
迹归一（`tr ρ = 1`）请用 [`normalize_tr!`](@ref)。
"""
function LinearAlgebra.normalize!(psi::AbstractMPS)
    n = norm(psi)
    n > 0 || throw(ArgumentError("cannot normalize a zero MPS"))
    return _rescale_data!(psi, n)
end

"""
    normalize_tr!(rho::DensityOperatorMPS) -> rho

把 `rho` 归一化到 `tr ρ = 1`（就地）。注意 `LinearAlgebra.normalize!` 对
`DensityOperatorMPS` 与纯态一致，归一到融合指标空间的单位**范数**；
需要迹归一时请用本函数。
"""
function normalize_tr!(rho::DensityOperatorMPS)
    t = tr(rho)
    abs(t) > 0 || throw(ArgumentError("cannot normalize: trace is zero"))
    return _rescale_data!(rho, t)
end

"""
    LinearAlgebra.norm(psi::AbstractMPS)

态在融合指标空间的 2-范数（双网络全收缩，不依赖正则形式）；
对 `DensityOperatorMPS` 即 `sqrt(tr ρ²)`。复用 `FiniteMPSAlgorithms.norm`。
"""
LinearAlgebra.norm(psi::AbstractMPS) = norm(psi.core)

"""
    LinearAlgebra.dot(a::MPS, b::MPS)

重叠 `⟨a|b⟩`（全收缩，无截断）。复用 `FiniteMPSAlgorithms.dot`。
"""
LinearAlgebra.dot(a::MPS, b::MPS) = dot(a.core, b.core)

"""
    LinearAlgebra.dot(rho::DensityOperatorMPS, sigma::DensityOperatorMPS)

Hilbert-Schmidt 内积 `tr(ρ σ)`。
"""
function LinearAlgebra.dot(rho::DensityOperatorMPS, sigma::DensityOperatorMPS)
    length(rho) == length(sigma) || throw(DimensionMismatch("MPS length mismatch"))
    E = ones(promote_type(eltype(rho), eltype(sigma), Float64), 1, 1)
    for k in 1:length(rho)
        # tr(ρσ)：ρ 的 bra 与 σ 的 ket 配对、ρ 的 ket 与 σ 的 bra 配对
        Cl = reshape(rho[k], size(rho[k], 1), 2, 2, :)
        Cr = reshape(sigma[k], size(sigma[k], 1), 2, 2, :)
        E2 = similar(E, promote_type(eltype(E), eltype(Cl), eltype(Cr)), size(Cl, 4), size(Cr, 4))
        @tensor E2[x, y] := Cl[l, a, b, x] * E[l, m] * Cr[m, b, a, y]
        E = E2
    end
    return E[1, 1]
end

"""
    LinearAlgebra.tr(rho::DensityOperatorMPS)

`tr ρ`（δ 插入收缩）。
"""
function LinearAlgebra.tr(rho::DensityOperatorMPS)
    E = ones(promote_type(eltype(rho), Float64), 1)
    for k in 1:length(rho)
        E = _dtransfer(E, rho[k])
    end
    return E[1]
end

distance2(a::MPS, b::MPS) = distance2(a.core, b.core)
distance(a::MPS, b::MPS) = distance(a.core, b.core)

Base.isapprox(a::MPS, b::MPS; kwargs...) =
    length(a) == length(b) && distance2(a, b) <= (get(kwargs, :atol, 0) + get(kwargs, :rtol, 0) * max(norm(a), norm(b)))^2

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

"求迹站点转移：T'[x, a] = Σₚ C[l, (p,p), a] T[x, l]（δ 插入；一般开臂秩）。"
function _dtransfer(E::AbstractArray, C::AbstractArray{T,3}) where {T}
    o = size(E)[1:(end-1)]
    C4 = reshape(C, size(C, 1), 4, size(C, 3))
    Cd = C4[:, 1, :] + C4[:, 4, :]                             # δ 插入：(0,0)+(1,1)
    E2 = reshape(E, prod(o), size(E)[end]) * Cd
    return reshape(E2, o..., size(C, 3))
end
