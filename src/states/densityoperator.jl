# densityoperator.jl — 混合态密度算符 MPS（DensityOperatorMPS）
#
# 数据布局与公共约定见 `abstractmps.jl`；本文件只补充混合态专属内容：
#   * 每个站点的物理维为 **4**：融合指标 `p = i + 2j`，其中 `i` 为 ket（行）
#     指标、`j` 为 bra（列）指标，即站点张量 `C[l, (i,j), r]` 携带 `ρ` 的
#     局域元素 `ρ_{ij}`；
#   * `N` 体门的局域超算符作用在该融合指标上，指标顺序为逐站点交织
#     （site 1 为最高位），见 `kernels.jl` 的 `_superoperator`；
#   * 站点轴 / 比特序约定与 `MPS` 相同。

"""
    DensityOperatorMPS(data, [svectors]) <: AbstractMPS
    DensityOperatorMPS{T,R}(data, svectors)

混合态密度算符 MPS（局域物理维 4，融合指标 `p = i + 2j`，ket 为低位）。
数据布局与正则形式约定见 [`AbstractMPS`](@ref)；构造时校验键维匹配、
开放边界与局域维。
"""
mutable struct DensityOperatorMPS{T<:Number,R<:Real} <: AbstractMPS{T}
    data::Vector{Array{T,3}}
    svectors::Union{Nothing,Vector{Vector{R}}}

    function DensityOperatorMPS{T,R}(data::Vector{<:AbstractArray{<:Number,3}},
                                     svectors::Union{Nothing,Vector{Vector{R}}}) where {T<:Number,R<:Real}
        R == real(T) || throw(ArgumentError("scalar type for singular vectors must be real(T)"))
        _check_mps_data(data, svectors; d=4)
        new{T,R}(convert(Vector{Array{T,3}}, data), svectors)
    end
end

DensityOperatorMPS(data::Vector{<:AbstractArray{T,3}}, svectors::Union{Nothing,Vector{Vector{R}}}=nothing) where {T<:Number,R<:Real} =
    DensityOperatorMPS{T,real(T)}(data, svectors)

Base.eltype(::Type{DensityOperatorMPS{T,R}}) where {T,R} = T

"""
    canonicalize!(rho::DensityOperatorMPS; trunc=NoTruncation(), normalize=false) -> Float64

把密度算符 MPS（融合指标空间中的普通 MPS）扫描为混合正则规范型。
注意：这里的"范数"是融合空间中张量的范数（= `sqrt(tr ρ²)`），
一般**不等于** `tr ρ`；因此与纯态不同，这里 `normalize` 默认 `false`
（保持迹不变），`normalize=true` 只归一到该意义下的单位范数，
迹归一请用 [`LinearAlgebra.normalize!`](@ref) 对 `DensityOperatorMPS` 的方法。
"""
canonicalize!(rho::DensityOperatorMPS; trunc::TruncationScheme=NoTruncation(), normalize::Bool=false) =
    _canonicalize!(rho; trunc=trunc, normalize=normalize)

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

"""
    normalize_tr!(rho::DensityOperatorMPS) -> rho

把 `rho` 归一化到 `tr ρ = 1`（就地）。注意 `LinearAlgebra.normalize!` 对
`DensityOperatorMPS` 与纯态一致，归一到融合指标空间的单位**范数**；
需要迹归一时请用本函数。
"""
function normalize_tr!(rho::DensityOperatorMPS)
    t = tr(rho)
    abs(t) > 0 || throw(ArgumentError("cannot normalize: trace is zero"))
    # 缩放单个站点张量即可精确归一迹；站点正则性可能被破坏，需要时 canonicalize! 恢复
    rho.data[1] ./= t
    return rho
end

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

"求迹站点转移：T'[x, a] = Σₚ C[l, (p,p), a] T[x, l]（δ 插入；一般开臂秩）。"
function _dtransfer(E::AbstractArray, C::AbstractArray{T,3}) where {T}
    o = size(E)[1:(end-1)]
    C4 = reshape(C, size(C, 1), 4, size(C, 3))
    Cd = C4[:, 1, :] + C4[:, 4, :]                             # δ 插入：(0,0)+(1,1)
    E2 = reshape(E, prod(o), size(E)[end]) * Cd
    return reshape(E2, o..., size(C, 3))
end
