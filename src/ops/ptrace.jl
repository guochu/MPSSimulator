# ptrace.jl — 偏迹（约化密度矩阵）
#
# 纯态走双网络开臂收缩；混合态走单网络（融合指标）开臂收缩。
# 开臂（保留）比特按升序重编号；输出矩阵行 = ket、列 = bra（小端序），
# 行/列的第 t 位对应第 t 个保留比特。

"""
    partial_tr(psi::MPS, sites) -> Matrix

对 `sites`（1-based）求偏迹，返回保留比特的稠密约化密度矩阵
（保留比特按升序重编号）。空 `sites` 返回完整密度矩阵（指数级）；
全比特求迹返回标量 `⟨ψ|ψ⟩`。
"""
function partial_tr(psi::MPS, sites::AbstractVector{Int})
    n = length(psi)
    _check_ptrace_sites(sites, n)
    isempty(sites) && return _pure_densitymatrix(psi)
    (length(sites) == n) && return real(dot(psi, psi))
    T, opens, ntail = _rdm_sweep(psi, collect(sites))
    return _collapse_rdm(T, opens, ntail)
end

partial_tr(psi::MPS, sites::Int...) = partial_tr(psi, collect(sites))

"""
    partial_tr(rho::DensityOperatorMPS, sites) -> Matrix

对 `sites` 求偏迹，返回保留比特的稠密约化密度矩阵。
空 `sites` 返回完整密度矩阵；全比特求迹返回 `tr ρ`。
"""
function partial_tr(rho::DensityOperatorMPS, sites::AbstractVector{Int})
    n = length(rho)
    _check_ptrace_sites(sites, n)
    isempty(sites) && return densitymatrix(rho)
    (length(sites) == n) && return tr(rho)
    T, opens, ntail = _rdm_sweep(rho, collect(sites))
    return _collapse_rdm(T, opens, ntail)
end

partial_tr(rho::DensityOperatorMPS, sites::Int...) = partial_tr(rho, collect(sites))

function _check_ptrace_sites(sites, n::Int)
    length(unique(sites)) == length(sites) || throw(ArgumentError("duplicate sites not allowed"))
    all(q -> 1 <= q <= n, sites) || throw(ArgumentError("site out of range [1, $n]"))
    return nothing
end

# ── 收缩张量 → 稠密 RDM ──────────────────────────────────────────────────────
#
# T 的开臂轴按 (ket₁, bra₁, ket₂, bra₂, …) 交错排列（前导 1 维边界轴
# 不影响展平序）；尾部 `ntail` 个键轴与（1-维）右边界收缩后，把
# 交织展平序 (k₁ + 2b₁ + 4k₂ + …) 重排为标准 (ket…, bra…) 位序矩阵。

function _collapse_rdm(T::AbstractArray, opens::Int, ntail::Int)
    n = ndims(T)
    idx = ntuple(i -> i <= n - ntail ? Colon() : 1, n)
    body = vec(T[idx...])                        # flat = Σ (kₜ + 2·bₜ)·4^(t-1)
    M = zeros(ComplexF64, 1 << opens, 1 << opens)
    for flat in 0:(1 << (2*opens))-1
        i = sum(((flat >> (2*(t-1))) & 1) << (t-1) for t in 1:opens)
        j = sum(((flat >> (2*(t-1)+1)) & 1) << (t-1) for t in 1:opens)
        M[i+1, j+1] += body[flat+1]
    end
    return M
end

# ── 纯态：双网络开臂扫描 ─────────────────────────────────────────────────────
#
# T 形状 (open…(ket/bra 交错), Dket', Dbra')；求迹站点走双网络转移。

function _rdm_sweep(psi::MPS, traced::Vector{Int})
    T = ones(promote_type(eltype(psi), Float64), 1, 1)   # (bra-l, ket-l) 边界
    opens = 0
    ntail = 2
    for k in 1:length(psi)
        A = psi[k]
        if k in traced
            T = _open_transfer(T, A)
        else
            T = _open_append(T, A)
            opens += 1
        end
    end
    return T, opens, ntail
end

"求迹站点转移：T' = Σᵢ conj(A[bra,i,rb]) T[bra,ket] A[ket,i,rk]。"
function _open_transfer(X::AbstractArray, A::AbstractArray{T,3}) where {T}
    o = size(X)[1:(end-2)]
    T3 = reshape(X, prod(o), size(X)[end-1], size(X)[end])
    out = similar(T3, promote_type(eltype(T3), T), prod(o), size(A, 3), size(A, 3))
    @tensor out[x, c, d] := T3[x, a, b] * conj(A[a, i, c]) * A[b, i, d]
    return reshape(out, o..., size(A, 3), size(A, 3))
end

"保留站点追加一对开臂：T'[o…, ket, bra, rb, rk]。"
function _open_append(X::AbstractArray, A::AbstractArray{T,3}) where {T}
    o = size(X)[1:(end-2)]
    T3 = reshape(X, prod(o), size(X)[end-1], size(X)[end])
    out = similar(T3, promote_type(eltype(T3), T), prod(o), 2, 2, size(A, 3), size(A, 3))
    @tensor out[x, f, e, c, d] := T3[x, a, b] * conj(A[a, e, c]) * A[b, f, d]
    return reshape(out, o..., 2, 2, size(A, 3), size(A, 3))
end

# ── 混合态：单网络开臂扫描 ───────────────────────────────────────────────────
#
# T 形状 (open…(ket/bra 交错), D)；求迹站点走 δ 插入转移（_dtransfer）。

function _rdm_sweep(rho::DensityOperatorMPS, traced::Vector{Int})
    T = ones(promote_type(eltype(rho), Float64), 1)
    opens = 0
    ntail = 1
    for k in 1:length(rho)
        C = rho[k]
        if k in traced
            T = _dtransfer(T, C)
        else
            T = _dopen_append(T, C)
            opens += 1
        end
    end
    return T, opens, ntail
end

"保留站点追加融合物理对：T'[o…, i, j, r] = Σ_l T[o…, l] C[l, (i,j), r]。"
function _dopen_append(X::AbstractArray, C::AbstractArray{T,3}) where {T}
    o = size(X)[1:(end-1)]
    T2 = reshape(X, prod(o), size(X)[end])
    out = reshape(T2 * reshape(C, size(C, 1), :), o..., 2, 2, size(C, 3))
    return out
end

# ── 纯态完整密度矩阵 ─────────────────────────────────────────────────────────

"纯态完整稠密密度矩阵 `|ψ⟩⟨ψ|`。"
function _pure_densitymatrix(psi::MPS)
    v = statevector(psi)
    return v * v'
end
