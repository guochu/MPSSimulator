# test_apply.jl — 纯态门作用（与 VQC 交叉验证）+ 正则形式保持

@testset "apply (pure)" begin
    n = 4
    c = Circuit(n)
    push!(c, H(1))
    push!(c, CX(1, 2))
    push!(c, CZ(2, 4))          # 非相邻门
    push!(c, RZZ(0.3, 1, 3))    # 非相邻参数门
    push!(c, RX(0.7, 4))
    push!(c, SWAP(1, 4))
    push!(c, CCX(1, 2, 3))      # 三比特门

    ref = simulate(c, VQC.zero_state(n))
    mps = simulate(c, zero_state_mps(n))
    @test statevector(mps) ≈ VQC.storage(ref) atol = 1e-10

    # 核心层接口：矩阵 + 位置（positions[1] = 矩阵最高位）
    psi = zero_state_mps(3)
    apply!(psi, mat(CX), (2, 3))              # q2 控制、q3 目标（矩阵 MSB-first）
    ref2 = VQC.apply!(VQC.zero_state(3), mat(CX), (2, 3))
    @test statevector(psi) ≈ VQC.storage(ref2) atol = 1e-12

    # 非就地 apply 与类型提升
    psi_r = onehot_encoding_mps(Float64, [1])
    psi_c = apply(psi_r, Matrix{ComplexF64}(mat(H)), (1,))
    @test eltype(psi_c) == ComplexF64
    ref1 = VQC.onehot_encoding([1])
    VQC.apply!(ref1, mat(H), (1,))
    @test statevector(psi_c) ≈ VQC.storage(ref1) atol = 1e-12

    # ── 幺正演化保持正则形式（逐步检查）──
    Random.seed!(11)
    cvar = variational_circuit_1d(4, 2; θs = rand(36) .* 2π)
    mps2 = zero_state_mps(4)
    @test iscanonical(mps2)
    for op in cvar.ops
        apply!(mps2, op; trunc=NoTruncation())
        @test iscanonical(mps2; atol = 1e-8)
    end
    exact = mps2
    @test statevector(exact) ≈ VQC.storage(simulate(cvar, VQC.zero_state(4))) atol = 1e-10

    # ── 3 比特门（非相邻 + 乱序 positions）与 VQC 稠密后端一致 ──
    using LinearAlgebra
    Random.seed!(21)
    c3 = Circuit(5)
    push!(c3, H(1)); push!(c3, H(3)); push!(c3, X(5))
    push!(c3, ctrl(ctrl(X(5), 1), 3))                  # Toffoli（双控制 X）
    q8 = qr(randn(ComplexF64, 8, 8)); U8 = Matrix(q8.Q)
    push!(c3, usergate(:G3, U8)(1, 3, 5))              # 非相邻 3 比特任意酉
    ψ3g = simulate(c3, zero_state_mps(5))
    ref3g = simulate(c3, VQC.zero_state(5))
    @test statevector(ψ3g) ≈ VQC.storage(ref3g) atol = 1e-10
    @test iscanonical(ψ3g)
    # 核心层直接接口：8×8 矩阵 + 乱序 positions
    ψ3c = apply(zero_state_mps(5), U8, (5, 3, 1))
    ref3c = VQC.apply!(VQC.zero_state(5), U8, (5, 3, 1))
    @test statevector(ψ3c) ≈ VQC.storage(ref3c) atol = 1e-10
    @test iscanonical(ψ3c)

    # ── 随机多比特幺正门（1–5 比特，位置多样化）与 VQC 稠密后端一致 ──
    # 初态：同一组随机直积态（qubit_encoding）；门：Haar 随机酉矩阵；
    # positions 覆盖升序相邻 / 高位相邻 / 逆序 / 随机乱序（含非相邻）。
    rrng2 = Random.MersenneTwister(123)
    nq = 8
    for k in 1:5
        dk = 1 << k
        qk = qr(randn(rrng2, ComplexF64, dk, dk))
        Uk = Matrix(qk.Q)
        pos_sets = [collect(1:k),                      # 低位升序相邻
                    collect((nq - k + 1):nq),          # 高位相邻
                    reverse(collect(1:k))]             # 逆序（重排快速路径）
        for _ in 1:2
            pk = randperm(rrng2, nq)[1:k]              # 随机乱序（可含非相邻）
            rand(rrng2) < 0.5 && sort!(pk)
            push!(pos_sets, pk)
        end
        for positions in pos_sets
            pt = Tuple(positions)
            θs = rand(rrng2, nq) .* 2                  # 直积态初态（同一组 θs 两侧共用）
            psi_k = qubit_encoding_mps(θs)
            apply!(psi_k, Uk, pt)
            ref_k = VQC.qubit_encoding(θs)
            VQC.apply!(ref_k, Uk, pt)
            @test statevector(psi_k) ≈ VQC.storage(ref_k) atol = 1e-9
            @test iscanonical(psi_k)
        end
    end

    # 电路层：5 比特随机酉 usergate（乱序非相邻位置）
    q32 = qr(randn(rrng2, ComplexF64, 32, 32)); U5 = Matrix(q32.Q)
    p5 = Tuple(randperm(rrng2, nq)[1:5])
    θs5 = rand(rrng2, nq) .* 2
    c5 = Circuit(nq)
    push!(c5, usergate(:G5, U5)(p5...))
    ref5 = simulate(c5, VQC.qubit_encoding(θs5))
    ψ5 = simulate(c5, qubit_encoding_mps(θs5))
    @test statevector(ψ5) ≈ VQC.storage(ref5) atol = 1e-9
    @test iscanonical(ψ5)

    # 截断：键维封顶 + 近似保真
    trunced = simulate(cvar, zero_state_mps(4); trunc=truncdim(D=2))
    @test max_bond_dimension(trunced) <= 2
    @test fidelity(trunced, exact) > 0.99

    # 截断误差可控：TruncateDim(1) 会丢弃纠缠 → 保真度明显下降
    hard = simulate(cvar, zero_state_mps(4); trunc=truncdim(D=1))
    @test fidelity(hard, exact) < 0.99

    # ── 随机线路（含非相邻两比特门）逐门正则性检查 ──
    # 固化 debug/canonical_form.jl 的纯态行为：随机酉门（含 SWAP 聚集路径）
    # 全程保持混合正则形式，终态与 VQC 稠密后端一致。
    # 注意：@testset 内定义的函数会把函数体中对“与外层 testset 局部变量同名”
    # 的赋值解析为对外层变量的重绑定，故内层线路变量须避开外层已用名（c）。
    function random_canonical_circuit(rng::AbstractRNG, n::Int, depth::Int)
        ckt = Circuit(n)
        for _ in 1:depth
            for q in 1:n
                rand(rng) < 0.5 ? push!(ckt, H(q)) : push!(ckt, RX(2π * rand(rng), q))
            end
            for _ in 1:n
                a, b = rand(rng, 1:n, 2)
                a == b && continue
                rand(rng) < 0.5 ? push!(ckt, CZ(a, b)) : push!(ckt, RZZ(2π * rand(rng), a, b))
            end
        end
        return ckt
    end
    rrng = Random.MersenneTwister(42)
    crc = random_canonical_circuit(rrng, 5, 3)
    ψr = zero_state_mps(5)
    canon_flags = Bool[]
    for op in crc.ops
        op isa BarrierOp && continue
        apply!(ψr, op)
        push!(canon_flags, iscanonical(ψr))
    end
    @test all(canon_flags)
    @test length(canon_flags) == length(crc.ops)
    @test statevector(ψr) ≈ VQC.storage(simulate(crc, VQC.zero_state(5))) atol = 1e-10

    # canonicalize! 不改变态
    psi3 = simulate(c, zero_state_mps(n))
    before = statevector(psi3)
    err = canonicalize!(psi3)
    @test err ≈ 0 atol = 1e-12
    @test statevector(psi3) ≈ before atol = 1e-10
    @test iscanonical(psi3; atol = 1e-8)

    # 非幺正算子破坏正则性，但状态保持精确；canonicalize! 可恢复
    pn = onehot_encoding_mps([1, 1])
    @test iscanonical(pn)
    apply!(pn, [1.0 1.0; 1.0 1.0], (1,))       # 非幺正（行不正交）
    @test !iscanonical(pn)
    normalize!(pn)
    @test statevector(pn) ≈ [0, 0, 1, 1] ./ sqrt(2) atol = 1e-10   # (|01⟩+|11⟩)/√2
    canonicalize!(pn)
    @test iscanonical(pn; atol = 1e-8)

    # reset_qubit_zero!：|11⟩ 重置 qubit 1 → |01⟩（小端序下振幅索引为 2）
    p4 = onehot_encoding_mps([1, 1])
    reset_qubit_zero!(p4, 1)
    @test statevector(p4) ≈ VQC.storage(VQC.onehot_encoding([0, 1])) atol = 1e-12
end
