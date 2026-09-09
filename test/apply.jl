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

    # 截断：键维封顶 + 近似保真
    trunced = simulate(cvar, zero_state_mps(4); trunc=truncdim(D=2))
    @test max_bond_dimension(trunced) <= 2
    @test fidelity(trunced, exact) > 0.99

    # 截断误差可控：TruncateDim(1) 会丢弃纠缠 → 保真度明显下降
    hard = simulate(cvar, zero_state_mps(4); trunc=truncdim(D=1))
    @test fidelity(hard, exact) < 0.99

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
