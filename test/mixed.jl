# test_mixed.jl — 混合态（DensityOperatorMPS，局域维 4）

@testset "mixed states" begin
    n = 3

    # 零混合态 / 基态混合态
    rho0 = densitymatrix_mps(n)
    @test tr(rho0) ≈ 1 atol = 1e-12
    @test densitymatrix(rho0) ≈ VQC.storage(VQC.DensityMatrix(ComplexF64, n)) atol = 1e-12

    # 纯态 → 混合态转换：与 VQC 的 DensityMatrix 一致
    psi = rand_state_mps(n; D=4)
    rho = densitymatrix(psi)
    @test tr(rho) ≈ 1 atol = 1e-10
    s = statevector(psi)
    @test densitymatrix(rho) ≈ VQC.storage(VQC.DensityMatrix(s * s', n)) atol = 1e-10
    # 纯度
    @test real(dot(rho, rho)) ≈ 1 atol = 1e-8

    # 门作用与 VQC 一致
    c = Circuit(n)
    push!(c, H(1))
    push!(c, CX(1, 2))
    push!(c, CZ(2, 3))       # 非相邻
    push!(c, RY(0.5, 3))
    ref_dm = simulate(c, VQC.DensityMatrix(VQC.zero_state(n)))
    rho2 = simulate(c, densitymatrix_mps(n))
    @test densitymatrix(rho2) ≈ VQC.storage(ref_dm) atol = 1e-10
    @test tr(rho2) ≈ 1 atol = 1e-10

    # Kraus 信道：振幅阻尼（纯态自动提升）
    γ = 0.2
    k0 = [1 0; 0 sqrt(1 - γ)]
    k1 = [0 sqrt(γ); 0 0]
    ch = KrausChannel([k0, k1])
    cc = Circuit(n)
    push!(cc, H(1))
    push!(cc, QuantumCircuits.KrausOp(ch, 2))
    ref_k = simulate(cc, VQC.DensityMatrix(VQC.zero_state(n)))
    out = simulate(cc, zero_state_mps(n))
    @test out isa DensityOperatorMPS
    @test densitymatrix(out) ≈ VQC.storage(ref_k) atol = 1e-10
    @test tr(out) ≈ 1 atol = 1e-10

    # 混合态上的信道 + 门混合线路
    noisy = Circuit(n)
    push!(noisy, H(1))
    push!(noisy, CX(1, 2))
    push!(noisy, Depolarizing([2], 0.1))
    push!(noisy, AmplitudeDamping(3, 0.15))
    ref_n = simulate(noisy, VQC.DensityMatrix(VQC.zero_state(n)))
    outn = simulate(noisy, densitymatrix_mps(n))
    @test densitymatrix(outn) ≈ VQC.storage(ref_n) atol = 1e-10

    # 保真度：ψ-ρ 与 VQC 一致
    sv = statevector(psi)
    @test fidelity(rho, psi) ≈ VQC.fidelity(VQC.DensityMatrix(sv * sv'), VQC.StateVector(sv, n)) atol = 1e-8

    # reset_qubit_zero!（混合态）
    r1 = densitymatrix(onehot_encoding_mps([1, 1]))
    reset_qubit_zero!(r1, 1)
    @test densitymatrix(r1) ≈ VQC.storage(VQC.DensityMatrix(VQC.onehot_encoding([0, 1]))) atol = 1e-10
end
