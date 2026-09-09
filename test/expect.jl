# test_expect.jl — 期望值 / 保真度 / overlap

@testset "expectation" begin
    using QuantumCircuits.Hamiltonian
    n = 4
    Random.seed!(5)
    c = variational_circuit_1d(n, 2; θs = rand(36) .* 2π)
    psi = simulate(c, zero_state_mps(n))
    ref = simulate(c, VQC.zero_state(n))

    # Pauli 项 / Pauli 和
    h = PauliSum([PauliTerm(1.0, 1 => :Z, 2 => :Z),
                  PauliTerm(0.5, 2 => :X),
                  PauliTerm(0.3im, 3 => :Y, 4 => :Z)])
    @test expectation(h, psi) ≈ VQC.expectation(h, ref) atol = 1e-10

    # 局域算符链：对照稠密直积（site 1 为最低位 = 最后一个 kron 因子）
    ops = [1 => ComplexF64[0 1; 1 0], 3 => ComplexF64[1 0; 0 -1]]
    od = Dict(ops)
    M = reduce(kron, [get(od, k, Matrix{ComplexF64}(I, 2, 2)) for k in n:-1:1])
    @test expectation(ops, psi) ≈ VQC.expectation(M, ref) atol = 1e-10

    # ⟨ψA|O|ψB⟩：对照稠密矩阵元
    psiB = rand_state_mps(n; D=4)
    val = expectation(ops, psi, psiB)
    svA, svB = statevector(psi), statevector(psiB)
    @test val ≈ dot(svA, M, svB) atol = 1e-10

    # GateOp 可观测
    op = RZZ(0.7, 2, 3)
    @test expectation(op, psi) ≈ VQC.expectation(op, ref) atol = 1e-10

    # 混合态：tr(ρ O) 与 ⟨ψ|ρ|ψ⟩
    rho = densitymatrix(psi)
    @test expectation(h, rho) ≈ VQC.expectation(h, VQC.DensityMatrix(ref)) atol = 1e-10
    @test expectation(op, rho) ≈ VQC.expectation(op, VQC.DensityMatrix(ref)) atol = 1e-10
    @test fidelity(rho, psi) ≈ 1 atol = 1e-8
    @test tr(rho) ≈ 1 atol = 1e-10

    # Hilbert-Schmidt overlap 与稠密一致
    psi2 = rand_state_mps(n; D=4)
    rho2 = densitymatrix(psi2)
    @test dot(rho, rho2) ≈ tr(densitymatrix(rho) * densitymatrix(rho2)) atol = 1e-8
    rf = VQC.storage(ref)
    @test fidelity(rho, psi2) ≈ VQC.fidelity(VQC.DensityMatrix(rf * rf'),
                                             VQC.StateVector(statevector(psi2), n)) atol = 1e-8

    # 自旋算符（含非 Pauli 的 :P/:M 与同位多因子）：与 VQC 稠密展开对拍
    using QuantumCircuits.Hamiltonian: SpinOpTerm, SpinOpSum
    spt = SpinOpSum(SpinOpTerm(0.7, 1 => :Z, 2 => :P),
                    SpinOpTerm(0.4im, 3 => [0 1; -1 0], 3 => :Y),
                    SpinOpTerm(0.2, 4 => :M, 1 => :P))
    @test expectation(spt, psi) ≈ VQC.expectation(spt, ref) atol = 1e-10
    @test expectation(spt, rho) ≈ VQC.expectation(spt, VQC.DensityMatrix(ref)) atol = 1e-10

    # amplitude
    for bits in ([0, 0, 0, 0], [1, 0, 1, 0])
        @test amplitude(psi, bits) ≈ VQC.amplitude(ref, bits) atol = 1e-10
    end
end
