# test_circuit.jl — QuantumCircuits 接口扩展：simulate / 经典控制 / 线路工具

@testset "circuit interface" begin
    n = 4
    # ── simulate（非就地 / 就地）──
    c = Circuit(n)
    push!(c, H(1))
    push!(c, CX(1, 2))
    push!(c, CZ(2, 4))
    ref = simulate(c, VQC.zero_state(n))
    out = simulate(c, zero_state_mps(n))
    @test statevector(out) ≈ VQC.storage(ref) atol = 1e-10
    x = zero_state_mps(n)
    simulate!(c, x)
    @test statevector(x) ≈ VQC.storage(ref) atol = 1e-10

    # 混合态
    outr = simulate(c, densitymatrix_mps(n))
    refr = simulate(c, VQC.DensityMatrix(VQC.zero_state(n)))
    @test densitymatrix(outr) ≈ VQC.storage(refr) atol = 1e-10

    # ── 符号参数 ──
    cp = Circuit(2)
    push!(cp, RX(:θ, 1))
    push!(cp, RY(:θ, 2))                     # 同名参数 = 权重共享
    push!(cp, CX(1, 2))
    θv = 0.4
    s1 = simulate(cp, zero_state_mps(2); params=[θv])
    c2 = QuantumCircuits.assign(copy(cp), Dict(:θ => θv))
    s2 = simulate(c2, zero_state_mps(2))
    @test statevector(s1) ≈ statevector(s2) atol = 1e-12
    @test statevector(s1) ≈ VQC.storage(simulate(cp, VQC.zero_state(2); params=[θv])) atol = 1e-12
    s3 = simulate(cp, zero_state_mps(2); params=Dict(:θ => θv))
    @test statevector(s3) ≈ statevector(s1) atol = 1e-12

    # ── 测量 + 经典条件分支（IfOp / ClassicalStore）──
    cm = Circuit(1)
    push!(cm, H(1))
    push!(cm, measure(1, cm.cregs[1][1]))
    QuantumCircuits.if_then(cm, cm.cregs[1][1] == 1, Circuit([X(1)]))
    outm = simulate(cm, zero_state_mps(1))
    @test probabilities(outm)[1] ≈ 1 atol = 1e-12   # 测到 1 会被 X 翻回 |0⟩

    # ── BlockOp（重复 + 映射）──
    body = Circuit([X(1)]; n=1)
    cb = Circuit(4)
    push!(cb, QuantumCircuits.block(body; name=:flip, at=[3]))
    outb = simulate(cb, zero_state_mps(4))
    @test probabilities(outb)[(1 << 2) + 1] ≈ 1 atol = 1e-12

    cb2 = Circuit(2)
    push!(cb2, QuantumCircuits.block(Circuit([X(1)]; n=1); at=[1], repeat=2))
    @test statevector(simulate(cb2, zero_state_mps(2))) ≈ statevector(zero_state_mps(2)) atol = 1e-12

    # ── ResetOp / BarrierOp ──
    cr = Circuit(2)
    push!(cr, X(2))
    push!(cr, QuantumCircuits.reset(2))
    @test statevector(simulate(cr, zero_state_mps(2))) ≈ statevector(zero_state_mps(2)) atol = 1e-12

    cbar = Circuit(2)
    push!(cbar, barrier(1, 2))
    @test statevector(simulate(cbar, zero_state_mps(2))) ≈ statevector(zero_state_mps(2)) atol = 1e-12

    # ── QFT 线路与 VQC 稠密后端一致 ──
    # 测试临时构造：教科书 QFT（H + CPHASE 阶梯，输出比特序为输入逆序）
    function qft(L::Int)
        c = Circuit(L)
        for i in L:-1:1
            push!(c, H(i))
            for j in (i-1):-1:1
                push!(c, CPHASE(π / 2^(i - j), j, i))
            end
        end
        return c
    end
    q = qft(3)
    outq = simulate(q, onehot_encoding_mps([1, 0, 0]))
    refq = simulate(q, VQC.onehot_encoding([1, 0, 0]))
    @test statevector(outq) ≈ VQC.storage(refq) atol = 1e-10

    # ── 量子相位估计式线路（端到端：非相邻受控门 + dagger + pow 修饰符）──
    # 构造约定注记：本线路的受控幂次按 i 降序指派、与 dagger(qft) 的输入
    # 约定不构成标准 PEA——测试目的是验证 MPS 与稠密后端对同一 IR 线路
    # 给出完全一致的输出（含非相邻 SWAP 聚集、usergate、pow/ctrl/dagger）。
    to_digits(s) = sum(s[i] * 2.0^(-i) for i in 1:length(s))
    function phase_est_circuit(j::Vector{Int})
        L = length(j)
        φ = to_digits(j)
        c = Circuit(L + 1)
        for i in 1:L
            push!(c, H(i))
        end
        g = usergate(:U, [exp(2π * im * φ) 0; 0 1])
        for i in L:-1:1
            push!(c, ctrl(g(L + 1), i))
            g = pow(g, 2)
        end
        append!(c, dagger(qft(L)))
        return c
    end
    function phase_est(j::Vector{Int})
        L = length(j)
        return simulate(phase_est_circuit(j), zero_state_mps(L + 1))
    end
    for j in ([0, 0, 1], [1, 0, 1])
        L = length(j)
        st = phase_est(j)
        # 跨后端一致性（同一 IR 线路，MPS 与稠密后端必须逐步一致）
        refpe = simulate(phase_est_circuit(j), VQC.zero_state(L + 1))
        @test probabilities(st) ≈ VQC.probabilities(refpe) atol = 1e-8
        @test tr(densitymatrix(st)) ≈ 1 atol = 1e-8
    end

    # ── 变分线路（QuantumCircuits 构造器，显式数值参数）──
    Random.seed!(3)
    vc = variational_circuit_1d(3, 1; θs = rand(18) .* 2π)
    outv = simulate(vc, zero_state_mps(3))
    refv = simulate(vc, VQC.zero_state(3))
    @test statevector(outv) ≈ VQC.storage(refv) atol = 1e-10
    vcr = real_variational_circuit_1d(3, 1; θs = rand(6) .* 2π)
    outr2 = simulate(vcr, zero_state_mps(3))
    @test statevector(outr2) ≈ VQC.storage(simulate(vcr, VQC.zero_state(3))) atol = 1e-10

    # ── nqubits / measure 协议 ──
    @test QuantumCircuits.nqubits(out) == n
    st, o, p = QuantumCircuits.measure(onehot_encoding_mps([1]), 1)
    @test o == 1
    @test p ≈ 1 atol = 1e-12

    # ── Interface 后端（MatrixProductBackend）──
    b = MatrixProductBackend()
    @test QuantumCircuits.Interface.supports(b, :statevector)
    cb = Circuit(2)
    push!(cb, H(1))
    push!(cb, CX(1, 2))
    push!(cb, measure(1, cb.cregs[1][1]))
    push!(cb, measure(2, cb.cregs[1][2]))
    rs = simulate(cb, b)
    @test rs.state isa MPS
    rw = simulate(cb, b; shots = 32, seed = 7)
    @test rw.counts !== nothing && sum(values(rw.counts)) == 32
    @test all(k -> k == "c:00" || k == "c:11", keys(rw.counts))
end
