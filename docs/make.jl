using Documenter
using LinearAlgebra
using QuantumCircuits
using QuantumCircuits.Hamiltonian: PauliTerm, PauliSum, SpinOpTerm, SpinOpSum
using MPSSimulator

makedocs(
	sitename = "MPSSimulator.jl",
	authors = "Guo Chu",
	pages = ["Home" => "index.md",
	         "gettingstarted.md",
	         "states.md",
	         "coreops.md",
	         "observables.md",
	         "noise.md",
	         "backend.md"],
	format = Documenter.HTML(
	    prettyurls = get(ENV, "CI", nothing) == "true"
	)
	)
