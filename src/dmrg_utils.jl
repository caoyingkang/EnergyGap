using Random
using Statistics
using ITensors, ITensorMPS


"""
Build the Hamiltonian for the [[6k, 2k, 2]] code described in https://arxiv.org/abs/1911.01354.
Add to the Hamiltonian extra penalty terms with strength `g` (default 1.0) to lift the ground space degeneracy.

The sites of the lattice are indexed in this way:

1---ZZ---4---ZZ--- ... ---3⋅(2k)-2
|        |                 |
XX+gZZ  XX+gZZ            XX+gZZ
|        |                 |
2        5         ...    3⋅(2k)-1
|        |                 |
ZZ       ZZ                ZZ
|        |                 |
3---XX---6---XX--- ... ---3⋅(2k)
"""
function build_hamiltonian(k::Int; g::Float64=1.0)
    sites = siteinds("Qubit", 6 * k)
    m = 2 * k  # number of rungs of the lattice
    os = OpSum()
    # horizontal bonds
    for j in 1:(m-1)
        os += "Z", 3 * j - 2, "Z", 3 * (j + 1) - 2
        os += "X", 3 * j, "X", 3 * (j + 1)
    end
    # vertical bonds
    for j in 1:m
        os += g, "Z", 3 * j - 2, "Z", 3 * j - 1
        os += "X", 3 * j - 2, "X", 3 * j - 1
        os += "Z", 3 * j - 1, "Z", 3 * j
    end
    H = MPO(os, sites)
    return sites, H
end


mutable struct EnergyObserver <: AbstractObserver
    energies::Vector{Float64} # Recorded energies after each sweep.
    conv_check_length::Int # Must have at least this many recorded energies to check for convergence.
    conv_tol::Float64 # If the difference of the recently recorded energies is less than this value, early stop the DMRG.

    EnergyObserver(; conv_check_length::Int, conv_tol::Float64) =
        new([], conv_check_length, conv_tol)
end


function ITensorMPS.measure!(obs::EnergyObserver; kwargs...)
    if kwargs[:sweep_is_done]
        push!(obs.energies, kwargs[:energy])

        # Report memory usage
        if kwargs[:outputlevel] > 1
            psi_size = Base.format_bytes(Base.summarysize(kwargs[:psi]))
            PH_size = Base.format_bytes(Base.summarysize(kwargs[:projected_operator]))
            @info "psi_size = $psi_size, PH_size = $PH_size"
        end
    end
end


function ITensorMPS.checkdone!(obs::EnergyObserver; kwargs...)
    if length(obs.energies) < obs.conv_check_length
        return false
    end
    recent_energies = obs.energies[end-obs.conv_check_length+1:end]
    μ = mean(recent_energies)
    if any(abs.(recent_energies .- μ) .> obs.conv_tol)
        return false
    end
    if kwargs[:outputlevel] > 0
        println("Early stop DMRG.")
    end
    return true
end


"""
Run the DMRG algorithm on the Hamiltonian `H` and initial state `ψ0`.
If `ψs_to_avoid` is provided (as a vector of MPS), finds excited states orthogonal to them.
"""
function run_dmrg(
    H::MPO,
    ψ0::MPS;
    ψs_to_avoid::Union{Vector{MPS},Nothing}=nothing,
    nsweeps::Int,
    maxdim::Vector{Int},
    cutoff::Float64,
    outputlevel::Int,
    conv_check_length::Int,
    conv_tol::Float64,
)
    observer = EnergyObserver(conv_check_length=conv_check_length, conv_tol=conv_tol)
    if isnothing(ψs_to_avoid)
        # Find ground state
        E, ψ = dmrg(H, ψ0;
            nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=outputlevel, observer=observer)
    else
        # Find excited state orthogonal to ψs_to_avoid
        E, ψ = dmrg(H, ψs_to_avoid, ψ0; weight=1.0,
            nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=outputlevel, observer=observer)
    end
    return E, ψ
end

# TODO: change kwargs to try to improve convergence: eigsolve_krylovdim, noise


"""
Compute the energy gap between the ground state and first excited state using the DMRG algorithm.
"""
function compute_ground_and_first_excited_states(
    sites::Vector{Index{Int}},
    H::MPO;
    nsweeps::Int=100,
    maxdim::Vector{Int}=[10, 20, 100, 100, 200],
    cutoff::Float64=1e-10,
    outputlevel::Int=1,
    conv_check_length::Int=4,
    conv_tol::Float64=1e-8,
    rng::AbstractRNG=Random.default_rng()
)
    println("Computing energy gap using DMRG...")
    @show nsweeps
    @show maxdim
    @show cutoff
    @show outputlevel

    # First get the ground state
    ψ0_init = random_mps(rng, sites; linkdims=100)
    E0, ψ0 = run_dmrg(
        H, ψ0_init;
        nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=outputlevel,
        conv_check_length=conv_check_length, conv_tol=conv_tol
    )
    # Now find the first excited state (orthogonal to ground state)
    ψ1_init = random_mps(rng, sites; linkdims=100)
    E1, ψ1 = run_dmrg(
        H, ψ1_init; ψs_to_avoid=[ψ0],
        nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff, outputlevel=outputlevel,
        conv_check_length=conv_check_length, conv_tol=conv_tol
    )
    return E0, ψ0, E1, ψ1
end