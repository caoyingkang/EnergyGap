## --------------------------- Set up ----------------------------------------
using CairoMakie
using DataFrames
using CSV
using HDF5
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
        os += -1.0, "Z", 3 * j - 2, "Z", 3 * (j + 1) - 2
        os += -1.0, "X", 3 * j, "X", 3 * (j + 1)
    end
    # vertical bonds
    for j in 1:m
        os += -1.0 * g, "Z", 3 * j - 2, "Z", 3 * j - 1
        os += -1.0, "X", 3 * j - 2, "X", 3 * j - 1
        os += -1.0, "Z", 3 * j - 1, "Z", 3 * j
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
    eigsolve_krylovdim::Int,
    outputlevel::Int,
    conv_check_length::Int,
    conv_tol::Float64,
)
    observer = EnergyObserver(conv_check_length=conv_check_length, conv_tol=conv_tol)
    if isnothing(ψs_to_avoid)
        # Find ground state
        E, ψ = dmrg(
            H, ψ0;
            nsweeps=nsweeps,
            maxdim=maxdim,
            cutoff=cutoff,
            eigsolve_krylovdim=eigsolve_krylovdim,
            outputlevel=outputlevel,
            observer=observer
        )
    else
        # Find excited state orthogonal to ψs_to_avoid
        E, ψ = dmrg(
            H, ψs_to_avoid, ψ0;
            weight=1.0,
            nsweeps=nsweeps,
            maxdim=maxdim,
            cutoff=cutoff,
            eigsolve_krylovdim=eigsolve_krylovdim,
            outputlevel=outputlevel,
            observer=observer
        )
    end
    return E, ψ
end


"""
Compute the energy gap between the ground state and first excited state using the DMRG algorithm.
"""
function compute_ground_and_first_excited_states(
    sites::Vector{Index{Int}},
    H::MPO;
    nsweeps::Int=100,
    maxdim::Vector{Int}=[10, 20, 100, 100, 200],
    cutoff::Float64=1e-10,
    eigsolve_krylovdim::Int=3,
    outputlevel::Int=1,
    conv_check_length::Int=4,
    conv_tol::Float64=1e-6,
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
        nsweeps=nsweeps,
        maxdim=maxdim,
        cutoff=cutoff,
        eigsolve_krylovdim=eigsolve_krylovdim,
        outputlevel=outputlevel,
        conv_check_length=conv_check_length,
        conv_tol=conv_tol
    )
    # Now find the first excited state (orthogonal to ground state)
    ψ1_init = random_mps(rng, sites; linkdims=100)
    E1, ψ1 = run_dmrg(
        H, ψ1_init;
        ψs_to_avoid=[ψ0],
        nsweeps=nsweeps,
        maxdim=maxdim,
        cutoff=cutoff,
        eigsolve_krylovdim=eigsolve_krylovdim,
        outputlevel=outputlevel,
        conv_check_length=conv_check_length,
        conv_tol=conv_tol
    )
    return E0, ψ0, E1, ψ1
end


k_values = 1:1:27
g = 1.0
df = DataFrame(k=k_values, E0=NaN, E1=NaN, ΔE=NaN, log10_ΔE=NaN)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "DMRG_energy_gap_vs_k.csv")
ΔE_vs_k_png_path = joinpath(results_dir, "DMRG_energy_gap_vs_k.png")
E0_vs_k_png_path = joinpath(results_dir, "DMRG_E0_vs_k.png")
h5_path = joinpath(results_dir, "DMRG_mps.h5")

# Load existing data points if they exist
if isfile(csv_path)
    let saved_df = CSV.read(csv_path, DataFrame)
        for r in eachrow(saved_df)
            idx = findfirst(==(r.k), df.k)
            df[idx, :] = r
        end
    end
end


## --------------------------- Run DMRG ----------------------------------------
for r in eachrow(df)
    if !isnan(r.ΔE)
        continue
    end
    k = r.k
    println("\nComputing for k = $k (total sites: $(6*k))...")
    sites, H = build_hamiltonian(k; g=g)
    E0, ψ0, E1, ψ1 = compute_ground_and_first_excited_states(
        sites, H,
        nsweeps=300,
        maxdim=[10, 20, 100, 100, 200],
        cutoff=1e-12,
        eigsolve_krylovdim=10,
        outputlevel=1,
        conv_check_length=4,
        conv_tol=1e-10
    )
    E0 += g * 2k
    E1 += g * 2k
    r.E0 = E0
    r.E1 = E1
    r.ΔE = E1 - E0
    r.log10_ΔE = log10(E1 - E0)
    h5write(h5_path, "k=$k/psi0", ψ0)
    h5write(h5_path, "k=$k/psi1", ψ1)
    println(r)
end

## --------------------------- Plot results ------------------------------------
fig1 = Figure()
ax1 = Axis(fig1[1, 1], xlabel="k", ylabel="ΔE", yscale=log10, title="Energy Gap vs k")
lines!(ax1, df.k, df.ΔE)
scatter!(ax1, df.k, df.ΔE)

# display(fig)
save(ΔE_vs_k_png_path, fig1)


fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel="k", ylabel="E0", title="Ground State Energy vs k")
lines!(ax2, df.k, df.E0)
scatter!(ax2, df.k, df.E0)

# display(fig)
save(E0_vs_k_png_path, fig2)


## --------------------------- Save results ------------------------------------
CSV.write(csv_path, df)
