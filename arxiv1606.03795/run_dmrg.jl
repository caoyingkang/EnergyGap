## --------------------------- Set up ----------------------------------------
using CairoMakie
using DataFrames
using CSV
using HDF5
using Random
using Statistics
using ITensors, ITensorMPS


"""
Build the Hamiltonian for the [[2N+2, N, 2]] code described in https://arxiv.org/abs/1606.03795.
Add to the Hamiltonian extra penalty terms with strength `g` (default 1.0) to lift the ground space degeneracy.

The sites of the lattice are indexed in this way:

1---XX---2---(ZZ+gXX)---3---XX---4---(ZZ+gXX)---5--- ... ---(2N)---(ZZ+gXX)---(2N+1)---XX---(2N+2)
|                                                                                            |
|_______________________________________________ZZ___________________________________________|
"""
function build_hamiltonian(N::Int; g::Float64=1.0)
    sites = siteinds("Qubit", 2N + 2)
    os = OpSum()
    for i in 1:(N+1)
        os += -1.0, "X", 2i - 1, "X", 2i
    end
    for i in 1:N
        os += -1.0, "Z", 2i, "Z", 2i + 1
        os += -1.0 * g, "X", 2i, "X", 2i + 1
    end
    os += -1.0, "Z", 1, "Z", 2N + 2
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


N_values = 2:1:15
g = 1.0
df = DataFrame(N=N_values, E0=NaN, E1=NaN, ΔE=NaN, log10_ΔE=NaN, ln_ΔE=NaN)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)
csv_path = joinpath(results_dir, "DMRG_energy_gap_vs_N.csv")
ΔE_vs_N_png_path = joinpath(results_dir, "DMRG_energy_gap_vs_N.png")
E0_vs_N_png_path = joinpath(results_dir, "DMRG_E0_vs_N.png")
h5_path = joinpath(results_dir, "DMRG_mps.h5")

# Load existing data points if they exist
if isfile(csv_path)
    let saved_df = CSV.read(csv_path, DataFrame)
        for r in eachrow(saved_df)
            idx = findfirst(==(r.N), df.N)
            df[idx, :] = r
        end
    end
end


## --------------------------- Run DMRG ----------------------------------------
for r in eachrow(df)
    if !isnan(r.ΔE)
        continue
    end
    N = r.N
    println("\nComputing for N = $N (total sites: $(2N+2))...")
    sites, H = build_hamiltonian(N; g=g)
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
    E0 += g * N
    E1 += g * N
    r.E0 = E0
    r.E1 = E1
    r.ΔE = E1 - E0
    r.log10_ΔE = log10(E1 - E0)
    r.ln_ΔE = log(E1 - E0)
    h5write(h5_path, "N=$N/psi0", ψ0)
    h5write(h5_path, "N=$N/psi1", ψ1)
    println(r)
end

## --------------------------- Plot results ------------------------------------
fig1 = Figure()
ax1 = Axis(fig1[1, 1], xlabel="ln(N+1)", ylabel="ln(ΔE)", title="Energy Gap vs N")
lines!(ax1, log.(df.N .+ 1), df.ln_ΔE)
scatter!(ax1, log.(df.N .+ 1), df.ln_ΔE)
xlims!(ax1, 1, 3)
ylims!(ax1, -2.2, -0.6)


# display(fig)
save(ΔE_vs_N_png_path, fig1)


fig2 = Figure()
ax2 = Axis(fig2[1, 1], xlabel="N", ylabel="E0", title="Ground State Energy vs N")
lines!(ax2, df.N, df.E0)
scatter!(ax2, df.N, df.E0)

# display(fig)
save(E0_vs_N_png_path, fig2)


## --------------------------- Save results ------------------------------------
CSV.write(csv_path, df)
