using QuantumToolbox: sigmax, sigmaz, multisite_operator

const X = sigmax()
const Z = sigmaz()


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
function build_hamiltonian(::Val{K}; g::Float64=1.0) where {K}
    m = 2 * K  # number of rungs
    # horizontal couplings
    H = sum(
        multisite_operator(Val(6K), 3 * j - 2 => Z, 3 * (j + 1) - 2 => Z) +
        multisite_operator(Val(6K), 3 * j => X, 3 * (j + 1) => X)
        for j in 1:(m-1)
    )
    # vertical couplings
    H += sum(
        g * multisite_operator(Val(6K), 3 * j - 2 => Z, 3 * j - 1 => Z) +
        multisite_operator(Val(6K), 3 * j - 2 => X, 3 * j - 1 => X) +
        multisite_operator(Val(6K), 3 * j - 1 => Z, 3 * j => Z)
        for j in 1:m
    )
    return H.data
end
