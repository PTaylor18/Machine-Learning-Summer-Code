using Yao, YaoExtensions, YaoBase

include("w_ansatz.jl")

"""
    fully connected quantum neural network with no pooling layers

    architecture: feature_map |> w_ansatz |> w_ansatz... |> measure on all qubits

    following architecure from arXiv:2011.02966v1

"""

# define fully connected circuit
function FCNN(N)

    cirq = chain(N)
    push!(cirq, w_circuit(N, 4))

end
