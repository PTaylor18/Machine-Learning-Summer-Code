using Yao, YaoExtensions

include("w_ansatz.jl")
include("pooling.jl")

"""
    quantum convolutional neural network with convolution and pooling layers

    architecture: feature_map |> w_ansatz |> pool |> w_ansatz |> pool... |> measure

    following architecures from arXiv:2011.02966v1, arXiv:2108.00661v1, arXiv:2009.09423v1
"""

#---

# currently working for 8 qubits

function QCNN(N, θ_plus, θ_minus)
    """
    Number of qubits must be a power of 2
	N: No. of qubits
	θ_plus/θ_minus: Angles for pooling operation
    """

    n_layers = log10(N)/log10(2)

    cirq = chain(N)
    push!(cirq, w_circuit(N, 2)) # conv1
    push!(cirq, chain(N, prod([put(N, (i, i+1) => pool(θ_plus, θ_minus)) for i in 1:2:(N - 1)]))) # pool1

    push!(cirq, w_circuit2(N, 2)) # conv2
    push!(cirq, chain(N, prod([put(N, (i, i+2) => pool(θ_plus, θ_minus)) for i in 2:4:(N - 2)]))) # pool2

	push!(cirq, f_circuit(N, 1)) # conv3
	push!(cirq, chain(N, put(N, (4, 8) => pool(θ_plus, θ_minus)))) # pool 3

end
