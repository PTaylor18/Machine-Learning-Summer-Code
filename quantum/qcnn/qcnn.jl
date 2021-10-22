using Yao, YaoExtensions
using Optim
using DataFrames, CSV
using LIBSVM
using Random

include("w_ansatz.jl")
include("pooling.jl")

"""
    quantum convolutional neural network with convolution and pooling layers

    architecture: feature_map |> w_ansatz |> pool |> w_ansatz |> pool... |> measure

    following architecures from arXiv:2011.02966v1, arXiv:2108.00661v1, arXiv:2009.09423v1
"""

# define feature map for 2 features - feature maps can be custom defined
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(3*asin(x))) for i=1:n]), chain(n, [put(i=>Ry(2*acos(t))) for i=1:n]))
end

# example feature map for 3 features
function three_feature_map(n, x, t, z)
    return chain(n,
	 chain(n, [put(i=>Rz(3*acos(x))) for i=1:n]),
	 chain(n, [put(i=>Ry(2*asin(t))) for i=1:n]),
	 chain(n, [put(i=>Rx(2*acos(z))) for i=1:n]))
end

# observable
zz = chain(N, prod([put(N,i=>Z) for i=4:4:8]))
sz1 = chain(N, put(N=>Z))

#---

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

# train the circuit

# Random.seed!(28)

Uθ = QCNN(8, 0.1, 0.2)
dispatch!(Uθ, :random)
params = parameters(Uθ)
cost = sz1

"""
	need to figure out loss recording using optim
"""

function cost_optim(x)
	cost_sum = 0

	for i=1:length(y_train)
		cost_sum += (1/length(y_train)) * (-1* y_train[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ, x))) |> real)
	end
	println(cost_sum)
	cost_sum
end

begin
	res = Optim.minimizer((optimize(x->cost_optim(x),
			parameters(Uθ),
			LBFGS(),
			Optim.Options(iterations=1))))
end

# Thetas for the pooling layers are not yet constant as they should be although the model is still sufficient
# need to figure out how to keep constant in initial random dispatch and during optimization

# now use trained parameters to give classification of unseen data
new_params = parameters(Uθ)
solution = Vector{Float64}()
for i=1:length(y_test)
    dispatch!(Uθ, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ) |> real))
end

# support vector machine to classify
model = svmtrain(solution', y_test)
predictions = svmpredict(model, solution')[1]

"""
	now evaluate the model using accuracy and F1 score
"""

cm = EvalMetrics.ConfusionMatrix(y_test, predictions) # confusion matrix
accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)
