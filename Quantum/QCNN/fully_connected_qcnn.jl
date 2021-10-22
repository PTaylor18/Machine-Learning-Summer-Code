using Yao, YaoExtensions
using Optim
using DataFrames, CSV
using LIBSVM
using Random

include("w_ansatz.jl")

"""
    fully connected quantum convolutional neural network with no pooling layers

    architecture: feature_map |> w_ansatz |> w_ansatz... |> measure on all qubits

    following architecure from arXiv:2011.02966v1

"""

# define feature map for 2 features
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(3*asin(x))) for i=1:n]), chain(n, [put(i=>Ry(2*acos(t))) for i=1:n]))
end

# observable
magn = chain(N, prod([put(N, i=>Z) for i=1:N]))
sz1 = chain(N, put(N=>Z))

# define fully connected circuit
function FC_QCNN(N, θ_plus, θ_minus)

    cirq = chain(N)
    push!(cirq, w_circuit(N, 4))

end

#---

# Random.seed!(28)

# assign paramters to QCNN
Uθ_FC = FC_QCNN(8, 0.1, 0.2)
dispatch!(Uθ_FC, :random)
params = parameters(Uθ_FC)
cost = sz1

# train the circuit

function cost_optim(x)
	cost_sum = 0

	for i=1:length(labels)
		cost_sum += (1/length(labels)) * (-1* labels[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ_FC, x))) |> real)
	end
	cost_sum
end

begin
	res = Optim.minimizer((optimize(x->cost_optim(x),
			parameters(Uθ_FC),
			LBFGS(),
			Optim.Options(iterations=1))))
end

# now use trained parameters to give classification of unseen data
new_params = parameters(Uθ_FC)
solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ_FC, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ_FC) |> real))
end

# support vector machine to classify
model = svmtrain(solution', labels)
predictions = svmpredict(model, solution')[1]

"""
	now evaluate the model using accuracy and f1 score
"""
using EvalMetrics

cm = EvalMetrics.ConfusionMatrix(labels, predictions) # confusion matrix
accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)
