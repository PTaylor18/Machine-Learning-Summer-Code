using Yao, YaoExtensions
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using DataFrames, CSV
using LIBSVM
using Random

# define product feature map
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(3*asin(x))) for i=1:n]), chain(n, [put(i=>Ry(2*acos(t))) for i=1:n]))
end

# example feature map for 3 features
function feature_map(n, x, t, z)
    return chain(n,
	 chain(n, [put(i=>Rz(3*acos(x))) for i=1:n]),
	 chain(n, [put(i=>Ry(2*asin(t))) for i=1:n]),
	 chain(n, [put(i=>Rx(2*acos(z))) for i=1:n]))
end

# observable
magn = sum([chain(N, put(i=>Z)) for i=1:N])
sz1 = chain(N, put(1=>Z))

# train the model
optimizer = Adam(lr=0.01)
d = 6; # QNN depth
Uθ = variational_circuit(N, d)
dispatch!(Uθ, :random)
params = parameters(Uθ)
loss = 0. # total cost
loss_vec = Vector{Float64}()
grad_params_sum = zeros(length(params)) # vector of derivatives
niter = 200
cost = magn

for j = 1:niter
    grad_params = zeros(length(params)); # vector of derivatives
    for i=1:length(labels)
        # calculate sum of gradients
        dCdθ = expect'(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) => Uθ).second; # gradient of <C>
        grad_params += 2. * ((expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) => Uθ) - labels[i]) |> real) * dCdθ; # full loss function grads
    end
    # feed the gradients into the circuit
    dispatch!(Uθ, update!(params, grad_params, optimizer))
    loss = 0.; # total cost
    for i=1:length(labels)
        loss += (-1 * labels[i] * log(expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ)) |> real); # binary cross entropy
        #loss += (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) - labels[i])^2 |> real; # L2 loss
    end
    loss = loss * (1/length(labels))
    println("Step $j, loss = $loss "); flush(Core.stdout)
    append!(loss_vec, loss)
end

#plot the losses at each epoch
epoch = 1:niter
plot(epoch, loss_vec, xaxis=("Epoch"), yaxis=("Loss"))

# use final circuit parameters
solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ, params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ) |> real))
end

# classify using SVM
model = svmtrain(solution', labels)
predictions = svmpredict(model, solution')[1]

"""
	now evaluate the model using accuracy and F1 score
"""
using EvalMetrics

cm = EvalMetrics.ConfusionMatrix(labels, predictions) # confusion matrix
accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)
