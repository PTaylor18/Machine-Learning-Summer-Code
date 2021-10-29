using EvalMetrics
using DataFrames, CSV
using Random
using Plots

include("vqc/vqc.jl")
include("qcnn/qcnn.jl")
include("fcnn/fully_connected_neural_network.jl")

"""
	load in three feature dataset
"""

three_feature_data = CSV.File("github.com/PTaylor18/Machine-Learning-Summer-Code/tree/main/datasets/3 Feature Classification Data.csv") |> DataFrame
three_feature_data = three_feature_data[!, Not(:Column1)]
train = three_feature_data[!, Not(:label)]
begin
	feature_1 = train[!,1]
	feature_2 = train[!,2]
	feature_3 = train[!,3]
	labels = three_feature_data[!,4]
end

begin
	gdf = groupby(three_feature_data, :4)
	plot(gdf[2].x, gdf[2].y, seriestype=:scatter, color=:red, label='0')
	plot!(gdf[1].x, gdf[1].y, seriestype=:scatter, color=:blue, label='1')
end

"""
	define feature map for 3 features
"""

function feature_map(n, x, t, z)
    return chain(n,
	 chain(n, [put(i=>Rz(3*acos(x))) for i=1:n]),
	 chain(n, [put(i=>Ry(2*asin(t))) for i=1:n]),
	 chain(n, [put(i=>Rx(2*acos(z))) for i=1:n]))
end

#observable
magn = sum([chain(N, put(i=>Z)) for i=1:N])
sz1 = chain(N, put(1=>Z))

"""
	training the variational quantum classifier
	uses automatic differentiation and an ADAM optimizer
"""

# Random.seed!(28)

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
        dCdθ = expect'(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i])) => Uθ).second; # gradient of <C>
        grad_params += 2. * ((expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i])) => Uθ) - labels[i]) |> real) * dCdθ; # full loss function grads
    end
    # feed the gradients into the circuit
    dispatch!(Uθ, update!(params, grad_params, optimizer))
    loss = 0.; # total cost
    for i=1:length(labels)
        loss += (-1 * labels[i] * log(expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i])) |> Uθ)) |> real); # binary cross entropy
        #loss += (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) - labels[i])^2 |> real; # L2 loss
    end
    loss = loss * (1/length(labels))
    println("Step $j, loss = $loss "); flush(Core.stdout)
    append!(loss_vec, loss)
end

solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ, params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i])) |> Uθ) |> real))
end

# classify using SVM
model = svmtrain(solution', labels)
predictions = svmpredict(model, solution')[1]

# now evaluate the model using accuracy and f1 score
cm = EvalMetrics.ConfusionMatrix(labels, predictions) # confusion matrix
accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)

"""
	training the quantum convolutional neural network
	uses LBFGS optimization
"""

# Random.seed!(28)

Uθ = QCNN(8, 0.1, 0.2)
dispatch!(Uθ, :random)
params = parameters(Uθ)
cost = sz1

# optimization function
function cost_optim(x)
	cost_sum = 0

	for i=1:length(y_train)
		cost_sum += (1/length(y_train)) * (-1* y_train[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i]) => dispatch!(Uθ, x))) |> real)
	end
	println(cost_sum)
	cost_sum
end

# LBFGS optimizer
begin
	res = Optim.minimizer((optimize(x->cost_optim(x),
			parameters(Uθ),
			LBFGS(),
			Optim.Options(iterations=1))))
end

# now use trained parameters to give classification
new_params = parameters(Uθ)
solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i])) |> Uθ) |> real))
end

# support vector machine to classify
model = svmtrain(solution', labels)
predictions = svmpredict(model, solution')[1]

# now evaluate the model using accuracy and f1 score
cm = EvalMetrics.ConfusionMatrix(labels, predictions) # confusion matrix
accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)

"""
	training the fully connected quantum neural network
	uses LBFGS optimization
"""

# Random.seed!(28)

# assign paramters to fully connected neural network
Uθ_FC = FCNN(8)
dispatch!(Uθ_FC, :random)
params = parameters(Uθ_FC)
cost = sz1

# optimization function
function cost_optim(x)
	cost_sum = 0

	for i=1:length(labels)
		cost_sum += (1/length(labels)) * (-1* labels[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i]) => dispatch!(Uθ_FC, x))) |> real)
	end
	cost_sum
end

# LBFGS optimizer
begin
	res = Optim.minimizer((optimize(x->cost_optim(x),
			parameters(Uθ_FC),
			LBFGS(),
			Optim.Options(iterations=1))))
end

# now use trained parameters to give classification
new_params = parameters(Uθ_FC)
solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ_FC, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i], feature_3[i])) |> Uθ_FC) |> real))
end

# support vector machine to classify
model = svmtrain(solution', labels)
predictions = svmpredict(model, solution')[1]

# now evaluate the model using accuracy and f1 score
cm = EvalMetrics.ConfusionMatrix(labels, predictions) # confusion matrix
accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)
