using Yao, YaoExtensions, Compose
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using Optim
using DataFrames, CSV
using Plots
using YaoPlots
using Test, Random
using LIBSVM

include("w_ansatz.jl")
include("pooling.jl")

"""
    quantum convolutional neural network with convolution and pooling layers

    architecture: feature_map |> w_ansatz |> pool |> w_ansatz |> pool... |> measure

    following architecures from arXiv:2011.02966v1, arXiv:2108.00661v1, arXiv:2009.09423v1

"""

# read in data
two_d_data = CSV.File("2D Classification Data.csv") |> DataFrame
two_d_data = two_d_data[!, Not(:Column1)]

train = two_d_data[!, Not(:label)]
feature_1 = train[!,1]
feature_2 = train[!,2]

labels = two_d_data[!,3]

# plot data
begin
	gdf = groupby(two_d_data, :3)
	plot(gdf[2].x, gdf[2].y, seriestype=:scatter, color=:red, label='0')
	plot!(gdf[1].x, gdf[1].y, seriestype=:scatter, color=:blue, label='1')
end

# set number of qubits
N = 8

# define feature map
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(3*acos(x))) for i=1:n]), chain(n, [put(i=>Ry(2*asin(t))) for i=1:n]))
end

# observable
zz = chain(N, prod([put(N,i=>Z) for i=4:4:8]))
sz1 = chain(N, put(N=>Z))

#---

function QCNN(N, θ_plus, θ_minus)
    """
    Number of qubits must be a a power of 2
    """

    n_layers = log10(N)/log10(2)

    layer1 = chain(N)
    push!(layer1, w_circuit(N, 2)) # conv1
    push!(layer1, chain(N, prod([put(N, (i, i+1) => pool(θ_plus, θ_minus)) for i in 1:2:(N - 1)]))) # pool1

    #layer2 = chain(N)
    push!(layer1, w_circuit2(N, 2)) # conv1
    push!(layer1, chain(N, prod([put(N, (i, i+2) => pool(θ_plus, θ_minus)) for i in 2:4:(N - 2)]))) # pool2

	#layer3 = chain(N)
	push!(layer1, f_circuit(N, 1)) # fully connected
	push!(layer1, chain(N, put(N, (4, 8) => pool(θ_plus, θ_minus)))) # pool 3

    #cirq = chain(N, put((1:N)=>layer1), put((1:N)=>layer2))

end

#---

# tests to see if circuit works
statetest = rand_state(8)
stest = copy(statetest) |> QCNN(8, 0.1, 0.2)

statevec(stest)
rho = statevec(stest) * (statevec(stest))'


#---

# train the circuit

Uθ = QCNN(8, 0.1, 0.2)
dispatch!(Uθ, :random)
params = parameters(Uθ)
loss = 0. # total cost
loss_vec = Vector{Float64}()
grad_params_sum = zeros(length(params)) # vector of derivatives
niter = 200
cost = sz1

# plot circuit with YaoPlots
plot(Uθ)

"""
	potentially alter by having cost function as -Σ (y . log(ŷ)) and sum for each label.

	loop over each label so that the cost function is the sum from each label
		then optimize the sum cost function

			need to figure out dispatching for epochs and actual loss
"""



#(-1* labels[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ, x))) |> real)
#((expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ, x)) - labels[i]) |> real)
begin
    for i=1:length(labels)
		res = (optimize(x->(-1* labels[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ, x))) |> real),
	            parameters(Uθ),
	            LBFGS(),
	            Optim.Options(iterations=1)))
    end
    # feed the gradients into the circuit
    #dispatch!(Uθ, new_params)
    #loss = 0.; # total cost
    # for i=1:length(labels)
    #     loss += (-1 * labels[i] * log(expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ)) |> real); # binary cross entropy
    #     #loss += (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) - labels[i])^2 |> real; # L2 loss
    # end
    # loss = loss * (1/length(labels))
    # println("Step $j, loss = $loss "); flush(Core.stdout)
    # append!(loss_vec, loss)
end

plot(epoch, loss_vec, xaxis=("Epoch"), yaxis=("Loss"))

new_params = parameters(Uθ)

solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ) |> real))
end

#---

solution
plot(feature_1, solution, seriestype=:scatter)
labels

# support vector machine

using RDatasets

iris = dataset("datasets", "iris")

X = Matrix(iris[:, 1:4])'
y = iris.Species

Xtrain = X[:, 1:2:end]
Xtest  = X[:, 2:2:end]
ytrain = y[1:2:end]
ytest  = y[2:2:end]

model = svmtrain(Xtrain, ytrain)


solution = reshape(solution, length(solution), 1)

model = svmtrain(solution, labels)


typeof(solution)

function classifier(solution)
    predictions = Vector{Int64}()
    for i=1:length(solution)
        if solution[i] >= 0
            append!(predictions, 0)
        elseif solution[i] < 0
            append!(predictions, 1)
        end
    end
    predictions
end

# svm

predictions = classifier(solution)

begin
	gdf = groupby(two_d_data, :3)
	plot(gdf[2].x, gdf[2].y, seriestype=:scatter, color=:red, label='0')
	plot!(gdf[1].x, gdf[1].y, seriestype=:scatter, color=:blue, label='1')

	pred_df = DataFrame(x=feature_1, y=feature_2, predictions=predictions)
	gdf2 = groupby(pred_df,:3);
	plot!(gdf2[1].x, gdf2[1].y, seriestype=:scatter, color=:green, label='0')
	plot!(gdf2[2].x, gdf2[2].y, seriestype=:scatter, color=:orange, label='1')
end
