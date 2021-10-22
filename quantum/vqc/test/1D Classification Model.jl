using Yao, YaoExtensions
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using CSV
using DataFrames
using Plots
using StatsPlots
#using Flux

# read in data
one_d_data = CSV.File("1D Classification Data.csv") |> DataFrame
one_d_data = one_d_data[!, Not(:Column1)]

train = one_d_data[!, Not(:label)]
train = train[!,1]
labels = one_d_data[!, Not(:x)]
labels = labels[!,1]

# plot data
gdf = groupby(one_d_data, :2);
plot(gdf[2].x, gdf[2].label, seriestype=:scatter, color=:red, label='0')
plot!(gdf[1].x, gdf[1].label, seriestype=:scatter, color=:blue, label='1')

# split into train and test data
test = splice!(train, 20:59)
test_labels = splice!(labels, 20:59)

train
labels

#replace!(y, 0=>-1)

# set number of qubits
N = 6

# define product feature map
function feature_map(n, t)
    return chain(n, [put(i=>Ry(2*acos(t))) for i=1:n])
end

# cost function
magn = sum([chain(N, put(i=>Z)) for i=1:N])
sz1 = chain(N, put(1=>Z))

#---

sigmoid(x) = 1 ./(1. +exp(.-x))

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
        dCdθ = expect'(cost, (zero_state(N) |> feature_map(N, train[i])) => Uθ).second; # gradient of <C>
        grad_params += 2. * ((expect(cost, (zero_state(N) |> feature_map(N, train[i])) => Uθ) - labels[i]) |> real) * dCdθ; # full loss function grads
    end
    update!
    # feed the gradients into the circuit
    dispatch!(Uθ, update!(params, grad_params, optimizer))
    loss = 0.; # total cost
    for i=1:length(train)
        loss += (-1 * labels[i] * log(expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ)) |> real); #binary cross entropy
        #loss += (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) - labels[i])^2 |> real;
    end
    loss = loss * (1/length(train))
    println("Step $j, loss = $loss "); flush(Core.stdout)
    append!(loss_vec, loss)
end

#plot the losses at each epoch
epoch = 1:niter
plot(epoch, loss_vec, xaxis=("Epoch"), yaxis=("Loss"))

solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ, params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) |> real))
end

#---

# classify
function classifier(solution)
    predictions = Vector{Int64}()
    for i=1:length(solution)
        if solution[i] >= 0.5
            append!(predictions, 1)
        elseif solution[i] < 0.5
            append!(predictions, 0)
        end
    end
    predictions
end

predictions = classifier(solution)

# plot predictions
pred_df = DataFrame(x=train, predictions=predictions)
gdf2 = groupby(pred_df,:2);
plot!(gdf2[1].x, gdf2[1].predictions, seriestype=:scatter, color=:green, label='0')
plot!(gdf2[2].x, gdf2[2].predictions, seriestype=:scatter, color=:orange, label='1')

println("Finished")
#---
