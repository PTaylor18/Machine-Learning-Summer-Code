using Yao, YaoExtensions
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using CSV
using DataFrames
using Plots
#using Flux

# read in data
one_d_data = CSV.File("1D Classification Data.csv") |> DataFrame
one_d_data = one_d_data[!, Not(:Column1)]

train = one_d_data[!, Not(:label)]
train = train[!,1]
labels = one_d_data[!, Not(:x)]
labels = labels[!,1]

#replace!(y, 0=>-1)

# set number of qubits
N = 6

# define product feature map
function feature_map(n, t)
    return chain(n, [put(i=>Ry(asin(t))) for i=1:n])
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
    for i=1:length(train)
        # calculate sum of gradients
        dCdθ = expect'(cost, (zero_state(N) |> feature_map(N, train[i])) => Uθ).second; # gradient of <C>
        grad_params += 2. * ((expect(cost, (zero_state(N) |> feature_map(N, train[i])) => Uθ) - labels[i]) |> real) * dCdθ; # full loss function grads
    end
    # feed the gradients into the circuit
    dispatch!(Uθ, update!(params, grad_params, optimizer))
    loss = 0.; # total cost
    for i=1:length(train)
        loss += (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) - labels[i])^2 |> real;
    end

    println("Step $j, loss = $loss "); flush(Core.stdout)
    append!(loss_vec, loss)
end



#plot the losses at each epoch
epoch = 1:niter
plot(epoch, loss_vec, xaxis=("Epoch"), yaxis=("Loss"))

solution = Vector{Float64}()
for i=1:length(train)
    dispatch!(Uθ, params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ)) |> real)
end
println(solution)
solution

# plot the solution
fplot = plot!(tlist, solution, xaxis = ("t"), yaxis = ("f(t)"), label = "QNN fit", markersize = 6, c = :blue, marker = "x")

println("Finished")
#---
