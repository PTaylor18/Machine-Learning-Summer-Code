using Yao, YaoExtensions, YaoPlots, Compose
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using Optim
using DataFrames, CSV
using Plots

# read in data
two_d_data = CSV.File("2D Classification Data.csv") |> DataFrame
two_d_data = two_d_data[!, Not(:Column1)]

train = two_d_data[!, Not(:label)]
feature_1 = train[!,1]
feature_2 = train[!,2]

labels = two_d_data[!,3]

# set number of qubits
N = 8

# define feature map - Ry(arctan(x[i])) + Rz(arctan(x^2[i]))
function feature_map(n, x)
    return chain(n, chain(n, [put(i=>Ry(atan(x))) for i=1:n]), chain(n, [put(i=>Rz(atan(x^2))) for i=1:n]))
end

 # define feature map - Rz(arcsin(x) + arcsin(x)) + Ry(arcsin(x) + arcsin(x))
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(2*asin(x))) for i=1:n]), chain(n, [put(i=>Ry(2*acos(t))) for i=1:n]))
end

function LBFGS_optimizer(f, initial_x)
    return optimize(f, inital_x, LBFGS())
end

# observable
sz1 = chain(N, put(8=>Z))

function QCNN()
    # conv1
    Uθ_1 = variational_circuit(N,1)

    # pool1
    push!(Uθ_1, chain(N, prod([control(N, i,i+1 => Rz(0)) for i=1:2:N])))

    # conv2
    push!(Uθ_1, chain(N, prod([put(N, i => Rx(0)) for i=2:2:N])))
    push!(Uθ_1, chain(N, prod([put(N, i => Rz(0)) for i=2:2:N])))

    for i=2:2:N-1
        push!(Uθ_1, chain(N,cnot(N, i, i+2)))
    end

    push!(Uθ_1, chain(N, cnot(N,8,2)))

    push!(Uθ_1, chain(N, prod([put(N, i => Rz(0)) for i=2:2:N])))
    push!(Uθ_1, chain(N, prod([put(N, i => Rx(0)) for i=2:2:N])))

    # pool2
    push!(Uθ_1, chain(N, prod([control(N, i,i+2 => Rz(0)) for i=2:4:N-1])))

    #conv3
    push!(Uθ_1, chain(N, prod([put(N, i => Rx(0)) for i=4:4:N])))
    push!(Uθ_1, chain(N, prod([put(N, i => Rz(0)) for i=4:4:N])))

    push!(Uθ_1, chain(N, cnot(N,4,8)))
    push!(Uθ_1, chain(N, cnot(N,8,4)))

    push!(Uθ_1, chain(N, prod([put(N, i => Rz(0)) for i=4:4:N])))
    push!(Uθ_1, chain(N, prod([put(N, i => Rx(0)) for i=4:4:N])))

    #pool3
    push!(Uθ_1, control(N, 4,8=>Rz(0)))
end

Uθ = QCNN()
dispatch!(Uθ, :random)
params = parameters(Uθ)
optimizer = Adam(lr=0.01)
loss = 0. # total cost
loss_vec = Vector{Float64}()
grad_params_sum = zeros(length(params)) # vector of derivatives
niter = 200
cost = sz1

plot(Uθ_1)

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
