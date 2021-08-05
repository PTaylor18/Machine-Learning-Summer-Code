using Yao, YaoExtensions
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using CSV
using DataFrames
using Plots

# read in data
breast_cancer_pca_2_train = CSV.File("breast_cancer_pca_2_train.csv") |> DataFrame
breast_cancer_pca_2_train = breast_cancer_pca_2_train[!, Not(:Column1)]

breast_cancer_labels = CSV.File("breast_cancer_labels.csv") |> DataFrame
breast_cancer_labels = breast_cancer_labels[!, Not(:Column1)]

X_train = breast_cancer_pca_2_train
y_train = breast_cancer_labels

x = breast_cancer_pca_2_train[!,1]
t = breast_cancer_pca_2_train[!,2]

y = breast_cancer_labels[!,1]

# set number of qubits
N = 6

# define product feature map
function pca_1_feature_map(n, x)
    return chain(n, [put(i=>Rz(asin(x))) for i=1:n])
end

function pca_2_feature_map(n, t)
    return chain(n, [put(i=>Ry(asin(t))) for i=1:n])
end

# cost function
magn = sum([chain(N, put(i=>Z)) for i=1:N])

#---

optimizer = Adam(lr=0.01)
d = 6; # QNN depth
Uθ = variational_circuit(N, d)
dispatch!(Uθ, :random)
params = parameters(Uθ)
loss = 0. # total cost
grad_params_sum = zeros(length(params)) # vector of derivatives
niter = 200
training_data = flist
data_points = tlist
cost = magn
for j = 1:niter
    grad_params = zeros(length(params)); # vector of derivatives
    for i=1:length(x)
        # calculate sum of gradients
        dCdθ = expect'(cost, (zero_state(N) |> pca_1_feature_map(N, x[i]) |> pca_2_feature_map(N, t[i])) => Uθ).second; # gradient of <C>
        grad_params += 2. * ((expect(cost, (zero_state(N) |> pca_1_feature_map(N, x[i]) |> pca_2_feature_map(N, t[i])) => Uθ) - y[i]) |> real) * dCdθ; # full loss function grads
    end
    # feed the gradients into the circuit
    dispatch!(Uθ, update!(params, grad_params, optimizer))
    loss = 0.; # total cost
    for i=1:length(x)
        loss += (expect(cost, (zero_state(N) |> pca_1_feature_map(N, x[i]) |> pca_2_feature_map(N, t[i])) |> Uθ) - y[i])^2 |> real;
    end
    println("Step $j, loss = $loss "); flush(Core.stdout)
end

solution = Vector{Float64}()
for i=1:length(tlist)
    dispatch!(Uθ, params)
    append!(solution, (expect(cost, (zero_state(N) |> pca_1_feature_map(N, x[i]) |> pca_2_feature_map(N, t[i])) |> Uθ)) |> real)
end
println(solution)

# plot the solution
fplot = plot!(tlist, solution, xaxis = ("t"), yaxis = ("f(t)"), label = "QNN fit", markersize = 6, c = :blue, marker = "x")

println("Finished")
#---

expect
variat
