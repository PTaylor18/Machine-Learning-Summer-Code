using Yao, YaoExtensions, Compose
using LinearAlgebra
using QuAlgorithmZoo: Adam, update!
using Optim
using DataFrames, CSV
using Plots
using YaoPlots
using Test, Random

include("w_ansatz.jl")
include("pooling.jl")

# architecture: feature_map |> w_ansatz |> pool |> w_ansatz |> pool ...

# read in data
two_d_data = CSV.File("2D Classification Data.csv") |> DataFrame
two_d_data = two_d_data[!, Not(:Column1)]

train = two_d_data[!, Not(:label)]
feature_1 = train[!,1]
feature_2 = train[!,2]

labels = two_d_data[!,3]

# set number of qubits
N = 8

# define feature map - Rz(arcsin(x) + arcsin(x)) + Ry(arcsin(x) + arcsin(x))
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(2*asin(x))) for i=1:n]), chain(n, [put(i=>Ry(2*acos(t))) for i=1:n]))
end

# observable
zz = chain(N, prod([put(N,i=>Z) for i=4:4:8]))
sz1 = chain(N, put(8=>Z))

#---

function QCNN(N, θ_plus, θ_minus)
    """
    Number of qubits must be a a power of 2
    """

    n_layers = log10(N)/log10(2)

    layer1 = chain(N)
    push!(layer1, w_circuit(N, 2)) # conv1
    push!(layer1, chain(N, prod([put(N, (i, i+1) => pool(θ_plus, θ_minus)) for i in 1:2:(N - 1)]))) # pool1
    #rho1 = statevec(layer1)* (statevec(layer1))'

    #rho1 = ψ |> layer1

    #focus!(layer1,(2,4,6,8))
    #layer2 = chain(N)
    push!(layer1, w_circuit2(N, 2)) # conv1
    push!(layer1, chain(N, prod([put(N, (i, i+2) => pool(θ_plus, θ_minus)) for i in 2:4:(N - 2)]))) # pool2

	#layer3 = chain(N)
	#push!(layer3, )

    #cirq = chain(N, put((1:N)=>layer1), put((1:N)=>layer2))

end

#---

statetest = rand_state(8)

stest = copy(statetest) |> QCNN(8, 0.1, 0.2)

statevec(stest)
rho = statevec(stest) * (statevec(stest))'

density_matrix(zero_state(2))

r = rand_state(4)
focus!(r, (2,4))
r |> chain(2, put(2=>X))

#---

Uθ = QCNN(8, 0.1, 0.2)
dispatch!(Uθ, :random)
params = parameters(Uθ)
optimizer = Adam(lr=0.01)
loss = 0. # total cost
loss_vec = Vector{Float64}()
grad_params_sum = zeros(length(params)) # vector of derivatives
niter = 200
cost = sz1

# plot circuit with YaoPlots
plot(Uθ)

optimize(optim_func, params, LBFGS())

((expect(cost, (zero_state(N) |> feature_map(N, feature_1[1], feature_2[1]) |> Uθ)) - labels[1]) |> real)

zero_state(N) |> feature_map(N, feature_1[1], feature_2[1]) |> Uθ


p = Optim.minimizer(optimize(x->((expect(cost, zero_state(N) |> feature_map(N, feature_1[2], feature_2[2]) => dispatch!(Uθ, x)) - labels[2]) |> real),
		parameters(Uθ),
		LBFGS(),
		Optim.Options(iterations=niter)))

parameters(Uθ)
(expect(cost, (zero_state(N) |> feature_map(N, feature_1[2], feature_2[2])) |> Uθ) |> real)

begin
    grad_params = zeros(length(params)); # vector of derivatives
    for i=1:length(labels)
        # calculate sum of gradients
        #optim_func = (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> dispatch!(Uθ, θ)) - labels[i]) |> real # gradient of <C>
        #grad_params += 2. * ((expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) => Uθ) - labels[i]) |> real) * dCdθ; # full loss function grads
		#optimize(x->((expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) |> dispatch!(Uθ, x))) - labels[i]) |> real), LBFGS())
		Optim.minimizer(optimize(x->((expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ, x)) - labels[i]) |> real),
	            parameters(Uθ),
	            LBFGS(),
	            Optim.Options(iterations=1)))
    end
    # feed the gradients into the circuit
    #dispatch!(Uθ, new_params)
    loss = 0.; # total cost
    # for i=1:length(labels)
    #     loss += (-1 * labels[i] * log(expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ)) |> real); # binary cross entropy
    #     #loss += (expect(cost, (zero_state(N) |> feature_map(N, train[i])) |> Uθ) - labels[i])^2 |> real; # L2 loss
    # end
    # loss = loss * (1/length(labels))
    # println("Step $j, loss = $loss "); flush(Core.stdout)
    # append!(loss_vec, loss)
end

plot(epoch, loss_vec, xaxis=("Epoch"), yaxis=("Loss"))~

new_params = parameters(Uθ)

solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ) |> real))
end

#---

solution
plot!(feature_1, solution, seriestype=:scatter)
labels
function classifier(solution)
    predictions = Vector{Int64}()
    for i=1:length(solution)
        if solution[i] >= -0.8
            append!(predictions, 0)
        elseif solution[i] < -0.8
            append!(predictions, 1)
        end
    end
    predictions
end

predictions = classifier(solution)
scatter()
gdf = groupby(two_d_data, :3)
plot!(gdf[2].x, gdf[2].y, seriestype=:scatter, color=:red, label='0')
plot!(gdf[1].x, gdf[1].y, seriestype=:scatter, color=:blue, label='1')

pred_df = DataFrame(x=feature_1, y=feature_2, predictions=predictions)
gdf2 = groupby(pred_df,:3);
plot!(gdf2[1].x, gdf2[1].y, seriestype=:scatter, color=:green, label='0')
plot!(gdf2[2].x, gdf2[2].y, seriestype=:scatter, color=:orange, label='1')


begin
	Alices_and_Bobs_entangled_qubits = ArrayReg(bit"00") + ArrayReg(bit"11") |> normalize!
	Alicequbit = rand_state(1) #This function creates a qubit with a random state.
	state(Alicequbit)
end

state(Alices_and_Bobs_entangled_qubits)

begin
	teleportationcircuit = chain(3, control(1,2=>X), put(1=>H))
end

begin
	feeding = join(Alices_and_Bobs_entangled_qubits, Alicequbit) |> teleportationcircuit
	state(feeding)
end

Alices_measuredqubits = measure!(RemoveMeasured(), feeding, 1:2)

if(Alices_measuredqubits == bit"00")
	Bobs_qubit = feeding
elseif(Alices_measuredqubits == bit"01")
	Bobs_qubit = feeding |> chain(1, put(1=>Z))
elseif(Alices_measuredqubits == bit"10")
	Bobs_qubit = feeding |> chain(1, put(1=>X))
else
	Bobs_qubit = feeding |> chain(1, put(1=>Y))
end

state(Bobs_qubit)

[state(Alicequbit) state(Bobs_qubit)]


operator

optimize(x->-operator_fidelity(u, dispatch!(ansatz, x)),(G, x) -> (G .= -operator_fidelity'(u, dispatch!(ansatz, x))[2]), parameters(ansatz), LBFGS(),Optim.Options(iterations=niter))

"""
    learn_u4(u::AbstractMatrix; niter=100)

Learn a general U4 gate. The optimizer is LBFGS.
"""
function learn_u4(u::AbstractBlock; niter=100)
    ansatz = general_U4()
    params = parameters(ansatz)
    println("initial loss = $(operator_fidelity(u,ansatz))")
    optimize(x->-operator_fidelity(u, dispatch!(ansatz, x)),
            parameters(ansatz),
            LBFGS(),
            Optim.Options(iterations=niter))
    println("final fidelity = $(operator_fidelity(u,ansatz))")
    return ansatz, operator_fidelity(u,ansatz)
end

using Random
Random.seed!(2)
u = matblock(rand_unitary(4))
c = learn_u4(u; niter=15)

rand_unitary(4)

plot(general_U4())

a -> a^2
opti
