using Yao, YaoExtensions
using Optim
using QuAlgorithmZoo: Adam, update!
using DataFrames, CSV
using Plots
using LIBSVM

include("w_ansatz.jl")

"""
    fully connected quantum convolutional neural network with no pooling layers

    architecture: feature_map |> w_ansatz |> w_ansatz... |> measure on all qubits

    following architecure from arXiv:2011.02966v1

"""

# read in data
two_d_data = CSV.File("2D Classification Data.csv") |> DataFrame
two_d_data = two_d_data[!, Not(:Column1)]

train = two_d_data[!, Not(:label)]
feature_1 = train[!,1]
feature_2 = train[!,2]

labels = two_d_data[!,3]

#---

# define feature map
function feature_map(n, x, t)
    return chain(n, chain(n, [put(i=>Rz(3*acos(x))) for i=1:n]), chain(n, [put(i=>Ry(2*asin(t))) for i=1:n]))
end

# cost function
magn = chain(N, prod([put(N, i=>Z) for i=1:N]))
sz1 = chain(N, put(N=>Z))

#---

function FC_QCNN(N, θ_plus, θ_minus)

    cirq = chain(N)
    push!(cirq, w_circuit(N, 4)) # conv1

end

#---

# assign paramters to QCNN
Uθ_FC = FC_QCNN(8, 0.1, 0.2)
dispatch!(Uθ_FC, :random)
params = parameters(Uθ_FC)
loss = 0. # total cost
loss_vec = Vector{Float64}()
grad_params_sum = zeros(length(params)) # vector of derivatives
niter = 200
cost = sz1

plot(Uθ_FC)

#---

# train the model
begin
    for i=1:length(labels)
		optimize(x->(-1* labels[i] * log(expect(cost, zero_state(N) |> feature_map(N, feature_1[i], feature_2[i]) => dispatch!(Uθ_FC, x))) |> real),
	            parameters(Uθ_FC),
	            LBFGS(),
	            Optim.Options(iterations=1))
    end
end

new_params = parameters(Uθ_FC)
solution = Vector{Float64}()
for i=1:length(labels)
    dispatch!(Uθ_FC, new_params)
    append!(solution, (expect(cost, (zero_state(N) |> feature_map(N, feature_1[i], feature_2[i])) |> Uθ_FC) |> real))
end
solution

plot(feature_1, solution, seriestype=:scatter)
labels

begin
	gdf = groupby(two_d_data, :3)
	plot(gdf[2].x, gdf[2].y, seriestype=:scatter, color=:red, label='0')
	plot!(gdf[1].x, gdf[1].y, seriestype=:scatter, color=:blue, label='1')

	pred_df = DataFrame(x=feature_1, y=feature_2, predictions=predictions)
	gdf2 = groupby(pred_df,:3);
	plot!(gdf2[1].x, gdf2[1].y, seriestype=:scatter, color=:green, label='0')
	plot!(gdf2[2].x, gdf2[2].y, seriestype=:scatter, color=:orange, label='1')
end

model = svmtrain(solution', labels)
predictions = svmpredict(model, solution')[1]

"""
	now evaluate the model using F1 score
"""
using EvalMetrics

cm = EvalMetrics.ConfusionMatrix(labels, predictions) # confusion matrix

accuracy = EvalMetrics.accuracy(cm)
f1 = f1_score(cm)
