using Flux, MNIST
using ImageTransformations

raw_images = Flux.Data.MNIST.images()
raw_labels = Flux.Data.MNIST.labels()


raw_images[1]
raw_labels[1]

foo = raw_images[1]
foo = imresize(foo, (8,8))
foo = float(reshape(foo,:))
typeof(reshape(foo,:))

convert(Array{Float64},foo)

foo2 = Array{Float64,1}()



data_x = Vector{Float64}[]
data_y = Vector{Int64}()
data_ind = Int[]
for i = 1:60000
    if raw_labels[i] < 2
        push!(data_x, Array{Float64,1}(float(reshape(imresize(raw_images[i], (8, 8)),:))))
        push!(data_y, raw_labels[i])
        push!(data_ind, i)
    end
end

println(data_x[1])
data_y
data_ind
