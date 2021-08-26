using Yao
using YaoBase

export PoolingUnitary, pool

"""
    Pooling Layer:
    projected rotations acting as a 1 qubit measurement controlling a unitary of the neighbor qubit
    therefore mapping a 2 qubit Hilbert space to a 1 qubit Hilbert space
    from  arXiv:2011.02966v1

    𝜌^(i+1) = tr_A[I_(i+1) * (𝜌^i)^(⊗2) * I†_(i+1)]

    I_(i+1) = (|+><+| ⊗ exp((-im * θ_plus) * Ŷ))) + (|-><-| ⊗ exp((-im * θ_minus) * Ŷ)))
"""

mutable struct PoolingUnitary{T} <: PrimitiveBlock{2}
    theta_plus::T
    theta_minus::T
end

pool(θ_plus::Real, θ_minus::Real) = PoolingUnitary(θ_plus, θ_minus)
pool_dag(θ_plus::Real, θ_minus::Real) = Daggered(PoolingUnitary(θ_plus, θ_minus))

function pooling_matrix(gate)
    MA_1 = 0.5 * Matrix([[1,1] [1,1]])
    MA_2 = 0.5 * Matrix([[1,-1] [-1,1]])

    MB_1 = Matrix([[cos(gate.theta_plus*0.5), sin(gate.theta_plus*0.5)] [-1*sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5)]])
    MB_2 = Matrix([[cos(gate.theta_minus*0.5), sin(gate.theta_minus*0.5)] [-1*sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5)]])

    matrix = kron(MA_1, MB_1) + kron(MA_2, MB_2)
    return matrix
end

# (0.5*Matrix([[cos(gate.theta_plus*0.5), sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5), sin(gate.theta_plus*0.5)] [-1*sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5), -1*sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5)] [cos(gate.theta_plus*0.5), sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5), sin(gate.theta_plus*0.5)] [-1*sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5), -1*sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5)]])
#                                                         + 0.5*Matrix([[cos(gate.theta_minus*0.5), sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5), sin(gate.theta_minus*0.5)] [-1*sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5), -1*sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5)] [-1*cos(gate.theta_minus*0.5), -1*sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5), sin(gate.theta_minus*0.5)] [sin(gate.theta_minus*0.5), -1*cos(gate.theta_minus*0.5), -1*sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5)]]))

Yao.mat(::Type{T}, gate::PoolingUnitary) where {T} = kron(im*-im*0.5 * Matrix([[1,1] [1,1]]), Matrix([[cos(gate.theta_plus*0.5), sin(gate.theta_plus*0.5)] [-1*sin(gate.theta_plus*0.5), cos(gate.theta_plus*0.5)]])) + kron(0.5 * Matrix([[1,-1] [-1,1]]), Matrix([[cos(gate.theta_minus*0.5), sin(gate.theta_minus*0.5)] [-1*sin(gate.theta_minus*0.5), cos(gate.theta_minus*0.5)]]))

Base.:(==)(rb1::PoolingUnitary, rb2::PoolingUnitary) = rb1.theta_plus == rb2.theta_plus && rb1.theta_minus == rb2.theta_minus
Base.copy(gate::PoolingUnitary{T}) where T = PoolingUnitary{T}(gate.theta_plus, gate.theta_minus)
Yao.dispatch!(gate::PoolingUnitary, params::Vector) = ((gate.theta_plus, gate.theta_minus) = params; gate)

Yao.getiparams(gate::PoolingUnitary) = (gate.theta_plus, gate.theta_minus)
function Yao.setiparams!(gate::PoolingUnitary, θ_plus::Real, θ_minus::Real)
    gate.theta_plus, gate.theta_minus = θ_plus, θ_minus
    gate
end
Yao.niparams(::Type{<:PoolingUnitary}) = 2
Yao.niparams(::PoolingUnitary) = 2
Yao.render_params(gate::PoolingUnitary, ::Val{:random}) = rand()*π, rand()*2π

function YaoBase.isunitary(gate::PoolingUnitary)
    isreal(gate.theta_plus) && isreal(gate.theta_minus) && return true
    @warn "θ or ϕ in aswap(θ, ϕ) is not real, fallback to matrix-based method"
    return isunitary(Yao.mat(gate))
end

# function Base.hash(gate::PoolingUnitary, h::UInt)
#     hash(hash(gate.theta_plus, gate.theta_minus, objectid(gate)), h)
# end
#
# Yao.cache_key(gate::PoolingUnitary) = (gate.theta_plus, gate.theta_minus)


#---

"""
Now lets do some tests
"""
pool(0.1,0.2)

println(mat(pool(0.1,0.2)))

r = rand_state(2)
r |> pool(0.1,0.2)
println(statevec(r))

r0 = zero_state(2)
r0 |> pool(0., 0.)

println(statevec(r0))

# simple dispatch without circuit is difficult to get
dispatch!(pool(0.,0.), :random)

circ = chain(2, put((1, 2) => pool(0., 0.)))
println("Original unitary matrix: ", mat(circ))

dispatch!(circ, :random)
println("Dispatched unitary matrix: ", mat(circ))

circ3a = chain(3, put(3, (1, 2) => pool(0., 0.)))
circ3b = chain(3, put((1, 3) => pool(0., 0.)))
mat(circ3a) ≈ mat(circ3b)
println("3Q unitary matrix: ", mat(circ3a))
println("3Qunitary matrix: ", mat(circ3b))


I = Matrix([[1,0] [0, 1]])

P0 = 0.5*(I+mat(Z))
P1 = 0.5*(I-mat(Z))

P01 = Matrix([[0,0] [1,0]])
P10 = Matrix([[0,1] [0,0]])

circtest = chain(4, put((1, 2) => pool(0.1, 0.2)), put((3, 4) => pool(0.1, 0.2)))

rtest = zero_state(4)


rpool = copy(rtest) |> circtest

focus!(rpool, (1,2))

sv = statevec(partial_tr(rpool,1))

rho = sv * sv'

P0_expect = sv'*(P0*sv)
P1_expect = sv'*(P1*sv)

P01_expect = sv'*(P01*sv)
P10_expect = sv'*(P10*sv)

statevec(partial_tr(pooled, 2))

# example of defining the ansatz
function vqe_ansatz(nqbt::Int, dpth::Int)
    cirq = chain(nqbt)
    for i=1:nqbt
        push!(cirq, put(nqbt, i=>chain(Rx(0.),Rz(0.))) )
    end
    for d=1:dpth
        for i=1:2:nqbt
            push!(cirq, put(nqbt, (i, i%nqbt+1) => pool(0., 0.)) )
        end
        for i=2:2:nqbt
            push!(cirq, put(nqbt, (i, i%nqbt+1) => pool(0., 0.)) )
        end
    end
    return cirq
end

# example showing that dispatch works for a standard circuit
println(vqe_ansatz(2,1))
parameters(vqe_ansatz(2,1))
mat(dispatch!(vqe_ansatz(2,1), :random))
parameters(vqe_ansatz(2,1))

mat(dispatch!(vqe_ansatz(2,1), [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]))
