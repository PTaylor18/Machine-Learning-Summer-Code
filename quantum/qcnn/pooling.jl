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
