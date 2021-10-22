using Yao, YaoExtensions

function pool_measure(θ_plus, θ_minus)

    cirq = rand_state(2)
    measure!(RemoveMeasured(), cirq, 1)

    if (pc == bit"0")
        push!(pc, chain(N, put(N, 2 => Ry(θ_plus))))
    elseif (pc ==bit"1")
        push!(pc, chain(N, put(N, 2 => Ry(θ_minus))))
    end
    return cirq
end

pool_measure(0.1,0.2)

statevec(pool_measure(0.1,0.2))
