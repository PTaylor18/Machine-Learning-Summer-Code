using Yao, YaoExtensions

function Rg(i)
    return chain(
        put(i=>Rz(0)),
        put(i=>Ry(0)),
        put(i=>Rz(0)))
end

function w_gate(i, j)
    return chain(
        Rg(i),
        Rg(j),
        cnot(j, i),
        put(i=>Rz(0)),
        put(j=>Ry(0)),
        cnot(i, j),
        put(j=>Ry(0)),
        cnot(j, i),
        Rg(i),
        Rg(j))
end

function w_layer(n, offset)
    return chain(n,
        w_gate(i, i + 1) for i in (1 + offset):2:(n - 1))
end

function w_layer2(n, offset)
    return chain(n,
        w_gate(i, i + 2) for i in (2 + offset):4:(n-2))
end

function f_layer(n)
    return chain(n,
        w_gate(i, i + 4) for i in 4:4:(n-4))
end

function w_circuit(n, depth)
    return chain(n,
        chain(w_layer(n, (layer - 1) % 2) for layer in 1:depth))
end

function w_circuit2(n, depth)
    return chain(n,
        chain(w_layer2(n, ((-1)^layer +1)) for layer in 1:depth))
end

function f_circuit(n, depth)
    f = chain(n)
    for i in 1:depth
        push!(f, chain(f_layer(n)))
    end
    return f
end

function w_circuit_full(n, layer=1, depth=1, final_layer=false)
    if layer == 1
        return chain(n,
            chain(w_layer(n, (layer - 1) % 2) for layer in 1:depth))

    elseif layer == 2
        return chain(n,
            chain(w_layer2(n, ((-1)^layer +1)) for layer in 1:depth))

    elseif final_layer == true
        f = chain(n)
        for i in 1:depth
            push!(f, chain(f_layer(n)))
        end
        return f
    end
end
