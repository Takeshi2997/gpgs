module Func
include("./gpcore.jl")
include("./setup.jl")
using .Const, .GPcore, LinearAlgebra, Random, Distributions

struct Flip
    flip::Vector{Diagonal{Float32}}
end
function Flip()
    flip = Vector{Diagonal{Float32}}(undef, Const.dim)
    for i in 1:Const.dim
        o = Diagonal(ones(Float32, Const.dim))
        o[i, i] *= -1f0
        flip[i] = o
    end
    Flip(flip)
end
const a = Flip()

function update(trace::GPcore.Trace)
    xs, ys = trace.xs, trace.ys
    x = xs[end]
    y = ys[end]
    n = length(x)
    rng = MersenneTwister(1234)
    randomnum = rand(rng, Float32, n)
    for ix in 1:n
        xflip = copy(x)
        xflip[ix] *= -1f0
        yflip = GPcore.model(trace, xflip)
        prob  = exp(2f0 * real(yflip - y))
        a = x[ix]
        if randomnum[ix] < prob
            x = xflip
            y = yflip
        end
    end
    return x, y
end

function update(trace::GPcore.Trace)
    xs, ys = trace.xs, trace.ys
    x = xs[end]
    y = ys[end]
    n = length(x)
    rng = MersenneTwister(1234)
    randomnum = rand(rng, Float32, n)
    for ix in 1:n
        xflip = copy(x)
        xflip[ix] *= -1f0
        yflip = GPcore.model(trace, xflip)
        prob  = exp(2f0 * real(yflip - y))
        a = x[ix]
        if randomnum[ix] < prob
            x = xflip
            y = yflip
        end
    end
    return x, y
end

function hamiltonian_heisenberg(x::Vector{Float32}, y::Complex{Float32}, ix::Integer)
    out = 0f0im
    ixnext = ix%Const.dim + 1
    if x[ix] != x[ixnext]
        yflip = GPcore.func(a.flip[ixnext] * a.flip[ix] * x)
        out  += 2f0 * exp(yflip - y) - 1f0
    else
        out += 1f0
    end
    return -Const.J * out / 4f0
end

function energy_heisenberg(x::Vector{Float32}, y::Complex{Float32})
    out = 0f0im
    for ix in 1:Const.dim
        out += hamiltonian_heisenberg(x, y, ix)
    end
    return out
end


function hamiltonian_heisenberg(x::Vector{Float32}, y::Complex{Float32}, 
                                trace::GPcore.Trace, ix::Integer)
    out = 0f0im
    ixnext = ix%Const.dim + 1
    if x[ix] != x[ixnext]
        yflip = GPcore.model(trace, a.flip[ixnext] * a.flip[ix] * x)
        out  += 2f0 * exp(yflip - y) - 1f0
    else
        out += 1f0
    end
    return -Const.J * out / 4f0
end

function energy_heisenberg(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    out = 0f0im
    for ix in 1:Const.dim
        out += hamiltonian_heisenberg(x, y, trace, ix)
    end
    return out
end

function hamiltonian_ising(x::Vector{Float32}, y::Complex{Float32}, 
                           trace::GPcore.Trace, iy::Integer)
    out = 0f0im
    iynext = iy%Const.dim + 1
    yflip = GPcore.model(trace, a.flip[iynext] * x)
    out  += x[iy] * x[iynext] / 4f0 + Const.h * exp(yflip - y) / 2f0
    return -out
end

function energy_ising(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    out = 0f0im
    for iy in 1:Const.dim
        out += hamiltonian_ising(x, y, trace, iy)
    end
    return out
end

function energy(x::Vector{Float32}, y::Complex{Float32}, trace::GPcore.Trace)
    e = 0f0im
    e = energy_ising(x, y, trace)
#    e = energy_heisenberg(x, y, trace)
    return e
end

function imaginary_evolution(trace::GPcore.Trace)
    xs, ys, invK = trace.xs, trace.ys, trace.invK
    xs′ = copy(xs)
    ys′ = copy(ys)
    for n in 1:Const.init
        x = xs[n]
        y = ys[n]
        e = energy(x, y, trace)
        ys′[n] = log((1f0 - Const.Δτ .* e / Const.dim) * exp(y))
    end 
    outtrace = GPcore.Trace(xs′, ys′, invK)
    return outtrace
end

end
