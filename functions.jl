include("./setup.jl")
include("./model.jl")
using LinearAlgebra, Random

struct Flip{T<:Real}
    flip::Vector{Diagonal{T}}
end
function Flip()
    flip = Vector{Diagonal{Float32}}(undef, c.N)
    for i in 1:c.N
        o = Diagonal(ones(Float32, c.N))
        o[i, i] *= -1f0
        flip[i] = o
    end
    Flip(flip)
end
const a = Flip()

function update!(model::GPmodel, x::Vector{Float32})
    n = length(x)
    rng = MersenneTwister(1234)
    randomnum = rand(rng, Float32, n)
    @inbounds for ix in 1:n
        x₁ = x[ix]
        y  = inference(model, x)
        yflip = inference(model, a.flip[ix] * x)
        prob  = exp(2f0 * real(yflip - y))
        x[ix] = ifelse(randomnum[ix] < prob, -x₁, x₁)
    end
end

function hamiltonian_heisenberg(x::Vector{Float32}, y::Complex{Float32}, 
                                model::GPmodel, ix::Integer)
    out = 0f0im
    ixnext = ix%c.N + 1
    if x[ix] * x[ixnext] < 0f0
        yflip = infelence(model, a.flip[ixnext] * a.flip[ix] * x)
        out  += 2f0 * exp(yflip - y) - 1f0
    else
        out += 1f0
    end
    return -c.J * out / 4f0
end

function energy_heisenberg(x::Vector{Float32}, y::Complex{Float32}, model::GPmodel)
    out = 0f0im
    for ix in 1:c.N
        out += hamiltonian_heisenberg(x, y, model, ix)
    end
    return out
end

function hamiltonian_ising(x::Vector{Float32}, y::Complex{Float32}, 
                           model::GPmodel, iy::Integer)
    out = 0f0im
    iynext = iy%c.N + 1
    yflip = inference(model, a.flip[iynext] * x)
    out  += x[iy] * x[iynext] / 4f0 + c.h * exp(yflip - y) / 2f0
    return -out
end

function energy_ising(x::Vector{Float32}, y::Complex{Float32}, model::GPmodel)
    out = 0f0im
    for iy in 1:c.N
        out += hamiltonian_ising(x, y, model, iy)
    end
    return out
end

function hamiltonian_XY(x::Vector{T}, y::Complex{T}, model::GPmodel, iy::Integer) where {T <: Real}
    out = 0f0im
    iynext = iy%c.N + 1
    if x[iy] * x[iynext] < 0f0
        yflip = inference(model, a.flip[iynext] * a.flip[iy] * x)
        out  += exp(yflip - y)
    end
    return c.t * out
end

function energy_XY(x::Vector{T}, y::Complex{T}, model::GPmodel) where {T <: Real}
    out = 0f0im
    for iy in 1:c.N
        out += hamiltonian_XY(x, y, model, iy)
    end
    return out
end

function energy(x::Vector{Float32}, y::Complex{Float32}, model::GPmodel)
    e = 0f0im
#    e = energy_ising(x, y, model)
#    e = energy_heisenberg(x, y, model)
    e = energy_XY(x, y, model)
    return e
end

