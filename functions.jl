include("./setup.jl")
include("./model.jl")
using LinearAlgebra, Random

struct Flip{T<:Real}
    flip::Vector{Diagonal{T}}
end
function Flip()
    flip = Vector{Diagonal{Float64}}(undef, c.nspin)
    for i in 1:c.nspin
        o = Diagonal(ones(Float64, c.nspin))
        o[i, i] *= -1.0
        flip[i] = o
    end
    Flip(flip)
end
const a = Flip()

function update!(model::GPmodel, x::State)
    pos = rand(collect(1:c.nspin))
    xflip_spin = copy(x.spin)
    xflip_spin[pos] *= -1
    xflip = State(xflip_spin)
    y = predict(model, x)
    yflip = predict(model, xflip)
    x.spin[pos] *= ifelse(rand(eng) < exp(2 * real(yflip - y)), -1.0, 1.0)
end

function energy_ising(x::State, y::Complex{T}, model::GPmodel) where {T<:Real}
    out = 0.0im
    for iy in 1:c.nspin
        iynext = iy%c.nspin + 1
        xflip_spin = a.flip[iynext] * x.spin
        xflip = State(xflip_spin)
        yflip = predict(model, xflip)
        out  += -x.spin[iy] * x.spin[iynext] / 4.0 - c.h * exp(yflip - y) / 2.0
    end
    return out
end

function energy(x::State, y::Complex{T}, model::GPmodel) where {T<:Real}
    e = energy_ising(x, y, model)
    return e
end

