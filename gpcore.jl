module GPcore
include("./setup.jl")
using .Const, Random, LinearAlgebra, Distributions, Base.Threads

mutable struct Trace{T<:AbstractArray, S<:Complex}
    xs::Vector{T}
    ys::Vector{S}
    invK::Array{S}
end

function model(trace::Trace, x::Vector{Float32})
    xs, ys, invK = trace.xs, trace.ys, trace.invK

    # Compute mu var
    mu, var = statcalc(xs, ys, invK, x)

    # sample from gaussian
    y = log.(var * randn(Complex{Float32}) + mu)
    return y
end

function kernel(x::Vector{Float32}, y::Vector{Float32})
    r = norm(x - y) / 2f0 / Const.dim
    Const.θ₁ * exp(-2f0 * π * r^2 / Const.θ₂)
end

function covar(xs::Vector{Vector{Float32}})
    n = length(xs)
    K = zeros(Complex{Float32}, n, n)
    for j in 1:n
        y = xs[j]
        for i in 1:n
            x = xs[i]
            K[i, j] = kernel(x, y)
        end
    end
    return K
end

function statcalc(xs::Vector{Vector{Float32}}, ys::Vector{Complex{Float32}}, 
                  invK::Array{Complex{Float32}}, x::Vector{Float32})
    kv = [kernel(xs[i], x) for i in 1:length(xs)]
    k0 = kernel(x, x)

    mu = transpose(kv) * invK * exp.(ys)
    var = abs(k0 - transpose(kv) * invK * kv)
    return  mu, var
end

end
