module GPcore
include("./setup.jl")
using .Const, Random, LinearAlgebra, Distributions, Base.Threads

mutable struct Trace{T<:AbstractArray, S<:Complex, U<:Real}
    xs::Vector{T}
    ys::Vector{S}
    invK::Array{U}
end

function model(trace::Trace, x::Vector{Float32})
    xs, ys, invK = trace.xs, trace.ys, trace.invK

    # Compute mu var
    realmu, imagmu, var = statcalc(xs, ys, invK, x)

    # sample from gaussian
    y = rand(Normal(realmu, var)) + im * rand(Normal(imagmu, var))
    return y
end

kernel(x::Vector{Float32}, y::Vector{Float32}) = Const.θ₁ * exp(-(norm(x - y))^2 / Const.θ₂)
const I = Diagonal(ones(Float32, Const.init))

function covar(xs::Vector{Vector{Float32}})
    n = length(xs)
    K = zeros(Float32, n, n)
    for j in 1:n
        y = xs[j]
        for i in 1:n
            x = xs[i]
            K[i, j] = kernel(x, y)
        end
    end
    return K
end

function statcalc(xs::Vector{Vector{Float32}}, ys::Vector{Complex{Float32}}, invK::Array{Float32}, x::Vector{Float32})
    kv = [kernel(xs[i], x) for i in 1:length(xs)]
    k0 = kernel(x, x)
    realy = real.(ys)
    imagy = imag.(ys)
    
    # Calculate inverse K
    realmu = kv' * invK * realy
    imagmu = kv' * invK * imagy
    var = abs(k0 - kv' * invK * kv)
    return  realmu, imagmu, var
end

end
