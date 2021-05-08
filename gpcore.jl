module GPcore
include("./setup.jl")
using .Const, Random, LinearAlgebra, Distributions, Base.Threads

mutable struct Trace{T <: AbstractArray, S <: Complex}
    xs::Vector{T}
    ys::Vector{S}
end

function model(trace::Trace, x::Vector{Float32})
    xs, ys = trace.xs, trace.ys

    # Compute mu var
    realmu, imagmu, var = statcalc(xs, ys, x)

    # sample from gaussian
    y = rand(Normal(realmu, var)) + im * rand(Normal(imagmu, var))
    return y
end

kernel(x::Vector{Float32}, y::Vector{Float32}) = Const.θ₁ * exp(-(norm(x - y))^2 / Const.θ₂)

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

function statcalc(xs::Vector{Vector{Float32}}, ys::Vector{Complex{Float32}}, x::Vector{Float32})
    K  = covar(xs)
    kv = [kernel(xs[i], x) for i in 1:length(xs)]
    k0 = kernel(x, x)
    realy = real.(ys)
    imagy = imag.(ys)
    
    # Calculate inverse K
    U, Δ, V = svd(K)
    invΔ = Diagonal(1f0 ./ Δ .* (Δ .> 1f-6))
    invK = V * invΔ * U'
 
    realmu = kv' * invK * realy
    imagmu = kv' * invK * imagy
    var = abs(k0 - kv' * invK * kv)
    return  realmu, imagmu, var
end

end
