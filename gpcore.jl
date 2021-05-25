module GPcore
include("./setup.jl")
using .Const, Random, LinearAlgebra, Distributions, Base.Threads

mutable struct Trace{T<:AbstractArray, S<:Complex}
    xs::Vector{T}
    ys::Vector{S}
    iKu::Vector{S}
    iΣ::Array{S}
end

function makedata(xs::Vector{Vector{Float32}}, ys::Vector{Complex{Float32}})
    # Step 1
    zs = [rand([1f0, -1f0], Const.dim) for i in 1:Const.auxn]
    KMM = covar(zs)
    KMN = [kernel(zs[i], xs[j]) for i in 1:length(zs), j in 1:length(xs)]

    # Step 2
    Λ = Diagonal([kernel(xs[i], xs[i]) + KMN[:, i]' * (KMM \ KMN[:, i]) + 1f-6 for i in 1:length(xs)])

    # Step3
    QMM = KMM + KMN * (Λ \ KMN')
    û = KMM * (QMM \ (KMN * (Λ \ exp.(ys))))
    Σ̂ = KMM * (QMM \ KMM)
    iKu = KMM \ û
    iΣ  = inv(Σ̂)

    # Output
    Trace(xs, ys, iKu, iΣ)
end

function model(trace::Trace, x::Vector{Float32})
    xs, ys, iKu, iΣ = trace.xs, trace.ys, trace.iKu, trace.iΣ

    # Compute mu var
    kv = [kernel(xs[i], x) for i in 1:Const.auxn]
    k0 = kernel(x, x)
    mu = kv' * iKu
    var = k0 - kv' * iΣ * kv

    # sample from gaussian
    log.(var * randn(Complex{Float32}) + mu)
end

function kernel(x::Vector{Float32}, y::Vector{Float32})
    r = norm(x - y) / 2f0 / Const.dim
    Const.θ₁ * exp(-2f0 * π * r^2 / Const.θ₂)
end

function covar(xs::Vector{Vector{Float32}})
    n = length(xs)
    K = zeros(Complex{Float32}, n, n)
    I = Diagonal(ones(Float32, n))
    for j in 1:n
        y = xs[j]
        for i in 1:n
            x = xs[i]
            K[i, j] = kernel(x, y)
        end
    end
    return K + 1f-6 * I
end
end
