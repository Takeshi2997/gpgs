include("./setup.jl")
using LinearAlgebra

mutable struct GPmodel{T<:AbstractArray, S<:Complex}
    xs::Vector{T}
    ys::Vector{S}
    iKu::Vector{S}
    iΣ::Array{S}
end

function makemodel(xs::Vector{Vector{T}}, ys::Vector{Complex{T}}) where {T<:Real}
    # Step 1
    zs = [rand([1f0, -1f0], c.N) for i in 1:c.auxn]
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
    GPmodel(xs, ys, iKu, iΣ)
end

function inference(model::GPmodel, x::Vector{T}) where {T<:Real}
    xs, ys, iKu, iΣ = model.xs, model.ys, model.iKu, model.iΣ

    # Compute mu var
    kv = [kernel(xs[i], x) for i in 1:c.auxn]
    k0 = kernel(x, x)
    mu = kv' * iKu
    var = k0 - kv' * iΣ * kv

    # sample from gaussian
    log.(sqrt(var) * randn(Complex{T}) + mu)
end

const I = Array(Diagonal(ones(Float32, c.num)))

struct CircshiftArrays{T<:AbstractArray}
    cshift::Vector{T}
end
function CircshiftArrays()
    cshift = Vector{Array{Float32}}(undef, c.N)
    for i in 1:c.N
        o = Array(Diagonal(ones(Float32, c.N)))
        cshift[i] = circshift(o, i-1)
    end
    CircshiftArrays(cshift)
end
const b = CircshiftArrays()

function distance(x::Vector{T}, y::Vector{T}) where {T<:Real}
    A = sum(y .* b.cshift)
    B = sum(x .* b.cshift)
    1f0 - maximum(A' * x + B' * y) / 2f0 / c.N
end

function kernel(x::Vector{T}, y::Vector{T}) where {T<:Real}
    r = distance(x, y)
    c.θ₁ * exp(-r^2 / c.θ₂)
end

function covar(xs::Vector{Vector{T}}) where {T<:Real}
    n = length(xs)
    K = zeros(Complex{Float32}, n, n)
    I0 = Diagonal(ones(Float32, n))
    for j in 1:n
        y = xs[j]
        for i in 1:n
            x = xs[i]
            K[i, j] = kernel(x, y)
        end
    end
    return K + 1f-6 * I0
end
