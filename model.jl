mutable struct State{T<:Real}
    spin::Vector{T}
end

# mutable struct State{T<:Real}
#     spin::Vector{T}
#     shift::Vector{Vector{T}}
# end
# function State(x::Vector{T}) where {T<:Real}
#     shift = [circshift(x, s) for s in 1:c.nspin]
#     State(x, shift)
# end

mutable struct GPmodel{T<:Complex}
    xs::Vector{State}
    ys::Vector{T}
    pvec::Vector{T}
    iK::Array{T}
end
function GPmodel(xs::Vector{State}, ys::Vector{T}) where {T<:Complex}
    iK = Array{T}(undef, c.ndata, c.ndata)
    makeinverse(iK, xs)
    pvec = iK * ys
    GPmodel(xs, ys, pvec, iK)    
end

function kernel(x1::State, x2::State)
    v = norm(x1.spin - x2.spin)
    v /= c.nspin
    c.θ₁ * exp.(-v / c.θ₂)
end

# function kernel(x1::State, x2::State)
#     v = [norm(x1.shift[n] - x2.spin) for n in 1:length(x1.spin)]
#     v ./= c.nspin
#     sum(c.θ₁ * exp.(-v ./ c.θ₂))
# end

function makeinverse(iK::Array{T}, data_x::Vector{State}) where {T<:Complex}
    for i in 1:c.ndata
        for j in i:c.ndata
            iK[i, j] = kernel(data_x[i], data_x[j])
            iK[j, i] = iK[i, j]
        end
    end
    U, Δ, V = svd(iK)
    invΔ = Diagonal(1.0 ./ Δ .* (Δ .> 1e-6))
    iK[:, :] = V * invΔ * U'
end

function predict(model::GPmodel, x::State)
    xs, ys, pvec, iK = model.xs, model.ys, model.pvec, model.iK

    # Compute mu var
    kv = map(xloc -> kernel(x, xloc), xs)
    k0 = kernel(x, x)
    mu = kv' * pvec
    var = k0 - kv' * iK * kv

    # sample from gaussian
    sqrt(var) * randn(Complex{Float64}) + mu
end

