include("./setup.jl")
include("./model.jl")
using Base.Threads, LinearAlgebra, Random, Folds

function imaginarytime(model::GPmodel)
    data_x, data_y = model.data_x, model.data_y
    ψ = copy(data_y)
    @threads for i in 1:c.NData
        e = localenergy(data_x[i], model)
        ψ[i] = (1.0 - c.Δτ * e / c.NSpin) * exp(data_y[i])
    end
    data_y[:] = log.(ψ)
    v = sum(ψ) / c.NData
    data_y[:] .-= log(v)
end

function tryflip(x::State, model::GPmodel, eng::MersenneTwister)
    pos = rand(eng, collect(1:c.NSpin))
    y = predict(x, model)
    xflip_spin = copy(x.spin)
    xflip_spin[pos] *= -1
    xflip = State(xflip_spin)
    y_new = predict(xflip, model)
    x.spin[pos] *= ifelse(rand(eng) < exp(2 * real(y_new - y)), -1, 1)
    x2 = State(x.spin)
    setfield!(x, :spin, x2.spin)
end

function localenergy(x::State, model::GPmodel)
    y = predict(x, model)
    eloc = 0.0im
    for i in 1:c.NSpin
        eloc -= x.spin[i] * x.spin[i%c.NSpin+1]
        xflip_spin = copy(x.spin)
        xflip_spin[i] *= -1
        xflip = State(xflip_spin)
        y2 = predict(xflip, model)
        eloc -= c.H * exp(y2 - y)
    end
    eloc
end

function energy(model::GPmodel)
    x_mc = Vector{State}(undef, c.NMC)
    x0 = State(rand([1.0, -1.0], c.NSpin))
    for i in 1:c.NMC
        for j in 1:c.MCSkip
            eng = EngArray[threadid()]
            tryflip(x0, model, eng)
        end
        x_mc[i] = x0
    end
    ene = Folds.sum(localenergy(x, model) for x in x_mc)
    real(ene / c.NMC)
end

